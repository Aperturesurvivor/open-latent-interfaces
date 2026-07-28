#!/usr/bin/env python3
"""Map causal carry computation with full-prompt residual interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.interventions import intervened_generate_sequence
from open_latent_interfaces.phase4_data import (
    build_phase4_carry_quartets,
    phase4_carry_sha256,
)
from open_latent_interfaces.prefill import (
    render_prefilled_chat,
    verify_decimal_digit_contract,
)


def verify_sha256(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"hash mismatch for {path}: {observed} != {expected}")


def value_sha256(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def subtract_sequences(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> tuple[torch.Tensor, ...]:
    if len(left) != len(right):
        raise ValueError(f"{label} sequence counts differ")
    differences = []
    for index, (left_value, right_value) in enumerate(
        zip(left, right, strict=True)
    ):
        if left_value.shape != right_value.shape:
            raise ValueError(
                f"{label} quartet {index} is not token-aligned: "
                f"{left_value.shape} != {right_value.shape}"
            )
        differences.append(left_value - right_value)
    return tuple(differences)


def sequence_norms(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.tensor([value.float().norm().item() for value in values])


def norm_match_sequences(
    values: tuple[torch.Tensor, ...],
    target_norms: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    observed = sequence_norms(values).clamp_min(1e-12)
    return tuple(
        value * (target / current)
        for value, target, current in zip(
            values,
            target_norms,
            observed,
            strict=True,
        )
    )


def random_norm_matched_sequences(
    templates: tuple[torch.Tensor, ...],
    target_norms: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_values = tuple(
        torch.randn(
            template.shape,
            dtype=torch.float32,
            generator=generator,
        )
        for template in templates
    )
    return norm_match_sequences(random_values, target_norms)


def pad_sequence_deltas(
    tokenizer: Any,
    prompts: list[str],
    values: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    finally:
        tokenizer.padding_side = previous_padding_side
    if len(values) != len(prompts):
        raise ValueError("one sequence intervention is required per prompt")
    widths = {value.shape[1] for value in values}
    if len(widths) != 1:
        raise ValueError("sequence intervention widths differ")
    padded = torch.zeros(
        len(prompts),
        encoded["input_ids"].shape[1],
        next(iter(widths)),
        dtype=torch.float32,
    )
    active = encoded["attention_mask"].bool()
    for row, value in enumerate(values):
        if int(active[row].sum()) != value.shape[0]:
            raise ValueError("captured sequence does not match re-tokenized prompt")
        padded[row, active[row]] = value
    return padded


def generate_chunks(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    values: tuple[torch.Tensor, ...],
    *,
    hidden_state_index: int,
    batch_size: int,
    device: torch.device,
) -> list[str]:
    responses = []
    for start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[start : start + batch_size]
        chunk_values = values[start : start + batch_size]
        responses.extend(
            intervened_generate_sequence(
                model,
                tokenizer,
                chunk_prompts,
                hidden_state_index=hidden_state_index,
                deltas=pad_sequence_deltas(
                    tokenizer,
                    chunk_prompts,
                    chunk_values,
                ),
                device=device,
                max_new_tokens=3,
            )
        )
    return responses


def summarize_outputs(
    responses: list[str],
    base_rows: list[Any],
    target_rows: list[Any],
    delta: tuple[torch.Tensor, ...],
    base_states: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    parsed = [parse_first_integer(response) for response in responses]
    target_correct = sum(
        value == target.result
        for value, target in zip(parsed, target_rows, strict=True)
    )
    base_correct = sum(
        value == base.result
        for value, base in zip(parsed, base_rows, strict=True)
    )
    position_correct = [
        sum(
            value is not None
            and len(str(value)) == 3
            and str(value)[position] == str(target.result)[position]
            for value, target in zip(parsed, target_rows, strict=True)
        )
        for position in range(3)
    ]
    relative_norms = sequence_norms(delta) / sequence_norms(base_states)
    return {
        "n": len(responses),
        "target_full_correct": target_correct,
        "target_full_accuracy": target_correct / len(responses),
        "base_full_correct": base_correct,
        "base_full_accuracy": base_correct / len(responses),
        "target_position_correct": position_correct,
        "target_position_accuracy": [
            count / len(responses) for count in position_correct
        ],
        "target_tens_correct": position_correct[1],
        "target_tens_accuracy": position_correct[1] / len(responses),
        "parse_count": sum(value is not None for value in parsed),
        "parse_rate": sum(value is not None for value in parsed) / len(responses),
        "mean_relative_frobenius_norm": float(relative_norms.mean()),
        "maximum_relative_frobenius_norm": float(relative_norms.max()),
        "outputs": [
            {
                "quartet_id": base.quartet_id,
                "base_result": base.result,
                "target_result": target.result,
                "response": response,
                "parsed": value,
            }
            for base, target, response, value in zip(
                base_rows,
                target_rows,
                responses,
                parsed,
                strict=True,
            )
        ],
    }


def select_layer(
    layers: dict[str, dict[str, Any]],
    *,
    targeted_condition: str,
    control_conditions: tuple[str, ...],
    minimum_tens_accuracy: float,
    minimum_control_advantage: float,
    maximum_relative_norm: float,
) -> tuple[int, bool]:
    def score(index: str) -> tuple[float, float, float, float, int]:
        rows = layers[index]
        target = rows[targeted_condition]
        control = max(
            rows[name]["target_tens_accuracy"] for name in control_conditions
        )
        return (
            target["target_tens_accuracy"],
            target["target_tens_accuracy"] - control,
            target["target_full_accuracy"],
            -target["mean_relative_frobenius_norm"],
            -int(index),
        )

    selected = max(layers, key=score)
    rows = layers[selected]
    target = rows[targeted_condition]
    strongest_control = max(
        rows[name]["target_tens_accuracy"] for name in control_conditions
    )
    passes = (
        target["target_tens_accuracy"] >= minimum_tens_accuracy
        and target["target_tens_accuracy"] - strongest_control
        >= minimum_control_advantage
        and target["mean_relative_frobenius_norm"] <= maximum_relative_norm
        and target["parse_rate"] == 1.0
    )
    return int(selected), passes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite sequence result: {args.output}")

    config = json.loads(args.config.read_text())
    dataset_path = Path(config["dataset_config"])
    behavior_path = Path(config["behavior_result"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    verify_sha256(behavior_path, config["behavior_result_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    behavior = json.loads(behavior_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("carry sequence selection requires a sealed audit")
    if (
        behavior["splits"]["fit"]["complete_correct_quartets"]
        < config["minimum_eligible_fit_quartets"]
    ):
        raise SystemExit("insufficient behavior-correct fit quartets")
    if not all(
        behavior["splits"][split]["passes"]
        for split in ("selection", "development")
    ):
        raise SystemExit("selection or development behavior gate failed")

    examples = build_phase4_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_hash = phase4_carry_sha256(examples)
    if observed_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 4 dataset hash mismatch")
    selection = [example for example in examples if example.split == "selection"]
    quartet_ids = sorted({example.quartet_id for example in selection})
    if value_sha256(quartet_ids) != config["selection_quartets_sha256"]:
        raise SystemExit("selection quartet hash mismatch")
    by_quartet = {
        quartet_id: {
            row.variant: row
            for row in selection
            if row.quartet_id == quartet_id
        }
        for quartet_id in quartet_ids
    }
    variant_names = (
        "carry_base",
        "carry_increment",
        "control_base",
        "control_increment",
    )
    variants = {
        name: [by_quartet[quartet_id][name] for quartet_id in quartet_ids]
        for name in variant_names
    }

    device = torch.device(args.device)
    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = {
        name: [
            render_prefilled_chat(
                tokenizer,
                example.prompt,
                assistant_prefix=dataset_config["assistant_prefix"],
            )
            for example in rows
        ]
        for name, rows in variants.items()
    }
    token_lengths = [
        [
            len(tokenizer(rendered[name][index])["input_ids"])
            for name in variant_names
        ]
        for index in range(len(quartet_ids))
    ]
    if value_sha256(token_lengths) != config["selection_token_lengths_sha256"]:
        raise SystemExit("selection token-length contract mismatch")
    if any(len(set(lengths)) != 1 for lengths in token_lengths):
        raise SystemExit("matched quartet prompts are not token-aligned")
    digit_token_ids = verify_decimal_digit_contract(
        tokenizer,
        rendered["carry_base"][0],
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()

    states = {
        name: capture.capture_sequences(
            prompts,
            hidden_state_indices=config["hidden_state_indices"],
            batch_size=config["base_model_batch_size"],
        )
        for name, prompts in rendered.items()
    }
    conditions = (
        "base",
        "carry_full_sequence",
        "carry_difference_in_differences_norm_matched",
        "control_increment_norm_matched",
        "random_norm_matched",
    )
    layers = {}
    for hidden_index in config["hidden_state_indices"]:
        carry_base = states["carry_base"][hidden_index].values
        carry_increment = states["carry_increment"][hidden_index].values
        control_base = states["control_base"][hidden_index].values
        control_increment = states["control_increment"][hidden_index].values
        carry_delta = subtract_sequences(
            carry_increment,
            carry_base,
            label="carry",
        )
        control_delta = subtract_sequences(
            control_increment,
            control_base,
            label="control",
        )
        did_delta = subtract_sequences(
            carry_delta,
            control_delta,
            label="difference-in-differences",
        )
        carry_norms = sequence_norms(carry_delta)
        layer = {}
        for condition_index, condition in enumerate(conditions):
            if condition == "base":
                delta = tuple(torch.zeros_like(value) for value in carry_delta)
            elif condition == "carry_full_sequence":
                delta = carry_delta
            elif condition == "carry_difference_in_differences_norm_matched":
                delta = norm_match_sequences(did_delta, carry_norms)
            elif condition == "control_increment_norm_matched":
                delta = norm_match_sequences(control_delta, carry_norms)
            else:
                delta = random_norm_matched_sequences(
                    carry_delta,
                    carry_norms,
                    seed=(
                        config["random_control_seed"]
                        + hidden_index * 10
                        + condition_index
                    ),
                )
            responses = generate_chunks(
                model,
                tokenizer,
                rendered["carry_base"],
                delta,
                hidden_state_index=hidden_index,
                batch_size=config["base_model_batch_size"],
                device=device,
            )
            layer[condition] = summarize_outputs(
                responses,
                variants["carry_base"],
                variants["carry_increment"],
                delta,
                carry_base,
            )
        layers[str(hidden_index)] = layer

    generic_controls = (
        "control_increment_norm_matched",
        "random_norm_matched",
    )
    selected_full, full_passes = select_layer(
        layers,
        targeted_condition="carry_full_sequence",
        control_conditions=generic_controls,
        minimum_tens_accuracy=config["selection_rule"][
            "full_minimum_tens_accuracy"
        ],
        minimum_control_advantage=config["selection_rule"][
            "minimum_control_advantage"
        ],
        maximum_relative_norm=config["selection_rule"][
            "maximum_relative_norm"
        ],
    )
    selected_did, did_passes = select_layer(
        layers,
        targeted_condition="carry_difference_in_differences_norm_matched",
        control_conditions=generic_controls,
        minimum_tens_accuracy=config["selection_rule"][
            "did_minimum_tens_accuracy"
        ],
        minimum_control_advantage=config["selection_rule"][
            "minimum_control_advantage"
        ],
        maximum_relative_norm=config["selection_rule"][
            "maximum_relative_norm"
        ],
    )
    report = {
        "schema_version": "oli.phase4-carry-sequence-boundary/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": observed_hash,
        "behavior_source": {
            "path": str(behavior_path),
            "sha256": config["behavior_result_sha256"],
            "original_gate_passed": behavior["passes"],
            "eligible_fit_quartets": behavior["splits"]["fit"][
                "complete_correct_quartets"
            ],
        },
        "selection_quartets_sha256": config["selection_quartets_sha256"],
        "selection_token_lengths_sha256": config[
            "selection_token_lengths_sha256"
        ],
        "digit_token_ids": digit_token_ids,
        "hidden_state_indices": config["hidden_state_indices"],
        "conditions": list(conditions),
        "layers": layers,
        "selection": {
            "full_sequence": {
                "hidden_state_index": selected_full,
                "passes": full_passes,
            },
            "difference_in_differences": {
                "hidden_state_index": selected_did,
                "passes": did_passes,
            },
        },
        "selection_rule": config["selection_rule"],
        "passes": full_passes and did_passes,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only full-prompt carry transport. A full matched-pair "
            "delta is a causal upper bound, while carry specificity requires "
            "the difference-in-differences gate and matched controls."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
