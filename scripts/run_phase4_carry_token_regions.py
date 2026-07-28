#!/usr/bin/env python3
"""Localize generic increment and carry-context effects to prompt regions."""

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
from run_phase4_carry_sequence_boundary import (
    generate_chunks,
    norm_match_sequences,
    random_norm_matched_sequences,
    sequence_norms,
    subtract_sequences,
    summarize_outputs,
    value_sha256,
    verify_sha256,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.phase4_data import (
    build_phase4_carry_quartets,
    phase4_carry_sha256,
)
from open_latent_interfaces.prefill import (
    render_prefilled_chat,
    verify_decimal_digit_contract,
)


def mask_positions(
    values: tuple[torch.Tensor, ...],
    positions: list[list[int]],
) -> tuple[torch.Tensor, ...]:
    if len(values) != len(positions):
        raise ValueError("position masks and sequence counts differ")
    masked = []
    for value, selected in zip(values, positions, strict=True):
        result = torch.zeros_like(value)
        result[selected] = value[selected]
        masked.append(result)
    return tuple(masked)


def select_condition(
    layers: dict[str, dict[str, Any]],
    *,
    target: str,
    controls: tuple[str, ...],
    minimum_accuracy: float,
    minimum_advantage: float,
) -> tuple[int, bool]:
    def score(index: str) -> tuple[float, float, float, int]:
        row = layers[index]
        strongest = max(row[name]["target_tens_accuracy"] for name in controls)
        metric = row[target]
        return (
            metric["target_tens_accuracy"],
            metric["target_tens_accuracy"] - strongest,
            metric["target_full_accuracy"],
            -int(index),
        )

    selected = max(layers, key=score)
    row = layers[selected]
    metric = row[target]
    strongest = max(row[name]["target_tens_accuracy"] for name in controls)
    passes = (
        metric["target_tens_accuracy"] >= minimum_accuracy
        and metric["target_tens_accuracy"] - strongest >= minimum_advantage
        and metric["parse_rate"] == 1.0
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
        raise SystemExit(f"refusing to overwrite token-region result: {args.output}")

    config = json.loads(args.config.read_text())
    dataset_path = Path(config["dataset_config"])
    behavior_path = Path(config["behavior_result"])
    parent_path = Path(config["parent_result"])
    verify_sha256(dataset_path, config["dataset_config_sha256"])
    verify_sha256(behavior_path, config["behavior_result_sha256"])
    verify_sha256(parent_path, config["parent_result_sha256"])
    dataset_config = json.loads(dataset_path.read_text())
    behavior = json.loads(behavior_path.read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("token-region selection requires a sealed audit")
    if (
        behavior["splits"]["fit"]["complete_correct_quartets"]
        < config["minimum_eligible_fit_quartets"]
    ):
        raise SystemExit("insufficient behavior-correct fit quartets")

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
    token_ids = {
        name: [tokenizer(prompt)["input_ids"] for prompt in prompts]
        for name, prompts in rendered.items()
    }
    changed_positions = []
    context_positions = []
    token_contract = []
    for index in range(len(quartet_ids)):
        carry_base = token_ids["carry_base"][index]
        carry_increment = token_ids["carry_increment"][index]
        control_base = token_ids["control_base"][index]
        control_increment = token_ids["control_increment"][index]
        changed = [
            position
            for position, (base, increment) in enumerate(
                zip(carry_base, carry_increment, strict=True)
            )
            if base != increment
        ]
        control_changed = [
            position
            for position, (base, increment) in enumerate(
                zip(control_base, control_increment, strict=True)
            )
            if base != increment
        ]
        context = [
            position
            for position, (carry, control) in enumerate(
                zip(carry_base, control_base, strict=True)
            )
            if carry != control
        ]
        increment_context = [
            position
            for position, (carry, control) in enumerate(
                zip(carry_increment, control_increment, strict=True)
            )
            if carry != control
        ]
        if not (
            len(changed) == len(context) == 1
            and changed == control_changed
            and context == increment_context
        ):
            raise SystemExit(f"quartet {quartet_ids[index]} violates token contract")
        changed_positions.append(changed[0])
        context_positions.append(context[0])
        token_contract.append([len(carry_base), changed[0], context[0]])
    if value_sha256(token_contract) != config["token_region_contract_sha256"]:
        raise SystemExit("token-region contract hash mismatch")
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
        "changed_operand_token",
        "changed_operand_random_norm_matched",
        "carry_context_token",
        "control_context_token_norm_matched",
        "context_random_norm_matched",
        "carry_downstream_tail",
        "control_downstream_tail_norm_matched",
        "tail_random_norm_matched",
    )
    changed_masks = [[position] for position in changed_positions]
    context_masks = [[position] for position in context_positions]
    tail_masks = [
        list(range(position + 1, length))
        for position, length in zip(
            context_positions,
            (len(ids) for ids in token_ids["carry_base"]),
            strict=True,
        )
    ]
    layers = {}
    for hidden_index in config["hidden_state_indices"]:
        carry_base = states["carry_base"][hidden_index].values
        carry_delta = subtract_sequences(
            states["carry_increment"][hidden_index].values,
            carry_base,
            label="carry",
        )
        control_delta = subtract_sequences(
            states["control_increment"][hidden_index].values,
            states["control_base"][hidden_index].values,
            label="control",
        )
        changed = mask_positions(carry_delta, changed_masks)
        carry_context = mask_positions(carry_delta, context_masks)
        control_context = mask_positions(control_delta, context_masks)
        carry_tail = mask_positions(carry_delta, tail_masks)
        control_tail = mask_positions(control_delta, tail_masks)
        condition_values = {
            "base": tuple(torch.zeros_like(value) for value in carry_delta),
            "changed_operand_token": changed,
            "changed_operand_random_norm_matched": random_norm_matched_sequences(
                changed,
                sequence_norms(changed),
                seed=config["random_control_seed"] + hidden_index * 100 + 1,
            ),
            "carry_context_token": carry_context,
            "control_context_token_norm_matched": norm_match_sequences(
                control_context,
                sequence_norms(carry_context),
            ),
            "context_random_norm_matched": random_norm_matched_sequences(
                carry_context,
                sequence_norms(carry_context),
                seed=config["random_control_seed"] + hidden_index * 100 + 2,
            ),
            "carry_downstream_tail": carry_tail,
            "control_downstream_tail_norm_matched": norm_match_sequences(
                control_tail,
                sequence_norms(carry_tail),
            ),
            "tail_random_norm_matched": random_norm_matched_sequences(
                carry_tail,
                sequence_norms(carry_tail),
                seed=config["random_control_seed"] + hidden_index * 100 + 3,
            ),
        }
        layer = {}
        for condition in conditions:
            delta = condition_values[condition]
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

    rule = config["selection_rule"]
    generic_index, generic_passes = select_condition(
        layers,
        target="changed_operand_token",
        controls=("changed_operand_random_norm_matched",),
        minimum_accuracy=rule["generic_minimum_tens_accuracy"],
        minimum_advantage=rule["minimum_control_advantage"],
    )
    context_index, context_passes = select_condition(
        layers,
        target="carry_context_token",
        controls=(
            "control_context_token_norm_matched",
            "context_random_norm_matched",
        ),
        minimum_accuracy=rule["specific_minimum_tens_accuracy"],
        minimum_advantage=rule["minimum_control_advantage"],
    )
    tail_index, tail_passes = select_condition(
        layers,
        target="carry_downstream_tail",
        controls=(
            "control_downstream_tail_norm_matched",
            "tail_random_norm_matched",
        ),
        minimum_accuracy=rule["specific_minimum_tens_accuracy"],
        minimum_advantage=rule["minimum_control_advantage"],
    )
    report = {
        "schema_version": "oli.phase4-carry-token-regions/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only",
        "model": model_config,
        "dataset_sha256": observed_hash,
        "parent_result_sha256": config["parent_result_sha256"],
        "behavior_source": {
            "sha256": config["behavior_result_sha256"],
            "original_gate_passed": behavior["passes"],
            "eligible_fit_quartets": behavior["splits"]["fit"][
                "complete_correct_quartets"
            ],
        },
        "selection_quartets_sha256": config["selection_quartets_sha256"],
        "token_region_contract_sha256": config["token_region_contract_sha256"],
        "digit_token_ids": digit_token_ids,
        "hidden_state_indices": config["hidden_state_indices"],
        "conditions": list(conditions),
        "layers": layers,
        "selection": {
            "generic_changed_operand": {
                "hidden_state_index": generic_index,
                "passes": generic_passes,
            },
            "carry_context_token": {
                "hidden_state_index": context_index,
                "passes": context_passes,
            },
            "carry_downstream_tail": {
                "hidden_state_index": tail_index,
                "passes": tail_passes,
            },
        },
        "selection_rule": rule,
        "passes": {
            "generic_changed_operand": generic_passes,
            "carry_context_token": context_passes,
            "carry_downstream_tail": tail_passes,
        },
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Selection-only token-region localization. Generic operand editing "
            "and carry-specific context transport have separate conjunctive "
            "gates; neither implies a thought transcript."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
