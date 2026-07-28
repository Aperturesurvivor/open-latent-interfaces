#!/usr/bin/env python3
"""Evaluate the complete donor-free Phi coordinate controller in closed loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from run_phase3_native_boundary import (
    predict_with_delta,
    render_examples,
    value_list_sha256,
    verify_sha256,
)
from run_phase3_prototype_selection import prototype_delta, scale_and_gate
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.evaluation import norm_match
from open_latent_interfaces.phase2_data import balanced_counterfactual_results
from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def wrong_all_digits(results: list[int]) -> list[int]:
    wrong = []
    for result in results:
        digits = [int(digit) for digit in str(result)]
        wrong.append(
            int(
                f"{digits[0] % 9 + 1}"
                f"{(digits[1] + 1) % 10}"
                f"{(digits[2] + 1) % 10}"
            )
        )
    return wrong


def evaluate_condition(
    condition: str,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    *,
    examples: list[Any],
    targets: list[int],
    rendered_prompts: list[str],
    bases: dict[int, torch.Tensor],
    prototypes: dict[int, torch.Tensor],
    digit_token_ids: dict[int, int],
    config: dict[str, Any],
    device: torch.device,
    condition_index: int,
) -> dict[str, Any]:
    originals = [example.result for example in examples]
    wrong_targets = wrong_all_digits(targets)
    shuffled_targets = targets[1:] + targets[:1]
    prefixes = ["" for _ in examples]
    predicted_ids: list[list[int]] = [[] for _ in examples]
    step_target_correct = []
    step_original_correct = []
    norms = []
    gate_rates = []
    for position in range(3):
        hidden_index = config["hidden_state_indices"][str(position)]
        prompts = [
            prompt + prefix
            for prompt, prefix in zip(rendered_prompts, prefixes, strict=True)
        ]
        states = capture.capture_last_token(
            prompts,
            hidden_state_indices=[hidden_index],
            batch_size=config["base_model_batch_size"],
        )[hidden_index].values.float()
        base_logits = capture.next_token_logits(
            prompts,
            batch_size=config["base_model_batch_size"],
        )
        target_ids = torch.tensor(
            [digit_token_ids[int(str(value)[position])] for value in targets]
        )
        raw_target = prototype_delta(
            states,
            targets,
            prototypes[position],
            bases[position],
            position=position,
        )
        targeted, targeted_gate = scale_and_gate(
            raw_target,
            states,
            base_logits,
            target_ids,
            scale=config["scales"][str(position)],
            norm_cap=config["norm_cap"],
        )
        if condition == "base":
            delta = torch.zeros_like(targeted)
            gate = targeted_gate
        elif condition == "donor_free_targeted":
            delta = targeted
            gate = targeted_gate
        elif condition == "identity_hard_gated":
            identity_ids = torch.tensor(
                [
                    digit_token_ids[int(str(value)[position])]
                    for value in originals
                ]
            )
            raw_identity = prototype_delta(
                states,
                originals,
                prototypes[position],
                bases[position],
                position=position,
            )
            delta, gate = scale_and_gate(
                raw_identity,
                states,
                base_logits,
                identity_ids,
                scale=config["scales"][str(position)],
                norm_cap=config["norm_cap"],
            )
        elif condition == "wrong_digit_norm_matched":
            raw_wrong = prototype_delta(
                states,
                wrong_targets,
                prototypes[position],
                bases[position],
                position=position,
            )
            delta = norm_match(raw_wrong, targeted.norm(dim=1))
            gate = torch.zeros(len(examples), dtype=torch.bool)
        elif condition == "shuffled_target_norm_matched":
            raw_shuffled = prototype_delta(
                states,
                shuffled_targets,
                prototypes[position],
                bases[position],
                position=position,
            )
            delta = norm_match(raw_shuffled, targeted.norm(dim=1))
            gate = torch.zeros(len(examples), dtype=torch.bool)
        else:
            rank = bases[position].shape[0]
            generator = torch.Generator().manual_seed(
                config["random_control_seed"]
                + condition_index * 100
                + position
            )
            coefficients = torch.randn(
                (len(examples), rank),
                generator=generator,
            )
            random_delta = coefficients @ bases[position]
            delta = norm_match(random_delta, targeted.norm(dim=1))
            gate = torch.zeros(len(examples), dtype=torch.bool)
        logits = predict_with_delta(
            model,
            tokenizer,
            prompts,
            delta,
            hidden_state_index=hidden_index,
            batch_size=config["base_model_batch_size"],
            device=device,
        )
        next_ids = logits.argmax(dim=1).tolist()
        target_expected = [
            digit_token_ids[int(str(value)[position])] for value in targets
        ]
        original_expected = [
            digit_token_ids[int(str(value)[position])] for value in originals
        ]
        step_target_correct.append(
            sum(
                actual == expected
                for actual, expected in zip(
                    next_ids,
                    target_expected,
                    strict=True,
                )
            )
        )
        step_original_correct.append(
            sum(
                actual == expected
                for actual, expected in zip(
                    next_ids,
                    original_expected,
                    strict=True,
                )
            )
        )
        norms.append(float((delta.norm(dim=1) / states.norm(dim=1)).mean()))
        gate_rates.append(float(gate.float().mean()))
        for index, token_id in enumerate(next_ids):
            predicted_ids[index].append(int(token_id))
            prefixes[index] += tokenizer.decode([int(token_id)])

    text = [tokenizer.decode(token_ids) for token_ids in predicted_ids]
    parsed = [parse_first_integer(value) for value in text]
    target_full_correct = sum(
        value == target for value, target in zip(parsed, targets, strict=True)
    )
    original_full_correct = sum(
        value == original
        for value, original in zip(parsed, originals, strict=True)
    )
    parse_count = sum(value is not None for value in parsed)
    digit_ids = set(digit_token_ids.values())
    digit_token_count = sum(
        token_id in digit_ids for row in predicted_ids for token_id in row
    )
    return {
        "n": len(examples),
        "step_target_correct": step_target_correct,
        "step_target_accuracy": [
            count / len(examples) for count in step_target_correct
        ],
        "step_original_correct": step_original_correct,
        "step_original_accuracy": [
            count / len(examples) for count in step_original_correct
        ],
        "target_full_result_correct": target_full_correct,
        "target_full_result_accuracy": target_full_correct / len(examples),
        "original_full_result_correct": original_full_correct,
        "original_full_result_accuracy": original_full_correct / len(examples),
        "parse_count": parse_count,
        "parse_rate": parse_count / len(examples),
        "digit_token_count": digit_token_count,
        "digit_token_rate": digit_token_count / (len(examples) * 3),
        "mean_relative_norm_by_step": norms,
        "hard_gate_rate_by_step": gate_rates,
        "outputs": [
            {
                "example_id": example.example_id,
                "original_result": originals[index],
                "target_result": targets[index],
                "generated_text": text[index],
                "parsed": parsed[index],
                "predicted_token_ids": predicted_ids[index],
            }
            for index, example in enumerate(examples)
        ],
    }


def advancement_gate(
    conditions: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    target = conditions["donor_free_targeted"]
    identity = conditions["identity_hard_gated"]
    controls = [
        conditions[name]
        for name in (
            "wrong_digit_norm_matched",
            "shuffled_target_norm_matched",
            "random_subspace_norm_matched",
        )
    ]
    n = target["n"]
    minimum_exact = math.ceil(config["gate"]["minimum_exact_accuracy"] * n)
    minimum_position = math.ceil(
        config["gate"]["minimum_position_accuracy"] * n
    )
    minimum_identity = math.ceil(
        config["gate"]["minimum_identity_accuracy"] * n
    )
    minimum_advantage = math.ceil(
        config["gate"]["minimum_control_advantage"] * n
    )
    strongest_control = max(
        row["target_full_result_correct"] for row in controls
    )
    checks = {
        "exact": target["target_full_result_correct"] >= minimum_exact,
        "positions": all(
            count >= minimum_position for count in target["step_target_correct"]
        ),
        "control_advantage": (
            target["target_full_result_correct"] - strongest_control
            >= minimum_advantage
        ),
        "identity": (
            identity["original_full_result_correct"] >= minimum_identity
        ),
        "norm": all(
            value <= config["norm_cap"]
            for value in target["mean_relative_norm_by_step"]
        ),
        "parse": target["parse_count"] == n,
        "digit_tokens": target["digit_token_count"] == n * 3,
    }
    return all(checks.values()), {
        "checks": checks,
        "required_counts": {
            "exact": minimum_exact,
            "each_position": minimum_position,
            "identity": minimum_identity,
            "control_advantage": minimum_advantage,
        },
        "strongest_control_exact": strongest_control,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite development result: {args.output}")
    config = json.loads(args.config.read_text())
    source_paths = {
        "dataset": Path(config["dataset_config"]),
        "basis": Path(config["basis"]),
        "suffix_result": Path(config["suffix_result"]),
        "suffix_prototype": Path(config["suffix_prototype"]),
        "leading_result": Path(config["leading_result"]),
        "leading_prototype": Path(config["leading_prototype"]),
    }
    for name, path in source_paths.items():
        verify_sha256(path, config[f"{name}_sha256"])
    dataset_config = json.loads(source_paths["dataset"].read_text())
    suffix_result = json.loads(source_paths["suffix_result"].read_text())
    leading_result = json.loads(source_paths["leading_result"].read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("development runner requires a sealed audit")
    if not leading_result["passes"]:
        raise SystemExit("leading prototype source did not pass")
    if not all(
        suffix_result["positions"][str(position)]["passes"]
        for position in (1, 2)
    ):
        raise SystemExit("suffix prototype sources did not pass")
    if leading_result["selected_rank"] != 32:
        raise SystemExit("unexpected leading rank")
    if suffix_result["positions"]["1"]["rank"] != 32:
        raise SystemExit("unexpected suffix rank")

    examples = build_phase3_additions(**dataset_config["dataset"]["parameters"])
    observed_hash = phase3_addition_sha256(examples)
    if observed_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 3 dataset hash mismatch")
    development = [
        example for example in examples if example.split == "development"
    ]
    targets = balanced_counterfactual_results(development)
    if value_list_sha256(targets) != config["development_targets_sha256"]:
        raise SystemExit("development target hash mismatch")

    basis_artifact = load_file(str(source_paths["basis"]))
    suffix_artifact = load_file(str(source_paths["suffix_prototype"]))
    leading_artifact = load_file(str(source_paths["leading_prototype"]))
    bases = {
        0: basis_artifact["leading_basis"][:32].float(),
        1: basis_artifact["suffix_basis"][:32].float(),
        2: basis_artifact["suffix_basis"][:32].float(),
    }
    prototypes = {
        0: leading_artifact["leading_digit"].float(),
        1: suffix_artifact["position_1_digit"].float(),
        2: suffix_artifact["position_2_digit"].float(),
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
    rendered = render_examples(
        tokenizer,
        development,
        assistant_prefix=dataset_config["assistant_prefix"],
    )
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])
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

    condition_names = (
        "base",
        "donor_free_targeted",
        "identity_hard_gated",
        "wrong_digit_norm_matched",
        "shuffled_target_norm_matched",
        "random_subspace_norm_matched",
    )
    conditions = {
        condition: evaluate_condition(
            condition,
            model,
            tokenizer,
            capture,
            examples=development,
            targets=targets,
            rendered_prompts=rendered,
            bases=bases,
            prototypes=prototypes,
            digit_token_ids=digit_token_ids,
            config=config,
            device=device,
            condition_index=index,
        )
        for index, condition in enumerate(condition_names)
    }
    passes, gate_details = advancement_gate(conditions, config)
    report = {
        "schema_version": "oli.phase3-closed-loop-development/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": model_config,
        "dataset": {
            "sha256": observed_hash,
            "split": "development",
            "examples": len(development),
        },
        "sources": {
            f"{name}_sha256": config[f"{name}_sha256"]
            for name in (
                "basis",
                "suffix_result",
                "suffix_prototype",
                "leading_result",
                "leading_prototype",
            )
        },
        "controller": {
            "hidden_state_indices": config["hidden_state_indices"],
            "ranks": {"0": 32, "1": 32, "2": 32},
            "scales": config["scales"],
            "norm_cap": config["norm_cap"],
            "hard_gate": "exact zero delta when base argmax is requested digit",
        },
        "development_targets_sha256": config["development_targets_sha256"],
        "conditions": conditions,
        "gate": {
            "thresholds": config["gate"],
            **gate_details,
        },
        "passes": passes,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "First closed-loop development evaluation of the complete "
            "donor-free Phi controller. Audit remains sealed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
