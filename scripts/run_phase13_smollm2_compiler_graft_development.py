#!/usr/bin/env python3
"""Evaluate the SmolLM2 read-compute-compiler-write graft on development."""

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
from run_phase3_closed_loop_development import wrong_all_digits
from run_phase3_native_boundary import verify_sha256
from run_phase4_carry_sequence_boundary import value_sha256
from run_phase8_latent_graft import group_predictions
from run_phase8_operand_reader_selection import (
    flatten_states_and_labels,
    reader_metrics,
    render_and_locate,
)
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.compiler_writer import (
    PositionCompilerSpec,
    sequential_compiler_condition,
)
from open_latent_interfaces.operand_reader import NearestCentroidDigitReader
from open_latent_interfaces.phase13_data import (
    build_phase13_examples,
    phase13_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def paired_summary(
    conditions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    base = {row["example_id"]: row for row in conditions["base"]["outputs"]}
    true_results = {
        example_id: row["true_target"] for example_id, row in base.items()
    }
    base_errors = [
        example_id
        for example_id, target in true_results.items()
        if base[example_id]["parsed"] != target
    ]
    base_correct = [
        example_id
        for example_id, target in true_results.items()
        if base[example_id]["parsed"] == target
    ]
    summaries = {}
    for name, condition in conditions.items():
        if name == "base":
            continue
        outputs = {row["example_id"]: row for row in condition["outputs"]}
        recovered = sum(
            outputs[example_id]["parsed"] == true_results[example_id]
            for example_id in base_errors
        )
        preserved = sum(
            outputs[example_id]["parsed"] == true_results[example_id]
            for example_id in base_correct
        )
        harmed = len(base_correct) - preserved
        summaries[name] = {
            "base_error_count": len(base_errors),
            "base_correct_count": len(base_correct),
            "recovered_base_errors": recovered,
            "base_error_recovery": (
                recovered / len(base_errors) if base_errors else 1.0
            ),
            "preserved_base_correct": preserved,
            "base_correct_preservation": (
                preserved / len(base_correct) if base_correct else 1.0
            ),
            "harmed_base_correct": harmed,
            "net_exact_improvement": recovered - harmed,
            "net_exact_improvement_rate": (
                (recovered - harmed) / len(true_results)
            ),
        }
    return summaries


def development_gate(
    conditions: dict[str, dict[str, Any]],
    *,
    reader: dict[str, Any],
    computed_accuracy: float,
    rule: dict[str, Any],
) -> dict[str, Any]:
    paired = paired_summary(conditions)
    latent = conditions["latent_read_compute_compiler_write"]
    oracle = conditions["oracle_compute_compiler_write"]
    shuffled = conditions["shuffled_read_compute_compiler_write"]
    shuffled_random = conditions["shuffled_random_norm_matched"]
    latent_paired = paired["latent_read_compute_compiler_write"]
    random_paired = paired["random_norm_matched"]
    wrong_paired = paired["wrong_target_norm_matched"]
    base_error_count = latent_paired["base_error_count"]
    excess_random = (
        latent_paired["recovered_base_errors"]
        - random_paired["recovered_base_errors"]
    ) / max(1, base_error_count)
    excess_wrong = (
        latent_paired["recovered_base_errors"]
        - wrong_paired["recovered_base_errors"]
    ) / max(1, base_error_count)
    oracle_gap = (
        oracle["true_result_accuracy"] - latent["true_result_accuracy"]
    )
    shuffled_advantage = (
        shuffled["evaluation_target_accuracy"]
        - shuffled_random["evaluation_target_accuracy"]
    )
    checks = {
        "reader": (
            reader["pair_accuracy"] >= rule["minimum_reader_pair_accuracy"]
        ),
        "compute": (
            computed_accuracy >= rule["minimum_computed_target_accuracy"]
        ),
        "final_exact": (
            latent["true_result_accuracy"]
            >= rule["minimum_final_exact_accuracy"]
        ),
        "position_accuracy": (
            min(latent["step_true_target_accuracy"])
            >= rule["minimum_position_accuracy"]
        ),
        "oracle_gap": oracle_gap <= rule["maximum_oracle_exact_gap"],
        "base_error_recovery": (
            latent_paired["base_error_recovery"]
            >= rule["minimum_base_error_recovery"]
        ),
        "base_correct_preservation": (
            latent_paired["base_correct_preservation"]
            >= rule["minimum_base_correct_preservation"]
        ),
        "net_improvement": (
            latent_paired["net_exact_improvement_rate"]
            >= rule["minimum_net_improvement_over_base"]
        ),
        "excess_recovery_over_random": (
            excess_random
            >= rule["minimum_excess_base_error_recovery_over_random"]
        ),
        "excess_recovery_over_wrong": (
            excess_wrong
            >= rule[
                "minimum_excess_base_error_recovery_over_wrong_target"
            ]
        ),
        "wrong_target_recovery": (
            wrong_paired["base_error_recovery"]
            <= rule["maximum_wrong_target_base_error_recovery"]
        ),
        "shuffled_true_control": (
            shuffled["true_result_accuracy"]
            <= rule["maximum_shuffled_true_accuracy"]
        ),
        "shuffled_target_following": (
            shuffled["evaluation_target_accuracy"]
            >= rule["minimum_shuffled_target_accuracy"]
        ),
        "shuffled_random_target_control": (
            shuffled_random["evaluation_target_accuracy"]
            <= rule["maximum_shuffled_random_target_accuracy"]
        ),
        "shuffled_target_advantage": (
            shuffled_advantage
            >= rule["minimum_shuffled_target_advantage_over_random"]
        ),
        "parse": (
            not rule["require_parse_rate"] or latent["parse_rate"] == 1.0
        ),
        "digit_tokens": (
            not rule["require_digit_token_rate"]
            or latent["digit_token_rate"] == 1.0
        ),
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "paired_metrics": paired,
        "derived": {
            "oracle_exact_gap": oracle_gap,
            "excess_base_error_recovery_over_random": excess_random,
            "excess_base_error_recovery_over_wrong_target": excess_wrong,
            "shuffled_target_advantage_over_random": shuffled_advantage,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite compiler-graft development")

    config = json.loads(args.config.read_text())
    if str(args.output) != config["output"]:
        raise SystemExit("development output differs from frozen path")
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("development runner hash mismatch")
    for dependency, expected_hash in config["code_dependencies"].items():
        verify_sha256(Path(dependency), expected_hash)
    source_names = (
        "dataset_config",
        "onboarding_result",
        "capability_result",
        "reader_result",
        "reader_artifact",
        "leading_result",
        "prototype_result",
        "suffix_result",
    )
    paths = {name: Path(config[name]) for name in source_names}
    for name, path in paths.items():
        verify_sha256(path, config[f"{name}_sha256"])

    dataset_config = json.loads(paths["dataset_config"].read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("development may not use sealed audit data")
    onboarding = json.loads(paths["onboarding_result"].read_text())
    capability = json.loads(paths["capability_result"].read_text())
    reader_result = json.loads(paths["reader_result"].read_text())
    leading_result = json.loads(paths["leading_result"].read_text())
    prototype_result = json.loads(paths["prototype_result"].read_text())
    suffix_result = json.loads(paths["suffix_result"].read_text())
    if onboarding.get("passes") is not True:
        raise SystemExit("model onboarding did not pass")
    if capability.get("status") != "exposed_fit_measurement":
        raise SystemExit("capability result has unexpected status")
    if reader_result.get("passes") is not True:
        raise SystemExit("operand reader did not pass")
    if leading_result.get("selection", {}).get("passes") is not True:
        raise SystemExit("leading compiler did not pass")
    if prototype_result.get("passes") is not False:
        raise SystemExit("native suffix nonpass is not preserved")
    if suffix_result.get("passes") is not True:
        raise SystemExit("suffix compiler did not pass")

    examples = build_phase13_examples(
        **dataset_config["dataset"]["parameters"]
    )
    dataset_hash = phase13_sha256(examples)
    if dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 13 dataset hash mismatch")
    development = [row for row in examples if row.split == "development"]
    if value_sha256([row.example_id for row in development]) != config[
        "development_examples_sha256"
    ]:
        raise SystemExit("development example hash mismatch")
    true_targets = [row.result for row in development]
    if value_sha256(true_targets) != config["development_targets_sha256"]:
        raise SystemExit("development target hash mismatch")

    model_config = dataset_config["model"]
    if model_config != config["model"]:
        raise SystemExit("development model differs from frozen model")
    for source in (reader_result, leading_result, suffix_result):
        if source.get("model") != model_config:
            raise SystemExit("component result used a different model")
        if source.get("dataset_sha256") != dataset_hash:
            raise SystemExit("component result used a different dataset")

    leading_selection = leading_result["selection"]
    expected_specs = {
        0: {
            "hidden_state_index": leading_selection["hidden_state_index"],
            "desired_margin": leading_selection["desired_margin"],
            "norm_cap": leading_selection["norm_cap"],
        },
        1: {
            "hidden_state_index": suffix_result["hidden_state_index"],
            "desired_margin": suffix_result["positions"]["1"]["selection"][
                "desired_margin"
            ],
            "norm_cap": suffix_result["positions"]["1"]["selection"][
                "norm_cap"
            ],
        },
        2: {
            "hidden_state_index": suffix_result["hidden_state_index"],
            "desired_margin": suffix_result["positions"]["2"]["selection"][
                "desired_margin"
            ],
            "norm_cap": suffix_result["positions"]["2"]["selection"][
                "norm_cap"
            ],
        },
    }
    configured_specs = {
        int(position): spec
        for position, spec in config["position_compilers"].items()
    }
    if configured_specs != expected_specs:
        raise SystemExit("position compiler settings differ from selection")
    position_specs = {
        position: PositionCompilerSpec(**spec)
        for position, spec in configured_specs.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered, operand_positions, token_contract = render_and_locate(
        tokenizer,
        development,
        dataset_config["assistant_prefix"],
    )
    if value_sha256(rendered) != config["rendered_prompts_sha256"]:
        raise SystemExit("rendered prompt hash mismatch")
    if value_sha256(token_contract) != config[
        "development_token_contract_sha256"
    ]:
        raise SystemExit("development token contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])
    if value_sha256(digit_token_ids) != config["digit_token_ids_sha256"]:
        raise SystemExit("digit-token map hash mismatch")
    candidate_ids = torch.tensor(
        [digit_token_ids[digit] for digit in range(10)],
        dtype=torch.long,
    )

    reader_tensors = load_file(str(paths["reader_artifact"]))
    reader = NearestCentroidDigitReader(
        classes=reader_tensors["digit_classes"],
        centroids=reader_tensors["digit_centroids"],
    )
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        dtype=getattr(torch, args.dtype),
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base model parameters must remain frozen")
    capture = ActivationCapture(model, tokenizer, device=device)
    started = time.perf_counter()

    reader_index = config["reader_hidden_state_index"]
    captured = capture.capture_token_positions(
        rendered,
        operand_positions,
        hidden_state_indices=[reader_index],
        batch_size=config["base_model_batch_size"],
    )[reader_index]
    reader_states, _ = flatten_states_and_labels(
        captured.values,
        development,
    )
    grouped = group_predictions(
        reader.predict(reader_states).tolist(),
        operand_positions,
    )
    read_metrics = reader_metrics(grouped, development)
    computed_targets = [
        row["predicted_operand_a"] + row["predicted_operand_b"]
        for row in read_metrics["rows"]
    ]
    computed_correct = sum(
        actual == expected
        for actual, expected in zip(
            computed_targets,
            true_targets,
            strict=True,
        )
    )
    computed_accuracy = computed_correct / len(development)
    if any(target < 100 or target > 999 for target in computed_targets):
        raise SystemExit("decoded target outside three-digit contract")
    shuffled_targets = computed_targets[1:] + computed_targets[:1]
    wrong_targets = wrong_all_digits(true_targets)
    example_ids = [row.example_id for row in development]
    plan_cache: dict[Any, Any] = {}
    common = {
        "model": model,
        "tokenizer": tokenizer,
        "capture": capture,
        "example_ids": example_ids,
        "original_results": true_targets,
        "rendered_prompts": rendered,
        "true_targets": true_targets,
        "digit_token_ids": digit_token_ids,
        "candidate_token_ids": candidate_ids,
        "position_specs": position_specs,
        "plan_cache": plan_cache,
        "compiler_batch_size": config["compiler_batch_size"],
        "base_model_batch_size": config["base_model_batch_size"],
        "device": device,
    }
    conditions = {
        "base": sequential_compiler_condition(
            "base",
            writer_targets=true_targets,
            evaluation_targets=true_targets,
            reference_targets=None,
            random_seed=config["random_control_seed"],
            **common,
        ),
        "oracle_compute_compiler_write": sequential_compiler_condition(
            "target",
            writer_targets=true_targets,
            evaluation_targets=true_targets,
            reference_targets=None,
            random_seed=config["random_control_seed"] + 100,
            **common,
        ),
        "latent_read_compute_compiler_write": sequential_compiler_condition(
            "target",
            writer_targets=computed_targets,
            evaluation_targets=computed_targets,
            reference_targets=None,
            random_seed=config["random_control_seed"] + 200,
            **common,
        ),
        "shuffled_read_compute_compiler_write": sequential_compiler_condition(
            "target",
            writer_targets=shuffled_targets,
            evaluation_targets=shuffled_targets,
            reference_targets=None,
            random_seed=config["random_control_seed"] + 300,
            **common,
        ),
        "random_norm_matched": sequential_compiler_condition(
            "random",
            writer_targets=true_targets,
            evaluation_targets=true_targets,
            reference_targets=None,
            random_seed=config["random_control_seed"] + 400,
            **common,
        ),
        "shuffled_random_norm_matched": sequential_compiler_condition(
            "random",
            writer_targets=shuffled_targets,
            evaluation_targets=shuffled_targets,
            reference_targets=None,
            random_seed=config["random_control_seed"] + 500,
            **common,
        ),
        "wrong_target_norm_matched": sequential_compiler_condition(
            "wrong",
            writer_targets=wrong_targets,
            evaluation_targets=true_targets,
            reference_targets=true_targets,
            random_seed=config["random_control_seed"] + 600,
            **common,
        ),
    }
    gate = development_gate(
        conditions,
        reader=read_metrics,
        computed_accuracy=computed_accuracy,
        rule=config["development_rule"],
    )
    report = {
        "schema_version": "oli.phase13-smollm2-compiler-graft-development/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": model_config,
        "dataset_sha256": dataset_hash,
        "evaluation_split": "phase13-development",
        "reader": {
            "hidden_state_index": reader_index,
            "metrics": read_metrics,
        },
        "deterministic_compute": {
            "operation": "integer_addition",
            "correct": computed_correct,
            "accuracy": computed_accuracy,
            "targets": computed_targets,
        },
        "writer": {
            "family": "prompt_local_margin_compiler",
            "position_compilers": config["position_compilers"],
        },
        "conditions": conditions,
        "gate": {
            "thresholds": config["development_rule"],
            **gate,
        },
        "passes": gate["passes"],
        "source_hashes": {
            f"{name}_sha256": config[f"{name}_sha256"]
            for name in source_names
        },
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "runner_sha256": runner_hash,
        "code_dependencies": config["code_dependencies"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Exposed Phase 13 development-only SmolLM2 operand read, "
            "deterministic host addition, and sequential prompt-local "
            "compiler write. No audit or model-general claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
