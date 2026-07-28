#!/usr/bin/env python3
"""Run the one-shot pair- and template-disjoint SmolLM2 compiler-graft audit."""

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
from run_phase13_smollm2_compiler_graft_development import development_gate
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.compiler_writer import (
    PositionCompilerSpec,
    sequential_compiler_condition,
)
from open_latent_interfaces.operand_reader import NearestCentroidDigitReader
from open_latent_interfaces.phase13_audit_data import (
    PHASE13_AUDIT_TEMPLATES,
    build_phase13_audit,
    phase13_audit_sha256,
    prior_dataset_hashes,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def template_sha256(templates: tuple[str, ...]) -> str:
    encoded = json.dumps(templates, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_pair_sha256(examples: list[Any]) -> str:
    pairs = sorted(sorted((row.operand_a, row.operand_b)) for row in examples)
    encoded = json.dumps(pairs, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify every frozen input without loading the model or auditing",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite one-shot audit result")

    config = json.loads(args.config.read_text())
    if config.get("audit_authorized") is not True:
        raise SystemExit("audit is sealed")
    if config.get("maximum_audit_runs") != 1:
        raise SystemExit("audit config must authorize exactly one run")
    if str(args.output) != config["audit_output"]:
        raise SystemExit("audit output differs from frozen path")
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("audit runner hash mismatch")
    for dependency, expected_hash in config["code_dependencies"].items():
        verify_sha256(Path(dependency), expected_hash)

    source_names = (
        "audit_dataset_config",
        "audit_dataset_generator",
        "onboarding_result",
        "capability_result",
        "reader_result",
        "reader_artifact",
        "leading_result",
        "prototype_result",
        "suffix_result",
        "development_config",
        "development_result",
    )
    paths = {name: Path(config[name]) for name in source_names}
    for name, path in paths.items():
        verify_sha256(path, config[f"{name}_sha256"])

    dataset_config = json.loads(paths["audit_dataset_config"].read_text())
    onboarding = json.loads(paths["onboarding_result"].read_text())
    capability = json.loads(paths["capability_result"].read_text())
    reader_result = json.loads(paths["reader_result"].read_text())
    leading_result = json.loads(paths["leading_result"].read_text())
    prototype_result = json.loads(paths["prototype_result"].read_text())
    suffix_result = json.loads(paths["suffix_result"].read_text())
    development_config = json.loads(paths["development_config"].read_text())
    development_result = json.loads(paths["development_result"].read_text())

    if dataset_config.get("audit_authorized", True):
        raise SystemExit("audit dataset was not frozen sealed")
    if dataset_config["dataset"]["generator"] != config["audit_dataset_generator"]:
        raise SystemExit("audit dataset generator path mismatch")
    if dataset_config["dataset"]["generator_sha256"] != config["audit_dataset_generator_sha256"]:
        raise SystemExit("audit dataset generator hash mismatch")
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
    if development_result.get("passes") is not True:
        raise SystemExit("development result did not pass")
    if development_result["config_sha256"] != config["development_config_sha256"]:
        raise SystemExit("development result/config mismatch")
    if development_result["runner_sha256"] != development_config["runner_sha256"]:
        raise SystemExit("development result/runner mismatch")

    locked_fields = (
        "model",
        "base_model_batch_size",
        "compiler_batch_size",
        "reader_hidden_state_index",
        "position_compilers",
        "onboarding_result_sha256",
        "capability_result_sha256",
        "reader_result_sha256",
        "reader_artifact_sha256",
        "leading_result_sha256",
        "prototype_result_sha256",
        "suffix_result_sha256",
    )
    for field in locked_fields:
        if config[field] != development_config[field]:
            raise SystemExit(f"audit changed development component: {field}")
    if config["audit_rule"] != development_config["development_rule"]:
        raise SystemExit("audit gate differs from passing development gate")
    for dependency, expected_hash in development_config["code_dependencies"].items():
        if config["code_dependencies"].get(dependency) != expected_hash:
            raise SystemExit(f"audit changed development code: {dependency}")

    leading_selection = leading_result["selection"]
    expected_specs = {
        0: {
            "hidden_state_index": leading_selection["hidden_state_index"],
            "desired_margin": leading_selection["desired_margin"],
            "norm_cap": leading_selection["norm_cap"],
        },
        1: {
            "hidden_state_index": suffix_result["hidden_state_index"],
            "desired_margin": suffix_result["positions"]["1"]["selection"]["desired_margin"],
            "norm_cap": suffix_result["positions"]["1"]["selection"]["norm_cap"],
        },
        2: {
            "hidden_state_index": suffix_result["hidden_state_index"],
            "desired_margin": suffix_result["positions"]["2"]["selection"]["desired_margin"],
            "norm_cap": suffix_result["positions"]["2"]["selection"]["norm_cap"],
        },
    }
    configured_specs = {
        int(position): spec for position, spec in config["position_compilers"].items()
    }
    if configured_specs != expected_specs:
        raise SystemExit("position compiler settings differ from selection")
    position_specs = {
        position: PositionCompilerSpec(**spec) for position, spec in configured_specs.items()
    }

    examples = build_phase13_audit(**dataset_config["dataset"]["parameters"])
    dataset_hash = phase13_audit_sha256(examples)
    if dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("fresh audit dataset hash mismatch")
    if value_sha256([row.example_id for row in examples]) != config["audit_examples_sha256"]:
        raise SystemExit("audit example ID hash mismatch")
    if canonical_pair_sha256(examples) != dataset_config["dataset"]["canonical_pairs_sha256"]:
        raise SystemExit("audit canonical-pair hash mismatch")
    if prior_dataset_hashes() != dataset_config["dataset"]["prior_dataset_hashes"]:
        raise SystemExit("prior dataset universe changed")
    if template_sha256(PHASE13_AUDIT_TEMPLATES) != dataset_config["dataset"]["template_sha256"]:
        raise SystemExit("audit template hash mismatch")
    true_targets = [row.result for row in examples]
    if value_sha256(true_targets) != config["audit_targets_sha256"]:
        raise SystemExit("audit target hash mismatch")

    model_config = dataset_config["model"]
    if model_config != config["model"]:
        raise SystemExit("audit model differs from frozen model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered, operand_positions, token_contract = render_and_locate(
        tokenizer,
        examples,
        dataset_config["assistant_prefix"],
    )
    if value_sha256(rendered) != config["audit_rendered_prompts_sha256"]:
        raise SystemExit("audit rendered-prompt hash mismatch")
    if (
        value_sha256([list(row) for row in operand_positions])
        != config["audit_operand_positions_sha256"]
    ):
        raise SystemExit("audit operand-position hash mismatch")
    if value_sha256(token_contract) != config["audit_token_contract_sha256"]:
        raise SystemExit("audit token contract mismatch")
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
    if args.preflight_only:
        print("audit preflight passed; no model evaluation was performed")
        return

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
    reader_states, _ = flatten_states_and_labels(captured.values, examples)
    grouped = group_predictions(
        reader.predict(reader_states).tolist(),
        operand_positions,
    )
    read_metrics = reader_metrics(grouped, examples)
    computed_targets = [
        row["predicted_operand_a"] + row["predicted_operand_b"] for row in read_metrics["rows"]
    ]
    computed_correct = sum(
        actual == expected
        for actual, expected in zip(
            computed_targets,
            true_targets,
            strict=True,
        )
    )
    computed_accuracy = computed_correct / len(examples)
    if any(target < 100 or target > 999 for target in computed_targets):
        raise SystemExit("decoded target outside three-digit contract")
    shuffled_targets = computed_targets[1:] + computed_targets[:1]
    wrong_targets = wrong_all_digits(true_targets)
    example_ids = [row.example_id for row in examples]
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
        rule=config["audit_rule"],
    )
    report = {
        "schema_version": ("oli.phase13-smollm2-compiler-graft-audit/v1"),
        "created_at": datetime.now(UTC).isoformat(),
        "status": "one_shot_audit",
        "audit_runs": 1,
        "audit_run": {
            "authorized_maximum": 1,
            "observed": 1,
        },
        "model": model_config,
        "dataset": {
            "sha256": dataset_hash,
            "examples": len(examples),
            "example_ids_sha256": config["audit_examples_sha256"],
            "canonical_pairs_sha256": dataset_config["dataset"]["canonical_pairs_sha256"],
            "prior_dataset_hashes": dataset_config["dataset"]["prior_dataset_hashes"],
        },
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
            "thresholds": config["audit_rule"],
            **gate,
        },
        "passes": gate["passes"],
        "source_hashes": {f"{name}_sha256": config[f"{name}_sha256"] for name in source_names},
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
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
