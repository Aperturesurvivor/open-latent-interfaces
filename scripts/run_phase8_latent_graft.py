#!/usr/bin/env python3
"""Evaluate Phi latent-read, deterministic-compute, native-write graft."""

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
from run_phase3_closed_loop_development import evaluate_condition
from run_phase4_carry_sequence_boundary import value_sha256, verify_sha256
from run_phase8_operand_reader_selection import (
    flatten_states_and_labels,
    reader_metrics,
    render_and_locate,
)
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.native_coordinates import NativeCoordinateManifest
from open_latent_interfaces.operand_reader import NearestCentroidDigitReader
from open_latent_interfaces.phase7_data import (
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def group_predictions(
    flat_predictions: list[int],
    positions: list[tuple[int, ...]],
) -> list[list[int]]:
    grouped = []
    cursor = 0
    for row in positions:
        width = len(row)
        grouped.append(flat_predictions[cursor : cursor + width])
        cursor += width
    if cursor != len(flat_predictions):
        raise ValueError("reader predictions were not fully consumed")
    return grouped


def true_result_metrics(
    condition: dict[str, Any],
    true_results: list[int],
) -> dict[str, Any]:
    correct = sum(
        row["parsed"] == target
        for row, target in zip(
            condition["outputs"],
            true_results,
            strict=True,
        )
    )
    return {
        "true_result_correct": correct,
        "true_result_accuracy": correct / len(true_results),
    }


def load_writer_components(
    manifest: NativeCoordinateManifest,
    root: Path,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    bases = {}
    prototypes = {}
    hidden_indices = {}
    scales = {}
    norm_caps = set()
    for answer_position, position in manifest.positions.items():
        basis_reference = position.basis or manifest.basis
        if basis_reference is None:
            raise ValueError(f"writer position {answer_position} has no basis")
        basis_tensors = load_file(str(root / basis_reference.path))
        prototype_tensors = load_file(
            str(root / position.prototypes.path)
        )
        bases[answer_position] = basis_tensors[basis_reference.key][
            : position.rank
        ].float()
        prototypes[answer_position] = prototype_tensors[
            position.prototypes.key
        ].float()
        hidden_indices[str(answer_position)] = position.hidden_state_index
        scales[str(answer_position)] = position.scale
        norm_caps.add(position.norm_cap)
    if len(norm_caps) != 1:
        raise ValueError("shared evaluation engine requires one norm cap")
    return (
        bases,
        prototypes,
        {
            "hidden_state_indices": hidden_indices,
            "scales": scales,
            "norm_cap": norm_caps.pop(),
        },
    )


def development_gate(
    conditions: dict[str, dict[str, Any]],
    *,
    reader: dict[str, Any],
    computed_target_accuracy: float,
    rule: dict[str, Any],
) -> dict[str, Any]:
    n = conditions["latent_read_compute_write"]["n"]
    latent_correct = conditions["latent_read_compute_write"][
        "true_result_correct"
    ]
    oracle_correct = conditions["oracle_compute_native_write"][
        "true_result_correct"
    ]
    strongest_control = max(
        conditions[name]["true_result_correct"]
        for name in (
            "shuffled_read_compute_write",
            "random_native_subspace",
        )
    )
    required_final = math.ceil(rule["minimum_final_exact_accuracy"] * n)
    required_advantage = math.ceil(rule["minimum_control_advantage"] * n)
    allowed_oracle_gap = math.floor(rule["maximum_oracle_exact_gap"] * n)
    checks = {
        "reader": (
            reader["pair_accuracy"] >= rule["minimum_reader_pair_accuracy"]
        ),
        "deterministic_compute": (
            computed_target_accuracy
            >= rule["minimum_computed_target_accuracy"]
        ),
        "final_exact": latent_correct >= required_final,
        "oracle_gap": oracle_correct - latent_correct <= allowed_oracle_gap,
        "control_advantage": (
            latent_correct - strongest_control >= required_advantage
        ),
        "parse": (
            not rule["require_parse_rate"]
            or conditions["latent_read_compute_write"]["parse_rate"] == 1.0
        ),
        "digit_tokens": (
            not rule["require_digit_token_rate"]
            or conditions["latent_read_compute_write"]["digit_token_rate"]
            == 1.0
        ),
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "counts": {
            "n": n,
            "latent_exact": latent_correct,
            "oracle_exact": oracle_correct,
            "strongest_control_exact": strongest_control,
            "required_final_exact": required_final,
            "required_control_advantage": required_advantage,
            "allowed_oracle_gap": allowed_oracle_gap,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite graft result: {args.output}")

    config = json.loads(args.config.read_text())
    evaluation_split = config["evaluation_split"]
    if evaluation_split not in ("development", "audit"):
        raise SystemExit(f"unsupported evaluation split: {evaluation_split}")
    if evaluation_split == "audit":
        if not config.get("audit_authorized", False):
            raise SystemExit("audit is sealed")
        if config.get("maximum_audit_runs") != 1:
            raise SystemExit("audit config must authorize exactly one run")
        if str(args.output) != config.get("audit_output"):
            raise SystemExit("output path differs from frozen audit path")
        verify_sha256(Path(__file__), config["runner_sha256"])
    elif config.get("audit_authorized", False):
        raise SystemExit("development config cannot authorize audit")

    paths = {
        "dataset_config": Path(config["dataset_config"]),
        "reader_selection_result": Path(config["reader_selection_result"]),
        "reader_artifact": Path(config["reader_artifact"]),
        "writer_manifest": Path(config["writer_manifest"]),
    }
    for name, path in paths.items():
        verify_sha256(path, config[f"{name}_sha256"])
    dataset_config = json.loads(paths["dataset_config"].read_text())
    reader_selection = json.loads(paths["reader_selection_result"].read_text())
    if not reader_selection["passes"]:
        raise SystemExit("operand reader selection did not pass")
    if reader_selection["selection"]["hidden_state_index"] != config[
        "reader_hidden_state_index"
    ]:
        raise SystemExit("reader hidden-state index differs from selection")
    if evaluation_split == "audit":
        development_paths = {
            "development_config": Path(config["development_config"]),
            "development_result": Path(config["development_result"]),
        }
        for name, path in development_paths.items():
            verify_sha256(path, config[f"{name}_sha256"])
        development_config = json.loads(
            development_paths["development_config"].read_text()
        )
        development_result = json.loads(
            development_paths["development_result"].read_text()
        )
        if not development_result["passes"]:
            raise SystemExit("development graft did not pass")
        locked = (
            "base_model_batch_size",
            "dataset_config_sha256",
            "development_rule",
            "random_control_seed",
            "reader_artifact_sha256",
            "reader_hidden_state_index",
            "reader_selection_result_sha256",
            "writer_manifest_sha256",
        )
        for key in locked:
            if config[key] != development_config[key]:
                raise SystemExit(f"audit changed development field: {key}")

    examples = build_phase7_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    observed_dataset_hash = phase7_carry_sha256(examples)
    if observed_dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 8 dataset hash mismatch")
    selected = [
        row
        for row in examples
        if row.split == evaluation_split and row.variant == "carry_base"
    ]
    if value_sha256([row.example_id for row in selected]) != config[
        f"{evaluation_split}_examples_sha256"
    ]:
        raise SystemExit(f"{evaluation_split} example hash mismatch")

    model_config = dataset_config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered, positions, token_contract = render_and_locate(
        tokenizer,
        selected,
        dataset_config["assistant_prefix"],
    )
    if value_sha256(token_contract) != config[
        f"{evaluation_split}_token_contract_sha256"
    ]:
        raise SystemExit(f"{evaluation_split} token contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])

    manifest = NativeCoordinateManifest.load(paths["writer_manifest"])
    manifest.verify(Path("."))
    if (
        manifest.model_id != model_config["id"]
        or manifest.model_revision != model_config["revision"]
    ):
        raise SystemExit("writer manifest/model mismatch")
    bases, prototypes, writer_config = load_writer_components(
        manifest,
        Path("."),
    )
    engine_config = {
        **writer_config,
        "base_model_batch_size": config["base_model_batch_size"],
        "random_control_seed": config["random_control_seed"],
    }

    reader_tensors = load_file(str(paths["reader_artifact"]))
    reader = NearestCentroidDigitReader(
        classes=reader_tensors["digit_classes"],
        centroids=reader_tensors["digit_centroids"],
    )
    device = torch.device(args.device)
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

    captured = capture.capture_token_positions(
        rendered,
        positions,
        hidden_state_indices=[config["reader_hidden_state_index"]],
        batch_size=config["base_model_batch_size"],
    )[config["reader_hidden_state_index"]]
    reader_states, _ = flatten_states_and_labels(captured.values, selected)
    flat_predictions = reader.predict(reader_states).tolist()
    grouped_predictions = group_predictions(flat_predictions, positions)
    read_metrics = reader_metrics(grouped_predictions, selected)
    computed_targets = [
        row["predicted_operand_a"] + row["predicted_operand_b"]
        for row in read_metrics["rows"]
    ]
    true_targets = [row.result for row in selected]
    computed_correct = sum(
        actual == expected
        for actual, expected in zip(
            computed_targets,
            true_targets,
            strict=True,
        )
    )
    computed_target_accuracy = computed_correct / len(selected)
    if any(target < 100 or target > 999 for target in computed_targets):
        raise SystemExit("decoded target lies outside three-digit writer contract")
    shuffled_targets = computed_targets[1:] + computed_targets[:1]

    raw_conditions = {
        "base": ("base", true_targets),
        "oracle_compute_native_write": (
            "donor_free_targeted",
            true_targets,
        ),
        "latent_read_compute_write": (
            "donor_free_targeted",
            computed_targets,
        ),
        "shuffled_read_compute_write": (
            "donor_free_targeted",
            shuffled_targets,
        ),
        "random_native_subspace": (
            "random_subspace_norm_matched",
            true_targets,
        ),
    }
    conditions = {}
    for condition_index, (name, (engine_name, targets)) in enumerate(
        raw_conditions.items()
    ):
        result = evaluate_condition(
            engine_name,
            model,
            tokenizer,
            capture,
            examples=selected,
            targets=targets,
            rendered_prompts=rendered,
            bases=bases,
            prototypes=prototypes,
            digit_token_ids=digit_token_ids,
            config=engine_config,
            device=device,
            condition_index=condition_index,
        )
        conditions[name] = {
            **result,
            **true_result_metrics(result, true_targets),
        }
    gate = development_gate(
        conditions,
        reader=read_metrics,
        computed_target_accuracy=computed_target_accuracy,
        rule=config["development_rule"],
    )
    report = {
        "schema_version": f"oli.phase8-phi-latent-graft-{evaluation_split}/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": (
            "one_shot_audit"
            if evaluation_split == "audit"
            else "one_shot_development"
        ),
        "model": model_config,
        "dataset_sha256": observed_dataset_hash,
        "evaluation_split": evaluation_split,
        "reader": {
            "hidden_state_index": config["reader_hidden_state_index"],
            "artifact_sha256": config["reader_artifact_sha256"],
            "metrics": read_metrics,
        },
        "deterministic_compute": {
            "operation": "integer_addition",
            "correct": computed_correct,
            "accuracy": computed_target_accuracy,
            "targets": computed_targets,
        },
        "writer": {
            "manifest": config["writer_manifest"],
            "manifest_sha256": config["writer_manifest_sha256"],
            **writer_config,
        },
        "conditions": conditions,
        "gate": {
            "thresholds": config["development_rule"],
            **gate,
        },
        "passes": gate["passes"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Complete latent digit read, deterministic addition, and audited "
            "native answer write under an external operand-token locator."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
