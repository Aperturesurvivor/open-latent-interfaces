#!/usr/bin/env python3
"""Evaluate the Qwen reader-compute-hybrid-writer graft on development data."""

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
from run_phase8_latent_graft import group_predictions, true_result_metrics
from run_phase8_operand_reader_selection import (
    flatten_states_and_labels,
    reader_metrics,
    render_and_locate,
)
from run_phase9d_hybrid_graft_development import evaluate_hybrid_condition
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.causal_compiler import (
    IterativeMarginTrace,
    compile_iterative_margin_deltas,
)
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.operand_reader import NearestCentroidDigitReader
from open_latent_interfaces.phase7_data import (
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def paired_summary(
    conditions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    base = {row["example_id"]: row for row in conditions["base"]["outputs"]}
    true_results = {
        example_id: row["original_result"] for example_id, row in base.items()
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
    for name in (
        "oracle_compute_hybrid_write",
        "latent_read_compute_hybrid_write",
        "shuffled_read_compute_hybrid_write",
        "shuffled_random_norm_matched",
        "random_norm_matched",
        "wrong_target_norm_matched",
    ):
        outputs = {row["example_id"]: row for row in conditions[name]["outputs"]}
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
    latent = conditions["latent_read_compute_hybrid_write"]
    oracle = conditions["oracle_compute_hybrid_write"]
    shuffled = conditions["shuffled_read_compute_hybrid_write"]
    shuffled_random = conditions["shuffled_random_norm_matched"]
    latent_paired = paired["latent_read_compute_hybrid_write"]
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
    shuffled_target_advantage = (
        shuffled["target_full_result_accuracy"]
        - shuffled_random["target_full_result_accuracy"]
    )
    checks = {
        "reader": reader["pair_accuracy"] >= rule["minimum_reader_pair_accuracy"],
        "compute": computed_accuracy
        >= rule["minimum_computed_target_accuracy"],
        "final_exact": latent["true_result_accuracy"]
        >= rule["minimum_final_exact_accuracy"],
        "position_accuracy": min(latent["step_target_accuracy"])
        >= rule["minimum_position_accuracy"],
        "oracle_gap": oracle_gap <= rule["maximum_oracle_exact_gap"],
        "base_error_recovery": latent_paired["base_error_recovery"]
        >= rule["minimum_base_error_recovery"],
        "base_correct_preservation": latent_paired["base_correct_preservation"]
        >= rule["minimum_base_correct_preservation"],
        "net_improvement": latent_paired["net_exact_improvement_rate"]
        >= rule["minimum_net_improvement_over_base"],
        "wrong_target_recovery": wrong_paired["base_error_recovery"]
        <= rule["maximum_wrong_target_base_error_recovery"],
        "excess_recovery_over_wrong": excess_wrong
        >= rule["minimum_excess_base_error_recovery_over_wrong_target"],
        "shuffled_true_control": shuffled["true_result_accuracy"]
        <= rule["maximum_shuffled_true_accuracy"],
        "shuffled_target_following": shuffled["target_full_result_accuracy"]
        >= rule["minimum_shuffled_target_accuracy"],
        "shuffled_random_target_control": shuffled_random[
            "target_full_result_accuracy"
        ]
        <= rule["maximum_shuffled_random_target_accuracy"],
        "shuffled_target_advantage_over_random": shuffled_target_advantage
        >= rule["minimum_shuffled_target_advantage_over_random"],
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
            "shuffled_target_advantage_over_random": shuffled_target_advantage,
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
        raise SystemExit("refusing to overwrite hybrid-graft result")
    config = json.loads(args.config.read_text())
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if runner_hash != config["runner_sha256"]:
        raise SystemExit("hybrid-graft runner hash mismatch")
    paths = {
        name: Path(config[name])
        for name in (
            "dataset_config",
            "reader_selection_result",
            "reader_artifact",
            "compiler_selection_result",
            "compiler_module",
            "initial_development_result",
            "suffix_manifest",
            "suffix_audit_result",
            "tens_prototype_artifact",
            "ones_prototype_artifact",
            "suffix_basis_artifact",
        )
    }
    for name, path in paths.items():
        verify_sha256(path, config[f"{name}_sha256"])
    for dependency, expected_hash in config["code_dependencies"].items():
        verify_sha256(Path(dependency), expected_hash)
    dataset_config = json.loads(paths["dataset_config"].read_text())
    reader_selection = json.loads(paths["reader_selection_result"].read_text())
    compiler_selection = json.loads(
        paths["compiler_selection_result"].read_text()
    )
    initial_development = json.loads(
        paths["initial_development_result"].read_text()
    )
    suffix_manifest = json.loads(paths["suffix_manifest"].read_text())
    if dataset_config.get("audit_authorized", False):
        raise SystemExit("hybrid development requires a sealed audit")
    if not reader_selection["passes"]:
        raise SystemExit("operand reader source did not pass")
    if not compiler_selection["selection"]["passes"]:
        raise SystemExit("leading compiler source did not pass")
    if initial_development["passes"]:
        raise SystemExit("refinement requires a non-passing initial development")
    if (
        compiler_selection["selection"]["iterations"]
        != config["leading_compiler"]["iterations"]
    ):
        raise SystemExit("leading compiler iteration count changed")
    for field in ("hidden_state_index", "desired_margin", "norm_cap"):
        if compiler_selection[field] != config["leading_compiler"][field]:
            raise SystemExit(f"leading compiler {field} changed")
    if suffix_manifest["model"] != dataset_config["model"]:
        raise SystemExit("suffix manifest model mismatch")
    if not suffix_manifest["evidence"]["audit_gate_passed"]:
        raise SystemExit("suffix manifest does not bind a passing audit")
    if suffix_manifest["evidence"]["audit_result"]["sha256"] != config[
        "suffix_audit_result_sha256"
    ]:
        raise SystemExit("suffix audit hash does not match manifest")
    for position, artifact_name in (
        ("1", "tens_prototype_artifact"),
        ("2", "ones_prototype_artifact"),
    ):
        manifest_position = suffix_manifest["positions"][position]
        configured = config["suffix_writer"]["positions"][position]
        if manifest_position["hidden_state_index"] != config["suffix_writer"][
            "hidden_state_index"
        ]:
            raise SystemExit(f"suffix position {position} boundary changed")
        for field in ("rank", "scale", "norm_cap"):
            if manifest_position[field] != configured[field]:
                raise SystemExit(f"suffix position {position} {field} changed")
        if manifest_position["prototypes"]["sha256"] != config[
            f"{artifact_name}_sha256"
        ]:
            raise SystemExit(f"suffix position {position} artifact changed")
    if suffix_manifest["representation"]["basis"]["sha256"] != config[
        "suffix_basis_artifact_sha256"
    ]:
        raise SystemExit("suffix basis changed")

    examples = build_phase7_carry_quartets(
        **dataset_config["dataset"]["parameters"]
    )
    dataset_hash = phase7_carry_sha256(examples)
    if dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("Phase 11 dataset hash mismatch")
    selected = [
        row
        for row in examples
        if row.split == "development"
        and row.variant in set(config["development_variants"])
    ]
    if value_sha256([row.example_id for row in selected]) != config[
        "development_examples_sha256"
    ]:
        raise SystemExit("development example hash mismatch")

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
        "development_token_contract_sha256"
    ]:
        raise SystemExit("development token contract mismatch")
    digit_token_ids = verify_decimal_digit_contract(tokenizer, rendered[0])
    candidate_ids = torch.tensor(
        [digit_token_ids[digit] for digit in range(10)],
        dtype=torch.long,
    )

    reader_tensors = load_file(str(paths["reader_artifact"]))
    reader = NearestCentroidDigitReader(
        classes=reader_tensors["digit_classes"],
        centroids=reader_tensors["digit_centroids"],
    )
    suffix_basis = load_file(str(paths["suffix_basis_artifact"]))[
        "delta_basis"
    ][: config["suffix_writer"]["rank"]].float()
    suffix_prototypes = {
        1: load_file(str(paths["tens_prototype_artifact"]))["digit"].float(),
        2: load_file(str(paths["ones_prototype_artifact"]))["digit"].float(),
    }

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

    reader_index = config["reader_hidden_state_index"]
    captured = capture.capture_token_positions(
        rendered,
        positions,
        hidden_state_indices=[reader_index],
        batch_size=config["base_model_batch_size"],
    )[reader_index]
    reader_states, _ = flatten_states_and_labels(captured.values, selected)
    grouped = group_predictions(reader.predict(reader_states).tolist(), positions)
    read_metrics = reader_metrics(grouped, selected)
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
    computed_accuracy = computed_correct / len(selected)
    if any(target < 100 or target > 999 for target in computed_targets):
        raise SystemExit("decoded target outside three-digit writer contract")
    shuffled_targets = computed_targets[1:] + computed_targets[:1]

    trace_cache: dict[tuple[int, ...], IterativeMarginTrace] = {}
    compiler = config["leading_compiler"]
    for targets in (true_targets, computed_targets, shuffled_targets):
        key = tuple(targets)
        if key in trace_cache:
            continue
        target_ids = torch.tensor(
            [digit_token_ids[int(str(value)[0])] for value in targets]
        )
        trace_cache[key] = compile_iterative_margin_deltas(
            model,
            tokenizer,
            rendered,
            hidden_state_index=compiler["hidden_state_index"],
            target_token_ids=target_ids,
            candidate_token_ids=candidate_ids,
            desired_margin=compiler["desired_margin"],
            iterations=compiler["iterations"],
            max_relative_norm=compiler["norm_cap"],
            device=device,
            batch_size=config["compiler_batch_size"],
        )
    true_trace = trace_cache[tuple(true_targets)]
    computed_trace = trace_cache[tuple(computed_targets)]
    shuffled_trace = trace_cache[tuple(shuffled_targets)]
    true_delta = true_trace.cumulative_deltas[-1]
    computed_delta = computed_trace.cumulative_deltas[-1]
    shuffled_delta = shuffled_trace.cumulative_deltas[-1]
    leading_random = random_norm_matched(
        tuple(true_delta.shape),
        true_delta.norm(dim=1),
        seed=config["random_control_seed"],
    )
    shuffled_leading_random = random_norm_matched(
        tuple(shuffled_delta.shape),
        shuffled_delta.norm(dim=1),
        seed=config["random_control_seed"] + 1,
    )
    wrong_targets = wrong_all_digits(true_targets)
    wrong_ids = torch.tensor(
        [digit_token_ids[int(str(value)[0])] for value in wrong_targets]
    )
    wrong_trace = compile_iterative_margin_deltas(
        model,
        tokenizer,
        rendered,
        hidden_state_index=compiler["hidden_state_index"],
        target_token_ids=wrong_ids,
        candidate_token_ids=candidate_ids,
        desired_margin=compiler["desired_margin"],
        iterations=compiler["iterations"],
        max_relative_norm=compiler["norm_cap"],
        device=device,
        batch_size=config["compiler_batch_size"],
    )
    leading_wrong = norm_match(
        wrong_trace.cumulative_deltas[-1],
        true_delta.norm(dim=1),
    )

    condition_specs = {
        "base": (
            "base",
            true_targets,
            torch.zeros_like(true_delta),
            true_trace.base_recipient_states,
        ),
        "oracle_compute_hybrid_write": (
            "oracle_compute_hybrid_write",
            true_targets,
            true_delta,
            true_trace.base_recipient_states,
        ),
        "latent_read_compute_hybrid_write": (
            "latent_read_compute_hybrid_write",
            computed_targets,
            computed_delta,
            computed_trace.base_recipient_states,
        ),
        "shuffled_read_compute_hybrid_write": (
            "shuffled_read_compute_hybrid_write",
            shuffled_targets,
            shuffled_delta,
            shuffled_trace.base_recipient_states,
        ),
        "random_norm_matched": (
            "random_norm_matched",
            true_targets,
            leading_random,
            true_trace.base_recipient_states,
        ),
        "shuffled_random_norm_matched": (
            "random_norm_matched",
            shuffled_targets,
            shuffled_leading_random,
            shuffled_trace.base_recipient_states,
        ),
        "wrong_target_norm_matched": (
            "wrong_target_norm_matched",
            true_targets,
            leading_wrong,
            true_trace.base_recipient_states,
        ),
    }
    conditions = {}
    for condition_index, (
        name,
        (engine_name, targets, leading_delta, reference_states),
    ) in enumerate(condition_specs.items()):
        result = evaluate_hybrid_condition(
            engine_name,
            model,
            tokenizer,
            capture,
            examples=selected,
            targets=targets,
            originals=true_targets,
            rendered_prompts=rendered,
            leading_delta=leading_delta,
            leading_reference_states=reference_states,
            suffix_basis=suffix_basis,
            suffix_prototypes=suffix_prototypes,
            digit_token_ids=digit_token_ids,
            config=config,
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
        computed_accuracy=computed_accuracy,
        rule=config["development_rule"],
    )
    report = {
        "schema_version": "oli.phase11-qwen-hybrid-graft-development/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_only",
        "model": model_config,
        "dataset_sha256": dataset_hash,
        "evaluation_split": "development",
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
            "leading_compiler": config["leading_compiler"],
            "suffix_writer": config["suffix_writer"],
        },
        "conditions": conditions,
        "gate": {
            "thresholds": config["development_rule"],
            **gate,
        },
        "passes": gate["passes"],
        "source_hashes": {
            f"{name}_sha256": config[f"{name}_sha256"]
            for name in paths
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
            "Exposed Qwen development-only latent operand read, deterministic "
            "addition, iterative leading write, and audited native suffix write."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
