#!/usr/bin/env python3
"""Run the one-shot pair-disjoint Phi hybrid-graft audit."""

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
from open_latent_interfaces.evaluation import random_norm_matched
from open_latent_interfaces.operand_reader import NearestCentroidDigitReader
from open_latent_interfaces.phase9e_data import (
    PHASE9E_TEMPLATES,
    build_phase9e_audit,
    phase9e_audit_sha256,
    prior_dataset_hashes,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def template_sha256() -> str:
    encoded = json.dumps(PHASE9E_TEMPLATES, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_pair_sha256(examples: list[Any]) -> str:
    pairs = [
        sorted((row.operand_a, row.operand_b))
        for row in examples
    ]
    encoded = json.dumps(pairs, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def paired_metrics(
    conditions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = {
        row["example_id"]: row for row in conditions["base"]["outputs"]
    }
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
    rows = {}
    for name in (
        "oracle_compute_hybrid_write",
        "latent_read_compute_hybrid_write",
        "random_norm_matched",
    ):
        outputs = {
            row["example_id"]: row for row in conditions[name]["outputs"]
        }
        recovered = sum(
            outputs[example_id]["parsed"] == true_results[example_id]
            for example_id in base_errors
        )
        preserved = sum(
            outputs[example_id]["parsed"] == true_results[example_id]
            for example_id in base_correct
        )
        harmed = len(base_correct) - preserved
        rows[name] = {
            "base_error_count": len(base_errors),
            "base_correct_count": len(base_correct),
            "recovered_base_errors": recovered,
            "base_error_recovery": (
                recovered / len(base_errors) if base_errors else None
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
    return {
        "base_error_count": len(base_errors),
        "base_correct_count": len(base_correct),
        "conditions": rows,
    }


def audit_gate(
    conditions: dict[str, dict[str, Any]],
    paired: dict[str, Any],
    *,
    reader_pair_accuracy: float,
    computed_accuracy: float,
    rule: dict[str, Any],
    leading_norm_cap: float,
    suffix_norm_cap: float,
) -> dict[str, Any]:
    latent = conditions["latent_read_compute_hybrid_write"]
    oracle = conditions["oracle_compute_hybrid_write"]
    shuffled = conditions["shuffled_read_compute_hybrid_write"]
    latent_paired = paired["conditions"]["latent_read_compute_hybrid_write"]
    random_paired = paired["conditions"]["random_norm_matched"]
    base_errors = paired["base_error_count"]
    if base_errors:
        latent_recovery = latent_paired["base_error_recovery"]
        random_recovery = random_paired["base_error_recovery"]
        recovery_checks = {
            "base_error_recovery": (
                latent_recovery >= rule["minimum_base_error_recovery"]
            ),
            "excess_recovery_over_random": (
                latent_recovery - random_recovery
                >= rule["minimum_excess_base_error_recovery_over_random"]
            ),
        }
        recovery_branch = "observed_base_errors"
    else:
        recovery_checks = {
            "zero_error_full_exact": latent["true_result_accuracy"] == 1.0,
            "zero_error_no_harm": (
                latent_paired["base_correct_preservation"] == 1.0
            ),
        }
        recovery_branch = "zero_base_errors"
    checks = {
        "reader": reader_pair_accuracy >= rule["minimum_reader_pair_accuracy"],
        "compute": computed_accuracy
        >= rule["minimum_computed_target_accuracy"],
        "latent_exact": latent["true_result_accuracy"]
        >= rule["minimum_final_exact_accuracy"],
        "oracle_exact": oracle["true_result_accuracy"]
        >= rule["minimum_final_exact_accuracy"],
        "oracle_gap": (
            oracle["true_result_accuracy"] - latent["true_result_accuracy"]
            <= rule["maximum_oracle_exact_gap"]
        ),
        "positions": all(
            value >= rule["minimum_position_accuracy"]
            for value in latent["step_target_accuracy"]
        ),
        "base_correct_preservation": (
            latent_paired["base_correct_preservation"]
            >= rule["minimum_base_correct_preservation"]
        ),
        "net_non_degradation": (
            latent_paired["net_exact_improvement_rate"]
            >= rule["minimum_net_improvement_over_base"]
        ),
        "shuffled_target_following": (
            shuffled["target_full_result_accuracy"]
            >= rule["minimum_shuffled_target_accuracy"]
        ),
        "shuffled_true_control": (
            shuffled["true_result_accuracy"]
            <= rule["maximum_shuffled_true_accuracy"]
        ),
        "parse": (
            not rule["require_parse_rate"] or latent["parse_rate"] == 1.0
        ),
        "digit_tokens": (
            not rule["require_digit_token_rate"]
            or latent["digit_token_rate"] == 1.0
        ),
        "leading_norm": (
            latent["mean_relative_norm_by_step"][0] <= leading_norm_cap
        ),
        "suffix_norms": all(
            value <= suffix_norm_cap
            for value in latent["mean_relative_norm_by_step"][1:]
        ),
        **recovery_checks,
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "recovery_branch": recovery_branch,
        "paired_metrics": paired,
        "derived": {
            "oracle_exact_gap": (
                oracle["true_result_accuracy"] - latent["true_result_accuracy"]
            ),
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
        raise SystemExit("refusing to overwrite one-shot audit result")
    config = json.loads(args.config.read_text())
    if not config.get("audit_authorized", False):
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
        "dataset_config",
        "dataset_generator",
        "reader_selection_result",
        "reader_artifact",
        "compiler_selection_result",
        "compiler_module",
        "suffix_selection_result",
        "suffix_artifact",
        "suffix_basis_artifact",
        "development_config",
        "development_result",
        "development_correction_config",
        "development_correction",
    )
    paths = {name: Path(config[name]) for name in source_names}
    for name, path in paths.items():
        verify_sha256(path, config[f"{name}_sha256"])
    dataset_config = json.loads(paths["dataset_config"].read_text())
    reader_selection = json.loads(paths["reader_selection_result"].read_text())
    compiler_selection = json.loads(
        paths["compiler_selection_result"].read_text()
    )
    suffix_selection = json.loads(
        paths["suffix_selection_result"].read_text()
    )
    development_config = json.loads(paths["development_config"].read_text())
    development_result = json.loads(paths["development_result"].read_text())
    development_correction = json.loads(
        paths["development_correction"].read_text()
    )
    if dataset_config.get("audit_authorized", True):
        raise SystemExit("audit dataset was not frozen sealed")
    if not reader_selection["passes"]:
        raise SystemExit("reader source did not pass")
    if not compiler_selection["selection"]["passes"]:
        raise SystemExit("leading compiler source did not pass")
    if not development_correction["passes"]:
        raise SystemExit("corrected development gate did not pass")
    if development_result["config_sha256"] != config[
        "development_config_sha256"
    ]:
        raise SystemExit("development result/config mismatch")
    if development_correction["original"]["result_sha256"] != config[
        "development_result_sha256"
    ]:
        raise SystemExit("development correction/result mismatch")
    locked = (
        "reader_hidden_state_index",
        "reader_selection_result_sha256",
        "reader_artifact_sha256",
        "compiler_selection_result_sha256",
        "compiler_module_sha256",
        "suffix_selection_result_sha256",
        "suffix_artifact_sha256",
        "suffix_basis_artifact_sha256",
        "leading_compiler",
        "suffix_writer",
    )
    for key in locked:
        if config[key] != development_config[key]:
            raise SystemExit(f"audit changed development component: {key}")
    if config["leading_compiler"]["iterations"] != compiler_selection[
        "selection"
    ]["iterations"]:
        raise SystemExit("audit changed selected compiler depth")
    for position in ("1", "2"):
        selected_suffix = suffix_selection["positions"][position]["selection"]
        if not selected_suffix["passes"]:
            raise SystemExit(f"suffix position {position} did not pass")
        if selected_suffix["scale"] != config["suffix_writer"]["scale"]:
            raise SystemExit(f"audit changed suffix position {position} scale")

    examples = build_phase9e_audit(**dataset_config["dataset"]["parameters"])
    if phase9e_audit_sha256(examples) != dataset_config["dataset"]["sha256"]:
        raise SystemExit("fresh audit dataset hash mismatch")
    if value_sha256([row.example_id for row in examples]) != config[
        "audit_examples_sha256"
    ]:
        raise SystemExit("audit example ID hash mismatch")
    if canonical_pair_sha256(examples) != dataset_config["dataset"][
        "canonical_pairs_sha256"
    ]:
        raise SystemExit("audit canonical-pair hash mismatch")
    if prior_dataset_hashes() != dataset_config["dataset"][
        "prior_dataset_hashes"
    ]:
        raise SystemExit("prior dataset universe changed")
    if template_sha256() != dataset_config["dataset"]["template_sha256"]:
        raise SystemExit("audit template hash mismatch")

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
        examples,
        dataset_config["assistant_prefix"],
    )
    if value_sha256(rendered) != config["audit_rendered_prompts_sha256"]:
        raise SystemExit("audit rendered-prompt hash mismatch")
    if value_sha256([list(row) for row in positions]) != config[
        "audit_operand_positions_sha256"
    ]:
        raise SystemExit("audit operand-position hash mismatch")
    if value_sha256(token_contract) != config["audit_token_contract_sha256"]:
        raise SystemExit("audit token contract mismatch")
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
    suffix_tensors = load_file(str(paths["suffix_artifact"]))
    basis_tensors = load_file(str(paths["suffix_basis_artifact"]))
    suffix_basis = basis_tensors["suffix_basis"][
        : config["suffix_writer"]["rank"]
    ].float()
    suffix_prototypes = {
        1: suffix_tensors["tens_digit"].float(),
        2: suffix_tensors["ones_digit"].float(),
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
    captured = capture.capture_token_positions(
        rendered,
        positions,
        hidden_state_indices=[config["reader_hidden_state_index"]],
        batch_size=config["base_model_batch_size"],
    )[config["reader_hidden_state_index"]]
    reader_states, _ = flatten_states_and_labels(captured.values, examples)
    grouped = group_predictions(reader.predict(reader_states).tolist(), positions)
    read_metrics = reader_metrics(grouped, examples)
    computed_targets = [
        row["predicted_operand_a"] + row["predicted_operand_b"]
        for row in read_metrics["rows"]
    ]
    true_targets = [row.result for row in examples]
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
        raise SystemExit("decoded target outside three-digit writer contract")
    shuffled_targets = computed_targets[1:] + computed_targets[:1]

    compiler = config["leading_compiler"]
    trace_cache: dict[tuple[int, ...], IterativeMarginTrace] = {}
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
    random_delta = random_norm_matched(
        tuple(true_delta.shape),
        true_delta.norm(dim=1),
        seed=config["random_control_seed"],
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
            random_delta,
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
            examples=examples,
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
    paired = paired_metrics(conditions)
    gate = audit_gate(
        conditions,
        paired,
        reader_pair_accuracy=read_metrics["pair_accuracy"],
        computed_accuracy=computed_accuracy,
        rule=config["audit_rule"],
        leading_norm_cap=compiler["norm_cap"],
        suffix_norm_cap=config["suffix_writer"]["norm_cap"],
    )
    report = {
        "schema_version": "oli.phase9e-phi-hybrid-graft-audit/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "one_shot_audit",
        "audit_runs": 1,
        "model": model_config,
        "dataset": {
            "sha256": dataset_config["dataset"]["sha256"],
            "examples": len(examples),
            "example_ids_sha256": config["audit_examples_sha256"],
            "canonical_pairs_sha256": dataset_config["dataset"][
                "canonical_pairs_sha256"
            ],
            "prior_dataset_hashes": dataset_config["dataset"][
                "prior_dataset_hashes"
            ],
        },
        "reader": {
            "hidden_state_index": config["reader_hidden_state_index"],
            "metrics": read_metrics,
        },
        "deterministic_compute": {
            "operation": "integer_addition",
            "correct": computed_correct,
            "accuracy": computed_accuracy,
            "targets": computed_targets,
        },
        "writer": {
            "leading_compiler": compiler,
            "suffix_writer": config["suffix_writer"],
        },
        "conditions": conditions,
        "gate": {
            "thresholds": config["audit_rule"],
            **gate,
        },
        "passes": gate["passes"],
        "source_hashes": {
            f"{name}_sha256": config[f"{name}_sha256"]
            for name in source_names
        },
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "runner_sha256": runner_hash,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "One-shot pair-disjoint audit of latent operand decoding, external "
            "deterministic addition, iterative leading-token compilation, and "
            "native suffix writing on the frozen Phi model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
