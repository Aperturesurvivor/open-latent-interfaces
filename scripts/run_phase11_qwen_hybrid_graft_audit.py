#!/usr/bin/env python3
"""Run the one-shot pair- and template-disjoint Qwen hybrid-graft audit."""

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
from run_phase11_qwen_hybrid_graft_development import paired_summary
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.causal_compiler import (
    IterativeMarginTrace,
    compile_iterative_margin_deltas,
)
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.operand_reader import NearestCentroidDigitReader
from open_latent_interfaces.phase11_audit_data import (
    PHASE11_AUDIT_TEMPLATES,
    build_phase11_audit,
    phase11_audit_sha256,
)
from open_latent_interfaces.phase11_audit_data import (
    prior_dataset_hashes as phase11_prior_dataset_hashes,
)
from open_latent_interfaces.phase12_audit_data import (
    PHASE12_AUDIT_TEMPLATES,
    build_phase12_audit,
    phase12_audit_sha256,
)
from open_latent_interfaces.phase12_audit_data import (
    prior_dataset_hashes as phase12_prior_dataset_hashes,
)
from open_latent_interfaces.prefill import verify_decimal_digit_contract


def template_sha256(templates: tuple[str, ...]) -> str:
    encoded = json.dumps(
        templates,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_pair_sha256(examples: list[Any]) -> str:
    pairs = [sorted((row.operand_a, row.operand_b)) for row in examples]
    encoded = json.dumps(pairs, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit_gate(
    conditions: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, Any]],
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
    shuffled_random = conditions["shuffled_random_norm_matched"]
    latent_paired = paired["latent_read_compute_hybrid_write"]
    random_paired = paired["random_norm_matched"]
    wrong_paired = paired["wrong_target_norm_matched"]
    base_errors = latent_paired["base_error_count"]
    if base_errors:
        latent_recovery = latent_paired["base_error_recovery"]
        random_recovery = random_paired["base_error_recovery"]
        wrong_recovery = wrong_paired["base_error_recovery"]
        recovery_checks = {
            "base_error_recovery": (
                latent_recovery >= rule["minimum_base_error_recovery"]
            ),
            "excess_recovery_over_random": (
                latent_recovery - random_recovery
                >= rule["minimum_excess_base_error_recovery_over_random"]
            ),
            "excess_recovery_over_wrong": (
                latent_recovery - wrong_recovery
                >= rule[
                    "minimum_excess_base_error_recovery_over_wrong_target"
                ]
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
    shuffled_target_advantage = (
        shuffled["target_full_result_accuracy"]
        - shuffled_random["target_full_result_accuracy"]
    )
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
        "shuffled_random_target_control": (
            shuffled_random["target_full_result_accuracy"]
            <= rule["maximum_shuffled_random_target_accuracy"]
        ),
        "shuffled_target_advantage_over_random": (
            shuffled_target_advantage
            >= rule["minimum_shuffled_target_advantage_over_random"]
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
            "shuffled_target_advantage_over_random": (
                shuffled_target_advantage
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify every frozen input without loading the model or auditing",
    )
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

    source_names = [
        "dataset_config",
        "dataset_generator",
        "reader_selection_result",
        "reader_artifact",
        "compiler_selection_result",
        "compiler_module",
        "suffix_manifest",
        "suffix_audit_result",
        "tens_prototype_artifact",
        "ones_prototype_artifact",
        "suffix_basis_artifact",
        "development_config",
        "development_result",
    ]
    correction_names = [
        "development_correction_config",
        "development_correction",
    ]
    correction_configured = any(name in config for name in correction_names)
    if correction_configured and not all(
        name in config for name in correction_names
    ):
        raise SystemExit("development correction sources are incomplete")
    if correction_configured:
        source_names.extend(correction_names)
    paths = {name: Path(config[name]) for name in source_names}
    for name, path in paths.items():
        verify_sha256(path, config[f"{name}_sha256"])
    dataset_config = json.loads(paths["dataset_config"].read_text())
    reader_selection = json.loads(paths["reader_selection_result"].read_text())
    compiler_selection = json.loads(
        paths["compiler_selection_result"].read_text()
    )
    suffix_manifest = json.loads(paths["suffix_manifest"].read_text())
    development_config = json.loads(paths["development_config"].read_text())
    development_result = json.loads(paths["development_result"].read_text())
    if dataset_config.get("audit_authorized", True):
        raise SystemExit("audit dataset was not frozen sealed")
    if not reader_selection["passes"]:
        raise SystemExit("reader source did not pass")
    if not compiler_selection["selection"]["passes"]:
        raise SystemExit("leading compiler source did not pass")
    if development_result["config_sha256"] != config[
        "development_config_sha256"
    ]:
        raise SystemExit("development result/config mismatch")
    if correction_configured:
        development_correction = json.loads(
            paths["development_correction"].read_text()
        )
        if development_result["passes"]:
            raise SystemExit("correction must preserve an original non-pass")
        if not development_correction["passes"]:
            raise SystemExit("corrected development gate did not pass")
        if development_correction["original"]["result_sha256"] != config[
            "development_result_sha256"
        ]:
            raise SystemExit("development correction/result mismatch")
        if development_correction["original"]["config_sha256"] != config[
            "development_config_sha256"
        ]:
            raise SystemExit("development correction/config mismatch")
    elif not development_result["passes"]:
        raise SystemExit("uncorrected development result did not pass")
    locked = (
        "reader_hidden_state_index",
        "reader_selection_result_sha256",
        "reader_artifact_sha256",
        "compiler_selection_result_sha256",
        "compiler_module_sha256",
        "suffix_manifest_sha256",
        "suffix_audit_result_sha256",
        "tens_prototype_artifact_sha256",
        "ones_prototype_artifact_sha256",
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
    for field in ("hidden_state_index", "desired_margin", "norm_cap"):
        if config["leading_compiler"][field] != compiler_selection[field]:
            raise SystemExit(f"audit changed compiler {field}")
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

    audit_dataset_kind = config.get("audit_dataset_kind", "phase11")
    if audit_dataset_kind == "phase11":
        templates = PHASE11_AUDIT_TEMPLATES
        examples = build_phase11_audit(
            **dataset_config["dataset"]["parameters"]
        )
        dataset_hash = phase11_audit_sha256(examples)
        prior_hashes = phase11_prior_dataset_hashes()
    elif audit_dataset_kind == "phase12":
        templates = PHASE12_AUDIT_TEMPLATES
        examples = build_phase12_audit(
            **dataset_config["dataset"]["parameters"]
        )
        dataset_hash = phase12_audit_sha256(examples)
        prior_hashes = phase12_prior_dataset_hashes()
    else:
        raise SystemExit(f"unsupported audit dataset kind: {audit_dataset_kind}")
    if dataset_hash != dataset_config["dataset"]["sha256"]:
        raise SystemExit("fresh audit dataset hash mismatch")
    if value_sha256([row.example_id for row in examples]) != config[
        "audit_examples_sha256"
    ]:
        raise SystemExit("audit example ID hash mismatch")
    if canonical_pair_sha256(examples) != dataset_config["dataset"][
        "canonical_pairs_sha256"
    ]:
        raise SystemExit("audit canonical-pair hash mismatch")
    if prior_hashes != dataset_config["dataset"][
        "prior_dataset_hashes"
    ]:
        raise SystemExit("prior dataset universe changed")
    if template_sha256(templates) != dataset_config["dataset"][
        "template_sha256"
    ]:
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
    suffix_basis = load_file(str(paths["suffix_basis_artifact"]))[
        "delta_basis"
    ][: config["suffix_writer"]["rank"]].float()
    suffix_prototypes = {
        1: load_file(str(paths["tens_prototype_artifact"]))["digit"].float(),
        2: load_file(str(paths["ones_prototype_artifact"]))["digit"].float(),
    }
    if args.preflight_only:
        print("audit preflight passed; no model evaluation was performed")
        return

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
    paired = paired_summary(conditions)
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
        "schema_version": config.get(
            "result_schema_version",
            "oli.phase11-qwen-hybrid-graft-audit/v1",
        ),
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
        "claim_boundary": config.get(
            "claim_boundary",
            (
                "One-shot pair- and template-disjoint Qwen audit of latent "
                "operand decoding, external deterministic addition, iterative "
                "leading-token compilation, and previously audited native "
                "suffix writing."
            ),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
