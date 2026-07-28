#!/usr/bin/env python3
"""Apply the frozen paired-uplift correction without rerunning the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_phase4_carry_sequence_boundary import verify_sha256


def indexed_outputs(condition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in condition["outputs"]}


def paired_metrics(
    base: dict[str, dict[str, Any]],
    condition: dict[str, dict[str, Any]],
    true_results: dict[str, int],
) -> dict[str, Any]:
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
    recovered = sum(
        condition[example_id]["parsed"] == true_results[example_id]
        for example_id in base_errors
    )
    preserved = sum(
        condition[example_id]["parsed"] == true_results[example_id]
        for example_id in base_correct
    )
    harmed = len(base_correct) - preserved
    return {
        "base_error_count": len(base_errors),
        "base_correct_count": len(base_correct),
        "recovered_base_errors": recovered,
        "base_error_recovery": recovered / len(base_errors),
        "preserved_base_correct": preserved,
        "base_correct_preservation": preserved / len(base_correct),
        "harmed_base_correct": harmed,
        "net_exact_improvement": recovered - harmed,
        "net_exact_improvement_rate": (
            (recovered - harmed) / len(true_results)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite correction: {args.output}")

    config = json.loads(args.config.read_text())
    original_config_path = Path(config["original_config"])
    original_result_path = Path(config["original_result"])
    verify_sha256(
        original_config_path,
        config["original_config_sha256"],
    )
    verify_sha256(
        original_result_path,
        config["original_result_sha256"],
    )
    original = json.loads(original_result_path.read_text())
    if original["passes"]:
        raise SystemExit("metric correction requires an original nonpass")
    failed_checks = [
        name
        for name, passed in original["gate"]["checks"].items()
        if not passed
    ]
    if failed_checks != ["control_advantage"]:
        raise SystemExit(f"unexpected original failed checks: {failed_checks}")

    conditions = original["conditions"]
    base = indexed_outputs(conditions["base"])
    true_results = {
        example_id: row["original_result"] for example_id, row in base.items()
    }
    paired = {
        name: paired_metrics(
            base,
            indexed_outputs(conditions[name]),
            true_results,
        )
        for name in (
            "latent_read_compute_write",
            "oracle_compute_native_write",
            "random_native_subspace",
            "shuffled_read_compute_write",
        )
    }
    rule = config["corrected_rule"]
    latent = conditions["latent_read_compute_write"]
    oracle = conditions["oracle_compute_native_write"]
    shuffled = conditions["shuffled_read_compute_write"]
    latent_paired = paired["latent_read_compute_write"]
    random_paired = paired["random_native_subspace"]
    n = latent["n"]
    oracle_gap = (
        oracle["true_result_correct"] - latent["true_result_correct"]
    ) / n
    excess_recovery = (
        latent_paired["recovered_base_errors"]
        - random_paired["recovered_base_errors"]
    ) / latent_paired["base_error_count"]
    checks = {
        "reader": (
            original["reader"]["metrics"]["pair_accuracy"]
            >= rule["minimum_reader_pair_accuracy"]
        ),
        "deterministic_compute": (
            original["deterministic_compute"]["accuracy"]
            >= rule["minimum_computed_target_accuracy"]
        ),
        "final_exact": (
            latent["true_result_accuracy"]
            >= rule["minimum_final_exact_accuracy"]
        ),
        "oracle_gap": (
            oracle_gap <= rule["require_oracle_exact_gap_at_most"]
        ),
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
            excess_recovery
            >= rule["minimum_excess_base_error_recovery_over_random"]
        ),
        "shuffled_control": (
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
    }
    report = {
        "schema_version": "oli.phase8-phi-latent-graft-metric-correction/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_metric_correction_no_model_rerun",
        "original": {
            "config": config["original_config"],
            "config_sha256": config["original_config_sha256"],
            "result": config["original_result"],
            "result_sha256": config["original_result_sha256"],
            "passes": original["passes"],
            "failed_checks": failed_checks,
        },
        "paired_metrics": paired,
        "derived": {
            "oracle_exact_gap": oracle_gap,
            "excess_base_error_recovery_over_random": excess_recovery,
            "required_final_exact_count": math.ceil(
                rule["minimum_final_exact_accuracy"] * n
            ),
        },
        "corrected_rule": rule,
        "checks": checks,
        "passes": all(checks.values()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "claim_boundary": (
            "Transparent development-only paired-uplift metric correction. "
            "No model inference was rerun and no audit data was accessed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
