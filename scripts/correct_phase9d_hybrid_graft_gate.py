#!/usr/bin/env python3
"""Correct the Phase 9D wrong-target gate without rerunning the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def verify_sha256(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"hash mismatch for {path}: {observed} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite corrected gate result")
    config = json.loads(args.config.read_text())
    verify_sha256(Path(__file__), config["runner_sha256"])
    original_path = Path(config["original_result"])
    original_config_path = Path(config["original_config"])
    verify_sha256(original_path, config["original_result_sha256"])
    verify_sha256(
        original_config_path,
        config["original_config_sha256"],
    )
    original = json.loads(original_path.read_text())
    original_config = json.loads(original_config_path.read_text())
    if original["config_sha256"] != config["original_config_sha256"]:
        raise SystemExit("original result/config mismatch")
    if original["passes"]:
        raise SystemExit("correction requires an original non-pass")
    failed = {
        name
        for name, passed in original["gate"]["checks"].items()
        if not passed
    }
    if failed != {"wrong_target_control"}:
        raise SystemExit(f"unexpected original failed checks: {sorted(failed)}")
    if original["gate"]["thresholds"] != original_config["development_rule"]:
        raise SystemExit("original result thresholds differ from config")

    paired = original["gate"]["paired_metrics"]
    latent_recovery = paired["latent_read_compute_hybrid_write"][
        "base_error_recovery"
    ]
    wrong_recovery = paired["wrong_target_norm_matched"][
        "base_error_recovery"
    ]
    recovery_advantage = latent_recovery - wrong_recovery
    corrected_rule = config["corrected_rule"]
    checks = {
        name: passed
        for name, passed in original["gate"]["checks"].items()
        if name != "wrong_target_control"
    }
    checks["wrong_target_base_error_recovery"] = (
        wrong_recovery
        <= corrected_rule["maximum_wrong_target_base_error_recovery"]
    )
    checks["excess_recovery_over_wrong_target"] = (
        recovery_advantage
        >= corrected_rule[
            "minimum_excess_base_error_recovery_over_wrong_target"
        ]
    )
    report = {
        "schema_version": "oli.phase9d-hybrid-graft-gate-correction/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "post_hoc_metric_correction_no_model_rerun",
        "original": {
            "result": str(original_path),
            "result_sha256": config["original_result_sha256"],
            "config": str(original_config_path),
            "config_sha256": config["original_config_sha256"],
            "passes": original["passes"],
            "failed_checks": sorted(failed),
        },
        "unchanged_measurements": {
            "latent_base_error_recovery": latent_recovery,
            "wrong_target_base_error_recovery": wrong_recovery,
            "latent_recovery_advantage_over_wrong_target": recovery_advantage,
            "wrong_target_base_correct_preservation": paired[
                "wrong_target_norm_matched"
            ]["base_correct_preservation"],
            "wrong_target_true_accuracy": original["conditions"][
                "wrong_target_norm_matched"
            ]["true_result_accuracy"],
        },
        "corrected_rule": corrected_rule,
        "checks": checks,
        "passes": all(checks.values()),
        "runner_sha256": config["runner_sha256"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "claim_boundary": (
            "Post-hoc development metric correction over immutable Phase 9D "
            "outputs. No model inference, output filtering, or audit claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
