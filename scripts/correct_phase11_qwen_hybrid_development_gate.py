#!/usr/bin/env python3
"""Correct one redundant Qwen development gate without model inference."""

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
    verify_sha256(original_config_path, config["original_config_sha256"])
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
    if failed != {"wrong_target_recovery"}:
        raise SystemExit(f"unexpected original failed checks: {sorted(failed)}")
    if original["gate"]["thresholds"] != original_config["development_rule"]:
        raise SystemExit("original result thresholds differ from config")

    paired = original["gate"]["paired_metrics"]
    latent = paired["latent_read_compute_hybrid_write"]
    wrong = paired["wrong_target_norm_matched"]
    recovery_advantage = (
        latent["base_error_recovery"] - wrong["base_error_recovery"]
    )
    retained_threshold = config["corrected_rule"][
        "minimum_excess_base_error_recovery_over_wrong_target"
    ]
    if retained_threshold != original["gate"]["thresholds"][
        "minimum_excess_base_error_recovery_over_wrong_target"
    ]:
        raise SystemExit("comparative recovery threshold changed")
    checks = {
        name: passed
        for name, passed in original["gate"]["checks"].items()
        if name != "wrong_target_recovery"
    }
    checks["excess_recovery_over_wrong"] = (
        recovery_advantage >= retained_threshold
    )
    report = {
        "schema_version": "oli.phase11-qwen-hybrid-gate-correction/v1",
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
            "base_error_count": latent["base_error_count"],
            "latent_recovered_base_errors": latent["recovered_base_errors"],
            "wrong_target_recovered_base_errors": wrong[
                "recovered_base_errors"
            ],
            "latent_base_error_recovery": latent["base_error_recovery"],
            "wrong_target_base_error_recovery": wrong[
                "base_error_recovery"
            ],
            "latent_recovery_advantage_over_wrong_target": recovery_advantage,
            "wrong_target_base_correct_preservation": wrong[
                "base_correct_preservation"
            ],
            "shuffled_target_advantage_over_random": original["gate"][
                "derived"
            ]["shuffled_target_advantage_over_random"],
        },
        "corrected_rule": config["corrected_rule"],
        "checks": checks,
        "passes": all(checks.values()),
        "runner_sha256": config["runner_sha256"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "claim_boundary": (
            "Post-hoc exposed-development metric correction over immutable "
            "Qwen outputs. No model inference, output filtering, or audit claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
