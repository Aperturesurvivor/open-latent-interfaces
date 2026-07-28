#!/usr/bin/env python3
"""Apply a bounded no-rerun correction to universal carry scale selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from run_phase4_carry_sequence_boundary import verify_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite correction result: {args.output}")

    config = json.loads(args.config.read_text())
    result_path = Path(config["original_result"])
    verify_sha256(result_path, config["original_result_sha256"])
    original = json.loads(result_path.read_text())
    if original["passes"]:
        raise SystemExit("bounded correction requires the original non-pass")
    if original["selection"]["passes"]:
        raise SystemExit("original selected scale unexpectedly passed")

    rule = config["correction_rule"]
    passing = []
    audit = {}
    for scale, metrics in original["metrics"].items():
        target = metrics["target"]
        control_accuracy = max(
            metrics["matched_no_carry_norm_matched"]["target_tens_accuracy"],
            metrics["random_norm_matched"]["target_tens_accuracy"],
        )
        advantage = target["target_tens_accuracy"] - control_accuracy
        passes = (
            target["target_tens_accuracy"] >= rule["minimum_tens_accuracy"]
            and advantage >= rule["minimum_control_advantage"]
            and (
                not rule["require_parse_rate"]
                or target["parse_rate"] == 1.0
            )
        )
        if passes:
            passing.append(float(scale))
        audit[scale] = {
            "target_tens_accuracy": target["target_tens_accuracy"],
            "target_full_accuracy": target["target_full_accuracy"],
            "strongest_control_tens_accuracy": control_accuracy,
            "control_advantage": advantage,
            "parse_rate": target["parse_rate"],
            "passes_all_original_gates": passes,
        }
    selected = min(passing) if passing else None
    report = {
        "schema_version": "oli.phase4-universal-carry-scale-correction/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "post_hoc_bounded_rule_correction_no_model_rerun",
        "original_result": {
            "path": str(result_path),
            "sha256": config["original_result_sha256"],
            "selected_scale": original["selection"]["scale"],
            "passes": original["passes"],
        },
        "correction_rule": rule,
        "scale_audit": audit,
        "passing_scales": sorted(passing),
        "selected_scale": selected,
        "passes": selected is not None,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "claim_boundary": (
            "Transparent post-hoc correction of a selection-rule defect. "
            "The original run remains a non-pass. No model inference, new "
            "scale, fitted weight, or additional data were introduced."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
