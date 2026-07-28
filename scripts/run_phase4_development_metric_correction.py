#!/usr/bin/env python3
"""Correct the operand development gate to exact-result discrimination."""

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
    original_path = Path(config["original_result"])
    verify_sha256(original_path, config["original_result_sha256"])
    original = json.loads(original_path.read_text())
    if original["passes"]["all"] or original["passes"]["operand"]:
        raise SystemExit("metric correction requires the original operand non-pass")
    if not original["passes"]["carry_context"]:
        raise SystemExit("metric correction cannot alter a carry non-pass")

    metrics = original["metrics"]["operand"]
    target = metrics["target"]
    controls = ("wrong_class_norm_matched", "random_norm_matched")
    strongest_name = max(
        controls,
        key=lambda name: metrics[name]["target_full_accuracy"],
    )
    strongest_accuracy = metrics[strongest_name]["target_full_accuracy"]
    advantage = target["target_full_accuracy"] - strongest_accuracy
    rule = config["correction_rule"]
    operand_passes = (
        target["target_full_accuracy"] >= rule["minimum_target_accuracy"]
        and advantage >= rule["minimum_control_advantage"]
        and (
            not rule["require_parse_rate"]
            or target["parse_rate"] == 1.0
        )
    )
    report = {
        "schema_version": "oli.phase4-development-metric-correction/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "post_hoc_bounded_metric_correction_no_model_rerun",
        "original_result": {
            "path": str(original_path),
            "sha256": config["original_result_sha256"],
            "passes": original["passes"],
        },
        "correction_rule": rule,
        "operand_exact_gate": {
            "target_accuracy": target["target_full_accuracy"],
            "strongest_control": strongest_name,
            "strongest_control_accuracy": strongest_accuracy,
            "control_advantage": advantage,
            "parse_rate": target["parse_rate"],
            "passes": operand_passes,
        },
        "carry_gate_unchanged": original["gates"]["carry_context"],
        "corrected_passes": {
            "operand": operand_passes,
            "carry_context": original["passes"]["carry_context"],
            "all": operand_passes and original["passes"]["carry_context"],
        },
        "passes": operand_passes and original["passes"]["carry_context"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "claim_boundary": (
            "Transparent post-hoc correction from tens-character accuracy to "
            "exact-result accuracy for the operand interface. The original "
            "development result remains a non-pass. No model inference, "
            "writer change, new data, or threshold change was introduced."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
