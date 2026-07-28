#!/usr/bin/env python3
"""Correct float32 threshold selection using exact example counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file


def verify_sha256(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"hash mismatch for {path}: {observed} != {expected}")


def exact_count(value: float, n: int) -> int:
    count = round(value * n)
    if abs(value - count / n) > 1e-6:
        raise ValueError(f"metric {value} is not an {n}-example proportion")
    return count


def candidate_passes(
    metrics: dict[str, Any],
    *,
    n: int,
    minimum_target_accuracy: float,
    minimum_identity_accuracy: float,
    norm_cap: float,
) -> bool:
    target = metrics["target"]
    identity = metrics["identity"]
    return (
        exact_count(target["top1_exact"], n)
        >= math.ceil(minimum_target_accuracy * n)
        and exact_count(identity["top1_exact"], n)
        >= math.ceil(minimum_identity_accuracy * n)
        and target["mean_relative_norm"] <= norm_cap
        and target["digit_token_rate"] == 1.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-prototype", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.prototype_output.exists():
        raise SystemExit("refusing to overwrite corrected result or artifact")
    correction = json.loads(args.config.read_text())
    verify_sha256(args.source_result, correction["source_result_sha256"])
    verify_sha256(args.source_prototype, correction["source_prototype_sha256"])
    source = json.loads(args.source_result.read_text())
    n = correction["selection_examples"]
    rule = source["selection_rule"]

    passing = []
    exact_counts = {}
    for rank in source["ranks"]:
        row = source["rank_results"][str(rank)]
        scale = str(row["selected_scale"])
        metrics = row["metrics_by_scale"][scale]
        target_count = exact_count(metrics["target"]["top1_exact"], n)
        identity_count = exact_count(metrics["identity"]["top1_exact"], n)
        passes = candidate_passes(
            metrics,
            n=n,
            minimum_target_accuracy=rule["minimum_target_accuracy"],
            minimum_identity_accuracy=rule["minimum_identity_accuracy"],
            norm_cap=source["norm_cap"],
        )
        exact_counts[str(rank)] = {
            "selected_scale": row["selected_scale"],
            "target_correct": target_count,
            "identity_correct": identity_count,
            "n": n,
            "passes": passes,
        }
        if passes:
            passing.append(rank)
    if not passing:
        raise SystemExit("no rank passes after exact-count correction")
    selected_rank = min(passing)
    selected_scale = source["rank_results"][str(selected_rank)]["selected_scale"]

    artifact = load_file(str(args.source_prototype))
    leading = artifact["leading_digit"]
    counts = artifact["leading_counts"]
    if leading.shape[1] < selected_rank:
        raise SystemExit("source prototype does not contain the corrected rank")
    args.prototype_output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "leading_digit": leading[:, :selected_rank].contiguous(),
            "leading_counts": counts,
        },
        str(args.prototype_output),
    )
    artifact_hash = hashlib.sha256(args.prototype_output.read_bytes()).hexdigest()

    corrected = {
        **source,
        "schema_version": "oli.phase3-leading-prototype-rank/v2",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selection_only_exact_count_correction",
        "selected_rank": selected_rank,
        "selected_scale": selected_scale,
        "passes": True,
        "prototype": {
            "path": str(args.prototype_output),
            "sha256": artifact_hash,
            "shape": list(leading[:, :selected_rank].shape),
        },
        "exact_count_selection": exact_counts,
        "correction": {
            "reason": (
                "float32 63/90 was stored as 0.699999988 and compared "
                "strictly against 0.7"
            ),
            "source_result": str(args.source_result),
            "source_result_sha256": correction["source_result_sha256"],
            "source_prototype": str(args.source_prototype),
            "source_prototype_sha256": correction["source_prototype_sha256"],
            "method": (
                "reselect from preserved metrics using integer counts; slice "
                "the nested rank-32 prototype to the selected rank"
            ),
            "no_model_evaluation_repeated": True,
        },
        "correction_config_sha256": hashlib.sha256(
            args.config.read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corrected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
