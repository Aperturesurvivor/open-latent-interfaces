#!/usr/bin/env python3
"""Package the independently audited Qwen operand coordinate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


def verify(path: Path, expected: str) -> None:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"hash mismatch for {path}: {observed} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite release artifact: {args.output}")

    config = json.loads(args.config.read_text())
    paths = {
        "class_prototype_artifact": Path(config["class_prototype_artifact"]),
        "development_result": Path(config["development_result"]),
        "audit_config": Path(config["audit_config"]),
        "audit_result": Path(config["audit_result"]),
    }
    for name, path in paths.items():
        verify(path, config[f"{name}_sha256"])
    development = json.loads(paths["development_result"].read_text())
    audit = json.loads(paths["audit_result"].read_text())
    if not development["passes"]["operand"]:
        raise SystemExit("packaging requires a passing operand development gate")
    if (
        audit["status"] != "one_shot_audit"
        or not audit["passes"]["operand"]
        or audit["passes"]["carry_context"]
    ):
        raise SystemExit(
            "package requires an operand pass and excludes the failed carry writer"
        )

    source = load_file(str(paths["class_prototype_artifact"]))
    tensors = {
        "source_digits": source["source_digits"].contiguous(),
        "fit_class_counts": source["class_counts"].contiguous(),
        "operand_delta": source["operand_delta"].contiguous(),
    }
    if tensors["source_digits"].tolist() != [1, 2, 3, 4]:
        raise SystemExit("unexpected source-digit classes")
    if tensors["operand_delta"].shape != (4, 1536):
        raise SystemExit("unexpected Qwen operand coordinate shape")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "schema_version": "oli.arithmetic-coordinate-tensors/v1",
            "model_id": audit["model"]["id"],
            "model_revision": audit["model"]["revision"],
            "audit_result_sha256": config["audit_result_sha256"],
            "included_interface": "operand_increment",
            "excluded_interface": "carry_context_audit_nonpass",
        },
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": digest,
                "bytes": args.output.stat().st_size,
                "tensors": {
                    name: list(tensor.shape) for name, tensor in tensors.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
