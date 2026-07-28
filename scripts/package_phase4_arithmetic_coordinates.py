#!/usr/bin/env python3
"""Package audited operand and carry coordinates into one tensor artifact."""

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
        "universal_carry_artifact": Path(config["universal_carry_artifact"]),
        "development_correction": Path(config["development_correction"]),
        "audit_config": Path(config["audit_config"]),
        "audit_result": Path(config["audit_result"]),
    }
    for name, path in paths.items():
        verify(path, config[f"{name}_sha256"])
    development = json.loads(paths["development_correction"].read_text())
    audit = json.loads(paths["audit_result"].read_text())
    if not development["passes"]:
        raise SystemExit("packaging requires corrected development pass")
    if not audit["passes"]["all"] or audit["status"] != "one_shot_audit":
        raise SystemExit("packaging requires passing one-shot audit")

    class_tensors = load_file(str(paths["class_prototype_artifact"]))
    carry_tensors = load_file(str(paths["universal_carry_artifact"]))
    tensors = {
        "source_digits": class_tensors["source_digits"].contiguous(),
        "fit_class_counts": class_tensors["class_counts"].contiguous(),
        "operand_delta": class_tensors["operand_delta"].contiguous(),
        "carry_context_delta": carry_tensors[
            "carry_context_delta"
        ].contiguous(),
        "control_context_delta": carry_tensors[
            "control_context_delta"
        ].contiguous(),
    }
    if tensors["operand_delta"].shape != (4, 3072):
        raise SystemExit("unexpected operand coordinate shape")
    if tensors["carry_context_delta"].shape != (3072,):
        raise SystemExit("unexpected carry coordinate shape")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "schema_version": "oli.arithmetic-coordinate-tensors/v1",
            "model_id": audit["model"]["id"],
            "model_revision": audit["model"]["revision"],
            "audit_result_sha256": config["audit_result_sha256"],
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
