#!/usr/bin/env python3
"""Package the selected Phi bases and prototypes into one release artifact."""

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
        "basis": Path(config["basis"]),
        "leading": Path(config["leading_prototype"]),
        "suffix": Path(config["suffix_prototype"]),
        "audit": Path(config["audit_result"]),
    }
    for name, path in paths.items():
        verify(path, config[f"{name}_sha256"])
    audit = json.loads(paths["audit"].read_text())
    if not audit["passes"] or not all(audit["gate"]["checks"].values()):
        raise SystemExit("release packaging requires a passing audit")

    bases = load_file(str(paths["basis"]))
    leading = load_file(str(paths["leading"]))
    suffix = load_file(str(paths["suffix"]))
    tensors = {
        "leading_basis": bases["leading_basis"][:32].contiguous(),
        "suffix_basis": bases["suffix_basis"][:32].contiguous(),
        "leading_digit": leading["leading_digit"].contiguous(),
        "leading_counts": leading["leading_counts"].contiguous(),
        "tens_digit": suffix["position_1_digit"].contiguous(),
        "tens_counts": suffix["position_1_counts"].contiguous(),
        "ones_digit": suffix["position_2_digit"].contiguous(),
        "ones_counts": suffix["position_2_counts"].contiguous(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "schema_version": "oli.native-coordinate-tensors/v1",
            "model_id": audit["model"]["id"],
            "model_revision": audit["model"]["revision"],
            "audit_result_sha256": config["audit_sha256"],
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
