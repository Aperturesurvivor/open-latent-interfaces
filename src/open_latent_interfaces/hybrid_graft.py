from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from open_latent_interfaces.operand_reader import OperandReaderManifest


def _verify_file(root: Path, reference: dict[str, Any], name: str) -> Path:
    path = root / str(reference["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != str(reference["sha256"]):
        raise ValueError(f"hybrid-graft hash mismatch: {name}")
    return path


@dataclass(frozen=True)
class HybridGraftManifest:
    schema_version: str
    name: str
    model_id: str
    model_revision: str
    residual_width: int
    prompt_contract: dict[str, Any]
    reader: dict[str, Any]
    deterministic_mechanism: dict[str, Any]
    leading_writer: dict[str, Any]
    suffix_writer: dict[str, Any]
    evidence: dict[str, Any]
    claim_boundary: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HybridGraftManifest:
        if value.get("schema_version") != "oli.hybrid-graft-interface/v1":
            raise ValueError("unsupported hybrid-graft manifest schema")
        model = value["model"]
        writer = value["writer"]
        return cls(
            schema_version=str(value["schema_version"]),
            name=str(value["name"]),
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            residual_width=int(value["representation"]["residual_width"]),
            prompt_contract=dict(value["prompt_contract"]),
            reader=dict(value["reader"]),
            deterministic_mechanism=dict(value["deterministic_mechanism"]),
            leading_writer=dict(writer["leading"]),
            suffix_writer=dict(writer["suffix"]),
            evidence=dict(value["evidence"]),
            claim_boundary=tuple(str(row) for row in value["claim_boundary"]),
        )

    @classmethod
    def load(cls, path: Path) -> HybridGraftManifest:
        return cls.from_dict(json.loads(path.read_text()))

    def verify(self, root: Path) -> None:
        if self.residual_width < 1:
            raise ValueError("residual width must be positive")
        if self.prompt_contract.get("assistant_prefix") != "Answer=":
            raise ValueError("unsupported hybrid-graft assistant prefix")
        if self.deterministic_mechanism.get("type") != "host_integer_addition":
            raise ValueError("unsupported deterministic mechanism")

        reader_manifest_path = _verify_file(
            root,
            self.reader["manifest"],
            "reader manifest",
        )
        reader = OperandReaderManifest.load(reader_manifest_path)
        reader.verify(root)
        if (
            reader.model_id != self.model_id
            or reader.model_revision != self.model_revision
            or reader.residual_width != self.residual_width
        ):
            raise ValueError("hybrid reader/model mismatch")
        if reader.hidden_state_index != int(self.reader["hidden_state_index"]):
            raise ValueError("hybrid reader boundary mismatch")

        compiler_path = _verify_file(
            root,
            self.leading_writer["implementation"],
            "leading compiler",
        )
        if compiler_path.suffix != ".py":
            raise ValueError("leading compiler implementation must be Python")
        if self.leading_writer.get("kind") != "iterative_local_margin_compiler":
            raise ValueError("unsupported leading writer kind")
        for key in ("hidden_state_index", "iterations"):
            if int(self.leading_writer[key]) < 1:
                raise ValueError(f"leading writer {key} must be positive")
        for key in ("desired_margin", "norm_cap"):
            if float(self.leading_writer[key]) <= 0:
                raise ValueError(f"leading writer {key} must be positive")

        suffix = self.suffix_writer
        if suffix.get("kind") != "native_coordinate_prototypes":
            raise ValueError("unsupported suffix writer kind")
        basis_path = _verify_file(root, suffix["basis"], "suffix basis")
        prototype_path = _verify_file(
            root,
            suffix["prototypes"],
            "suffix prototypes",
        )
        basis_tensors = load_file(str(basis_path))
        prototype_tensors = load_file(str(prototype_path))
        rank = int(suffix["rank"])
        basis_key = str(suffix["basis"]["key"])
        if basis_key not in basis_tensors:
            raise ValueError("suffix basis tensor is missing")
        basis = basis_tensors[basis_key]
        if basis.ndim != 2 or basis.shape[0] < rank:
            raise ValueError("suffix basis has the wrong shape")
        if basis.shape[1] != self.residual_width:
            raise ValueError("suffix basis residual width mismatch")
        for position in ("1", "2"):
            key = str(suffix["positions"][position]["prototype_key"])
            if key not in prototype_tensors:
                raise ValueError(f"suffix prototype tensor is missing: {key}")
            if prototype_tensors[key].shape != (10, rank):
                raise ValueError(f"suffix prototype has the wrong shape: {key}")

        required_evidence = (
            "dataset_config",
            "development_result",
            "development_correction",
            "audit_config",
            "audit_result",
        )
        for name in required_evidence:
            reference = self.evidence.get(name)
            if not isinstance(reference, dict):
                raise ValueError(f"missing hybrid-graft evidence: {name}")
            _verify_file(root, reference, name)
        if self.evidence.get("audit_runs") != 1:
            raise ValueError("hybrid graft must record exactly one audit run")
        audit_path = root / self.evidence["audit_result"]["path"]
        audit = json.loads(audit_path.read_text())
        if audit.get("passes") is not True or audit.get("audit_runs") != 1:
            raise ValueError("hybrid-graft audit did not pass exactly once")
        if audit["model"] != {
            "id": self.model_id,
            "revision": self.model_revision,
        }:
            raise ValueError("hybrid-graft audit/model mismatch")
        minimum_exact = float(self.evidence["minimum_audit_exact_accuracy"])
        latent = audit["conditions"]["latent_read_compute_hybrid_write"]
        if latent["true_result_accuracy"] < minimum_exact:
            raise ValueError("hybrid-graft audit exact accuracy is too low")
        if audit["reader"]["metrics"]["pair_accuracy"] < float(
            self.evidence["minimum_audit_reader_pair_accuracy"]
        ):
            raise ValueError("hybrid-graft audit reader accuracy is too low")

    def load_reader(self, root: Path):
        self.verify(root)
        path = root / str(self.reader["manifest"]["path"])
        return OperandReaderManifest.load(path).load_reader(root)

    def load_suffix_components(
        self,
        root: Path,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        self.verify(root)
        suffix = self.suffix_writer
        basis_tensors = load_file(str(root / suffix["basis"]["path"]))
        prototype_tensors = load_file(
            str(root / suffix["prototypes"]["path"])
        )
        rank = int(suffix["rank"])
        basis = basis_tensors[str(suffix["basis"]["key"])][:rank].float()
        prototypes = {
            int(position): prototype_tensors[
                str(spec["prototype_key"])
            ].float()
            for position, spec in suffix["positions"].items()
        }
        return basis, prototypes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an audited hybrid-graft interface manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or args.manifest.parent.parent
    manifest = HybridGraftManifest.load(args.manifest)
    manifest.verify(root)
    print(
        f"valid hybrid-graft interface: {manifest.name} "
        f"(model: {manifest.model_id})"
    )


if __name__ == "__main__":
    main()
