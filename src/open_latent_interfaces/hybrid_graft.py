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
        supported = {
            "oli.hybrid-graft-interface/v1",
            "oli.hybrid-graft-interface/v2",
        }
        if value.get("schema_version") not in supported:
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
        candidate_token_ids = self.leading_writer.get("candidate_token_ids")
        expected_digits = {str(digit) for digit in range(10)}
        if (
            not isinstance(candidate_token_ids, dict)
            or set(candidate_token_ids) != expected_digits
            or any(
                not isinstance(token_id, int) or token_id < 0
                for token_id in candidate_token_ids.values()
            )
            or len(set(candidate_token_ids.values())) != 10
        ):
            raise ValueError("leading writer digit-token map is invalid")

        suffix = self.suffix_writer
        if suffix.get("kind") != "native_coordinate_prototypes":
            raise ValueError("unsupported suffix writer kind")
        if int(suffix["hidden_state_index"]) < 1:
            raise ValueError("suffix hidden-state index must be positive")
        if float(suffix["norm_cap"]) <= 0:
            raise ValueError("suffix norm cap must be positive")
        basis_path = _verify_file(root, suffix["basis"], "suffix basis")
        basis_tensors = load_file(str(basis_path))
        basis_key = str(suffix["basis"]["key"])
        if basis_key not in basis_tensors:
            raise ValueError("suffix basis tensor is missing")
        basis = basis_tensors[basis_key]
        if basis.ndim != 2:
            raise ValueError("suffix basis has the wrong shape")
        if basis.shape[1] != self.residual_width:
            raise ValueError("suffix basis residual width mismatch")
        if self.schema_version == "oli.hybrid-graft-interface/v1":
            rank = int(suffix["rank"])
            prototype_path = _verify_file(
                root,
                suffix["prototypes"],
                "suffix prototypes",
            )
            prototype_tensors = load_file(str(prototype_path))
            if basis.shape[0] < rank:
                raise ValueError("suffix basis has the wrong shape")
            for position in ("1", "2"):
                key = str(suffix["positions"][position]["prototype_key"])
                if key not in prototype_tensors:
                    raise ValueError(
                        f"suffix prototype tensor is missing: {key}"
                    )
                if prototype_tensors[key].shape != (10, rank):
                    raise ValueError(
                        f"suffix prototype has the wrong shape: {key}"
                    )
            evidence_sources = self.evidence
            required_evidence = {
                "dataset_config",
                "development_result",
                "development_correction",
                "audit_config",
                "audit_result",
            }
        else:
            positions = suffix.get("positions")
            if not isinstance(positions, dict) or set(positions) != {"1", "2"}:
                raise ValueError("v2 suffix must define positions 1 and 2")
            for position, spec in positions.items():
                rank = int(spec["rank"])
                if rank < 1 or basis.shape[0] < rank:
                    raise ValueError(f"suffix position {position} rank is invalid")
                for key in ("scale", "norm_cap"):
                    if float(spec[key]) <= 0:
                        raise ValueError(
                            f"suffix position {position} {key} must be positive"
                        )
                prototype_path = _verify_file(
                    root,
                    spec["prototypes"],
                    f"suffix position {position} prototypes",
                )
                prototype_tensors = load_file(str(prototype_path))
                key = str(spec["prototypes"]["key"])
                if key not in prototype_tensors:
                    raise ValueError(
                        f"suffix prototype tensor is missing: {key}"
                    )
                if prototype_tensors[key].shape != (10, rank):
                    raise ValueError(
                        f"suffix prototype has the wrong shape: {key}"
                    )
            evidence_sources = self.evidence.get("sources")
            if not isinstance(evidence_sources, dict):
                raise ValueError("v2 hybrid evidence must contain sources")
            required_evidence = {
                "dataset_config",
                "compiler_selection_result",
                "development_config",
                "development_result",
                "audit_config",
                "audit_result",
            }
        if not required_evidence <= set(evidence_sources):
            missing = sorted(required_evidence - set(evidence_sources))
            raise ValueError(f"missing hybrid-graft evidence: {missing[0]}")
        source_names = (
            required_evidence
            if self.schema_version == "oli.hybrid-graft-interface/v1"
            else set(evidence_sources)
        )
        for name in source_names:
            reference = evidence_sources[name]
            if not isinstance(reference, dict):
                raise ValueError(f"invalid hybrid-graft evidence: {name}")
            _verify_file(root, reference, name)
        if self.evidence.get("audit_runs") != 1:
            raise ValueError("hybrid graft must record exactly one audit run")
        audit_path = root / evidence_sources["audit_result"]["path"]
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
        if self.schema_version == "oli.hybrid-graft-interface/v2":
            expected_model = {
                "id": self.model_id,
                "revision": self.model_revision,
            }
            development = json.loads(
                (
                    root / evidence_sources["development_result"]["path"]
                ).read_text()
            )
            if development.get("passes") is not True:
                raise ValueError("hybrid-graft development result did not pass")
            if development.get("model") != expected_model:
                raise ValueError("hybrid-graft development/model mismatch")
            compiler_selection = json.loads(
                (
                    root
                    / evidence_sources["compiler_selection_result"]["path"]
                ).read_text()
            )
            selection = compiler_selection.get("selection", {})
            if selection.get("passes") is not True:
                raise ValueError("hybrid-graft compiler selection did not pass")
            if compiler_selection.get("model") != expected_model:
                raise ValueError("hybrid-graft compiler/model mismatch")
            if int(selection["iterations"]) != int(
                self.leading_writer["iterations"]
            ):
                raise ValueError("hybrid-graft compiler depth mismatch")
            compiler_fields = {
                "hidden_state_index": int,
                "desired_margin": float,
                "norm_cap": float,
            }
            for field, cast in compiler_fields.items():
                if cast(compiler_selection[field]) != cast(
                    self.leading_writer[field]
                ):
                    raise ValueError(
                        f"hybrid-graft compiler {field} mismatch"
                    )
            audit_leading = audit["writer"]["leading_compiler"]
            for field, cast in {
                **compiler_fields,
                "iterations": int,
            }.items():
                if cast(audit_leading[field]) != cast(
                    self.leading_writer[field]
                ):
                    raise ValueError(
                        f"hybrid-graft audit leading {field} mismatch"
                    )
            audit_suffix = audit["writer"]["suffix_writer"]
            for field, cast in {
                "hidden_state_index": int,
                "norm_cap": float,
            }.items():
                if cast(audit_suffix[field]) != cast(suffix[field]):
                    raise ValueError(
                        f"hybrid-graft audit suffix {field} mismatch"
                    )
            for position, spec in suffix["positions"].items():
                audit_spec = audit_suffix["positions"][position]
                for field, cast in {
                    "rank": int,
                    "scale": float,
                    "norm_cap": float,
                }.items():
                    if cast(audit_spec[field]) != cast(spec[field]):
                        raise ValueError(
                            "hybrid-graft audit suffix position "
                            f"{position} {field} mismatch"
                        )
            shuffled = audit["conditions"][
                "shuffled_read_compute_hybrid_write"
            ]
            shuffled_random = audit["conditions"][
                "shuffled_random_norm_matched"
            ]
            shuffled_accuracy = shuffled["target_full_result_accuracy"]
            if shuffled_accuracy < float(
                self.evidence["minimum_shuffled_target_accuracy"]
            ):
                raise ValueError("hybrid-graft shuffled target accuracy is too low")
            advantage = (
                shuffled_accuracy
                - shuffled_random["target_full_result_accuracy"]
            )
            if advantage < float(
                self.evidence[
                    "minimum_shuffled_target_advantage_over_random"
                ]
            ):
                raise ValueError("hybrid-graft shuffled advantage is too low")

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
        if self.schema_version == "oli.hybrid-graft-interface/v1":
            rank = int(suffix["rank"])
            prototype_tensors = load_file(
                str(root / suffix["prototypes"]["path"])
            )
            prototypes = {
                int(position): prototype_tensors[
                    str(spec["prototype_key"])
                ].float()
                for position, spec in suffix["positions"].items()
            }
        else:
            ranks = {
                int(spec["rank"]) for spec in suffix["positions"].values()
            }
            if len(ranks) != 1:
                raise ValueError("suffix positions must share one basis rank")
            rank = ranks.pop()
            prototypes = {}
            for position, spec in suffix["positions"].items():
                tensors = load_file(str(root / spec["prototypes"]["path"]))
                prototypes[int(position)] = tensors[
                    str(spec["prototypes"]["key"])
                ].float()
        basis = basis_tensors[str(suffix["basis"]["key"])][:rank].float()
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
