from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_latent_interfaces.operand_reader import OperandReaderManifest


def _verify_file(root: Path, reference: dict[str, Any], name: str) -> Path:
    path = root / str(reference["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != str(reference["sha256"]):
        raise ValueError(f"compiler-graft hash mismatch: {name}")
    return path


@dataclass(frozen=True)
class CompilerGraftManifest:
    schema_version: str
    name: str
    model_id: str
    model_revision: str
    residual_width: int
    prompt_contract: dict[str, Any]
    reader: dict[str, Any]
    deterministic_mechanism: dict[str, Any]
    writer: dict[str, Any]
    evidence: dict[str, Any]
    claim_boundary: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CompilerGraftManifest:
        if value.get("schema_version") != "oli.compiler-graft-interface/v1":
            raise ValueError("unsupported compiler-graft manifest schema")
        model = value["model"]
        return cls(
            schema_version=str(value["schema_version"]),
            name=str(value["name"]),
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            residual_width=int(value["representation"]["residual_width"]),
            prompt_contract=dict(value["prompt_contract"]),
            reader=dict(value["reader"]),
            deterministic_mechanism=dict(value["deterministic_mechanism"]),
            writer=dict(value["writer"]),
            evidence=dict(value["evidence"]),
            claim_boundary=tuple(str(row) for row in value["claim_boundary"]),
        )

    @classmethod
    def load(cls, path: Path) -> CompilerGraftManifest:
        return cls.from_dict(json.loads(path.read_text()))

    def verify(self, root: Path) -> None:
        if self.residual_width < 1:
            raise ValueError("residual width must be positive")
        if self.prompt_contract.get("assistant_prefix") != "Answer=":
            raise ValueError("unsupported compiler-graft assistant prefix")
        if self.deterministic_mechanism.get("type") != "host_integer_addition":
            raise ValueError("unsupported deterministic mechanism")

        expected_model = {
            "id": self.model_id,
            "revision": self.model_revision,
        }
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
            raise ValueError("compiler reader/model mismatch")
        if reader.hidden_state_index != int(self.reader["hidden_state_index"]):
            raise ValueError("compiler reader boundary mismatch")

        if self.writer.get("kind") != ("sequential_prompt_local_margin_compiler"):
            raise ValueError("unsupported compiler writer kind")
        implementation = _verify_file(
            root,
            self.writer["implementation"],
            "compiler implementation",
        )
        if implementation.suffix != ".py":
            raise ValueError("compiler implementation must be Python")
        if int(self.writer.get("steps_per_position", 0)) != 1:
            raise ValueError("compiler must use one step per position")
        digit_ids = self.writer.get("candidate_token_ids")
        expected_digits = {str(digit) for digit in range(10)}
        if (
            not isinstance(digit_ids, dict)
            or set(digit_ids) != expected_digits
            or any(not isinstance(token_id, int) or token_id < 0 for token_id in digit_ids.values())
            or len(set(digit_ids.values())) != 10
        ):
            raise ValueError("compiler digit-token map is invalid")
        positions = self.writer.get("positions")
        if not isinstance(positions, dict) or set(positions) != {"0", "1", "2"}:
            raise ValueError("compiler must define positions 0, 1, and 2")
        for position, spec in positions.items():
            if int(spec["hidden_state_index"]) < 1:
                raise ValueError(f"position {position} boundary is invalid")
            for field in ("desired_margin", "norm_cap"):
                if float(spec[field]) <= 0:
                    raise ValueError(f"position {position} {field} must be positive")

        sources = self.evidence.get("sources")
        if not isinstance(sources, dict):
            raise ValueError("compiler evidence must contain sources")
        required_sources = {
            "audit_config",
            "audit_dataset_config",
            "audit_result",
            "development_config",
            "development_result",
            "leading_selection_result",
            "native_suffix_nonpass_result",
            "reader_selection_result",
            "suffix_selection_result",
        }
        if not required_sources <= set(sources):
            missing = sorted(required_sources - set(sources))
            raise ValueError(f"missing compiler-graft evidence: {missing[0]}")
        for name, reference in sources.items():
            if not isinstance(reference, dict):
                raise ValueError(f"invalid compiler-graft evidence: {name}")
            _verify_file(root, reference, name)
        if self.evidence.get("audit_runs") != 1:
            raise ValueError("compiler graft must record exactly one audit run")

        development = json.loads((root / sources["development_result"]["path"]).read_text())
        audit = json.loads((root / sources["audit_result"]["path"]).read_text())
        leading = json.loads((root / sources["leading_selection_result"]["path"]).read_text())
        suffix = json.loads((root / sources["suffix_selection_result"]["path"]).read_text())
        native_nonpass = json.loads(
            (root / sources["native_suffix_nonpass_result"]["path"]).read_text()
        )
        if development.get("passes") is not True:
            raise ValueError("compiler-graft development result did not pass")
        if audit.get("passes") is not True or audit.get("audit_runs") != 1:
            raise ValueError("compiler-graft audit did not pass exactly once")
        for result in (development, audit, leading, suffix, native_nonpass):
            if result.get("model") != expected_model:
                raise ValueError("compiler-graft evidence/model mismatch")
        if leading.get("selection", {}).get("passes") is not True:
            raise ValueError("leading compiler selection did not pass")
        if suffix.get("passes") is not True:
            raise ValueError("suffix compiler selection did not pass")
        if native_nonpass.get("passes") is not False:
            raise ValueError("native suffix nonpass was not preserved")

        expected_positions = {
            "0": {
                "hidden_state_index": leading["selection"]["hidden_state_index"],
                "desired_margin": leading["selection"]["desired_margin"],
                "norm_cap": leading["selection"]["norm_cap"],
            },
            "1": {
                "hidden_state_index": suffix["hidden_state_index"],
                "desired_margin": suffix["positions"]["1"]["selection"]["desired_margin"],
                "norm_cap": suffix["positions"]["1"]["selection"]["norm_cap"],
            },
            "2": {
                "hidden_state_index": suffix["hidden_state_index"],
                "desired_margin": suffix["positions"]["2"]["selection"]["desired_margin"],
                "norm_cap": suffix["positions"]["2"]["selection"]["norm_cap"],
            },
        }
        if positions != expected_positions:
            raise ValueError("compiler positions differ from selection")
        if audit["writer"] != {
            "family": "prompt_local_margin_compiler",
            "position_compilers": positions,
        }:
            raise ValueError("audit writer differs from manifest")

        latent = audit["conditions"]["latent_read_compute_compiler_write"]
        shuffled = audit["conditions"]["shuffled_read_compute_compiler_write"]
        shuffled_random = audit["conditions"]["shuffled_random_norm_matched"]
        paired = audit["gate"]["paired_metrics"]["latent_read_compute_compiler_write"]
        thresholds = {
            "minimum_audit_exact_accuracy": latent["true_result_accuracy"],
            "minimum_audit_reader_pair_accuracy": audit["reader"]["metrics"]["pair_accuracy"],
            "minimum_base_error_recovery": paired["base_error_recovery"],
            "minimum_base_correct_preservation": paired["base_correct_preservation"],
            "minimum_shuffled_target_accuracy": shuffled["evaluation_target_accuracy"],
            "minimum_shuffled_target_advantage_over_random": (
                shuffled["evaluation_target_accuracy"]
                - shuffled_random["evaluation_target_accuracy"]
            ),
        }
        for name, observed in thresholds.items():
            if observed < float(self.evidence[name]):
                raise ValueError(f"compiler-graft audit failed {name}")

    def load_reader(self, root: Path):
        self.verify(root)
        path = root / str(self.reader["manifest"]["path"])
        return OperandReaderManifest.load(path).load_reader(root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an audited compiler-graft interface manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or args.manifest.parent.parent
    manifest = CompilerGraftManifest.load(args.manifest)
    manifest.verify(root)
    print(f"valid compiler-graft interface: {manifest.name} (model: {manifest.model_id})")


if __name__ == "__main__":
    main()
