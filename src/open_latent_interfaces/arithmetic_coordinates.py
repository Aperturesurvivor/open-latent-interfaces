from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from open_latent_interfaces.native_coordinates import ArtifactReference, file_sha256


class TokenLocalTransportWriter:
    """Add a fixed or class-conditioned transport vector at one token."""

    def __init__(
        self,
        vectors: torch.Tensor,
        *,
        scale: float,
        class_labels: torch.Tensor | None = None,
    ) -> None:
        vectors = vectors.detach().float().cpu()
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        if vectors.ndim != 2 or vectors.shape[0] < 1:
            raise ValueError("transport vectors must have shape [classes, width]")
        if not bool(torch.isfinite(vectors).all()):
            raise ValueError("transport vectors must be finite")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if class_labels is None:
            if vectors.shape[0] != 1:
                raise ValueError("multi-vector writer requires class labels")
            labels = None
        else:
            labels = class_labels.detach().long().cpu()
            if labels.shape != (vectors.shape[0],):
                raise ValueError("class labels must align with transport vectors")
            if len(set(labels.tolist())) != labels.shape[0]:
                raise ValueError("class labels must be unique")
        self.vectors = vectors
        self.class_labels = labels
        self.scale = float(scale)

    @property
    def residual_width(self) -> int:
        return self.vectors.shape[1]

    @property
    def coordinate_count(self) -> int:
        return self.vectors.shape[0]

    def delta(
        self,
        *,
        batch_size: int | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one scaled residual delta per requested example."""

        if self.class_labels is None:
            if class_labels is not None:
                raise ValueError("fixed transport does not accept class labels")
            if batch_size is None or batch_size < 1:
                raise ValueError("fixed transport requires a positive batch size")
            return self.vectors.repeat(batch_size, 1) * self.scale
        if batch_size is not None:
            raise ValueError("class-conditioned transport infers its batch size")
        if class_labels is None or class_labels.ndim != 1:
            raise ValueError("class-conditioned transport requires label vector")
        requested = class_labels.detach().long().cpu()
        lookup = {
            int(label): index for index, label in enumerate(self.class_labels.tolist())
        }
        try:
            indices = torch.tensor([lookup[int(label)] for label in requested])
        except KeyError as error:
            raise ValueError(f"unsupported transport class: {error.args[0]}") from error
        return self.vectors[indices] * self.scale


@dataclass(frozen=True)
class ArithmeticCoordinateSpec:
    name: str
    hidden_state_index: int
    scale: float
    coordinate_rank: int
    token_selector: dict[str, Any]
    semantics: str
    vectors: ArtifactReference
    class_labels: ArtifactReference | None = None


@dataclass(frozen=True)
class ArithmeticCoordinateManifest:
    schema_version: str
    name: str
    model_id: str
    model_revision: str
    residual_width: int
    interfaces: dict[str, ArithmeticCoordinateSpec]
    evidence: dict[str, Any]
    claim_boundary: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArithmeticCoordinateManifest:
        if value.get("schema_version") != "oli.arithmetic-coordinate-interface/v1":
            raise ValueError("unsupported arithmetic-coordinate manifest schema")
        model = value["model"]
        representation = value["representation"]
        interfaces = {}
        for name, row in value["interfaces"].items():
            interfaces[name] = ArithmeticCoordinateSpec(
                name=name,
                hidden_state_index=int(row["hidden_state_index"]),
                scale=float(row["scale"]),
                coordinate_rank=int(row["coordinate_rank"]),
                token_selector=dict(row["token_selector"]),
                semantics=str(row["semantics"]),
                vectors=ArtifactReference.from_dict(row["vectors"]),
                class_labels=(
                    None
                    if "class_labels" not in row
                    else ArtifactReference.from_dict(row["class_labels"])
                ),
            )
        if not interfaces:
            raise ValueError("manifest requires at least one interface")
        return cls(
            schema_version=str(value["schema_version"]),
            name=str(value["name"]),
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            residual_width=int(representation["residual_width"]),
            interfaces=interfaces,
            evidence=dict(value["evidence"]),
            claim_boundary=tuple(str(item) for item in value["claim_boundary"]),
        )

    @classmethod
    def load(cls, path: Path) -> ArithmeticCoordinateManifest:
        return cls.from_dict(json.loads(path.read_text()))

    def verify(self, root: Path) -> None:
        for spec in self.interfaces.values():
            if spec.hidden_state_index < 1 or spec.scale <= 0:
                raise ValueError("interface boundary and scale must be positive")
            vector_path = spec.vectors.verify(root)
            tensors = load_file(str(vector_path))
            if spec.vectors.key not in tensors:
                raise ValueError(f"missing transport tensor: {spec.vectors.key}")
            vectors = tensors[spec.vectors.key]
            if vectors.ndim == 1:
                shape = (1, vectors.shape[0])
            elif vectors.ndim == 2:
                shape = tuple(vectors.shape)
            else:
                raise ValueError("transport tensor must be a vector or matrix")
            if shape[1] != self.residual_width:
                raise ValueError("transport tensor has wrong residual width")
            if spec.coordinate_rank < 1 or spec.coordinate_rank > shape[0]:
                raise ValueError("coordinate rank exceeds stored vectors")
            if spec.class_labels is None:
                if shape[0] != 1:
                    raise ValueError("multi-vector interface requires class labels")
            else:
                label_path = spec.class_labels.verify(root)
                if label_path != vector_path:
                    raise ValueError("labels and vectors must share one artifact")
                label_tensors = load_file(str(label_path))
                if spec.class_labels.key not in label_tensors:
                    raise ValueError(
                        f"missing class-label tensor: {spec.class_labels.key}"
                    )
                if label_tensors[spec.class_labels.key].shape != (shape[0],):
                    raise ValueError("class labels do not align with vectors")
        if self.evidence.get("audit_gate_passed") is not True:
            raise ValueError("manifest must record a passing audit")
        if self.evidence.get("audit_runs") != 1:
            raise ValueError("manifest must record exactly one audit run")
        for name in (
            "development_correction",
            "audit_config",
            "audit_result",
        ):
            reference = self.evidence.get(name)
            if not isinstance(reference, dict):
                raise ValueError(f"missing evidence reference: {name}")
            path = root / str(reference["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if file_sha256(path) != str(reference["sha256"]):
                raise ValueError(f"evidence hash mismatch: {name}")

    def load_writer(
        self,
        interface_name: str,
        *,
        root: Path,
    ) -> TokenLocalTransportWriter:
        if interface_name not in self.interfaces:
            raise KeyError(f"unknown arithmetic interface: {interface_name}")
        self.verify(root)
        spec = self.interfaces[interface_name]
        tensors = load_file(str(root / spec.vectors.path))
        labels = (
            None
            if spec.class_labels is None
            else tensors[spec.class_labels.key]
        )
        return TokenLocalTransportWriter(
            tensors[spec.vectors.key],
            scale=spec.scale,
            class_labels=labels,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an arithmetic-coordinate interface manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or args.manifest.parent.parent
    manifest = ArithmeticCoordinateManifest.load(args.manifest)
    manifest.verify(root)
    interfaces = ", ".join(sorted(manifest.interfaces))
    print(
        f"valid arithmetic-coordinate interface: {manifest.name} "
        f"(interfaces: {interfaces})"
    )


if __name__ == "__main__":
    main()
