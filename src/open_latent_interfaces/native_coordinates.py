from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_digit_prototypes(
    states: torch.Tensor,
    digits: torch.Tensor,
    basis: torch.Tensor,
    *,
    class_count: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average native coordinates by digit class."""
    if states.ndim != 2 or basis.ndim != 2:
        raise ValueError("states and basis must be matrices")
    if states.shape[1] != basis.shape[1]:
        raise ValueError("state width must match basis width")
    if digits.shape != (states.shape[0],):
        raise ValueError("digits must align with states")
    if class_count < 2:
        raise ValueError("at least two classes are required")
    digits = digits.long()
    if bool(((digits < 0) | (digits >= class_count)).any()):
        raise ValueError("digit class is outside the configured range")
    coordinates = states.float() @ basis.float().T
    prototypes = torch.empty(
        (class_count, basis.shape[0]),
        dtype=coordinates.dtype,
    )
    counts = torch.empty(class_count, dtype=torch.long)
    for digit in range(class_count):
        selected = coordinates[digits == digit]
        if selected.shape[0] == 0:
            raise ValueError(f"no fit state for digit class {digit}")
        prototypes[digit] = selected.mean(dim=0)
        counts[digit] = selected.shape[0]
    return prototypes, counts


@dataclass(frozen=True)
class CoordinateWrite:
    delta: torch.Tensor
    hard_gate: torch.Tensor


class NativeCoordinateWriter:
    """Typed coordinate replacement in a frozen residual subspace."""

    def __init__(
        self,
        basis: torch.Tensor,
        prototypes: torch.Tensor,
        *,
        scale: float,
        norm_cap: float,
    ) -> None:
        basis = basis.detach().float().cpu()
        prototypes = prototypes.detach().float().cpu()
        if basis.ndim != 2 or prototypes.ndim != 2:
            raise ValueError("basis and prototypes must be matrices")
        if prototypes.shape[1] != basis.shape[0]:
            raise ValueError("prototype width must equal basis rank")
        if min(scale, norm_cap) <= 0:
            raise ValueError("scale and norm cap must be positive")
        if not bool(torch.isfinite(basis).all() and torch.isfinite(prototypes).all()):
            raise ValueError("basis and prototypes must be finite")
        gram = basis @ basis.T
        identity = torch.eye(basis.shape[0])
        if not torch.allclose(gram, identity, atol=1e-3, rtol=1e-3):
            raise ValueError("basis rows must be approximately orthonormal")
        self.basis = basis
        self.prototypes = prototypes
        self.scale = float(scale)
        self.norm_cap = float(norm_cap)

    @property
    def rank(self) -> int:
        return self.basis.shape[0]

    @property
    def residual_width(self) -> int:
        return self.basis.shape[1]

    def raw_delta(
        self,
        states: torch.Tensor,
        requested_digits: torch.Tensor,
    ) -> torch.Tensor:
        states = states.detach().float().cpu()
        requested_digits = requested_digits.detach().long().cpu()
        if states.ndim != 2 or states.shape[1] != self.residual_width:
            raise ValueError("states do not match interface residual width")
        if requested_digits.shape != (states.shape[0],):
            raise ValueError("requested digits must align with states")
        if bool(
            (
                (requested_digits < 0)
                | (requested_digits >= self.prototypes.shape[0])
            ).any()
        ):
            raise ValueError("requested digit is outside prototype classes")
        desired = self.prototypes[requested_digits]
        recipient = states @ self.basis.T
        return (desired - recipient) @ self.basis

    def write(
        self,
        states: torch.Tensor,
        requested_digits: torch.Tensor,
        *,
        base_logits: torch.Tensor,
        requested_token_ids: torch.Tensor,
    ) -> CoordinateWrite:
        states = states.detach().float().cpu()
        base_logits = base_logits.detach().float().cpu()
        requested_token_ids = requested_token_ids.detach().long().cpu()
        if base_logits.ndim != 2 or base_logits.shape[0] != states.shape[0]:
            raise ValueError("base logits must align with states")
        if requested_token_ids.shape != (states.shape[0],):
            raise ValueError("requested token IDs must align with states")
        raw = self.raw_delta(states, requested_digits)
        delta = raw * self.scale
        maximum = states.norm(dim=1) * self.norm_cap
        factor = (maximum / delta.norm(dim=1).clamp_min(1e-12)).clamp(max=1.0)
        delta = delta * factor[:, None]
        hard_gate = base_logits.argmax(dim=1) == requested_token_ids
        delta[hard_gate] = 0
        return CoordinateWrite(delta=delta, hard_gate=hard_gate)

    @classmethod
    def from_artifacts(
        cls,
        *,
        basis_path: Path,
        prototype_path: Path,
        rank: int,
        scale: float,
        norm_cap: float,
        basis_key: str = "delta_basis",
        prototype_key: str = "digit",
    ) -> NativeCoordinateWriter:
        basis_tensors = load_file(str(basis_path))
        prototype_tensors = load_file(str(prototype_path))
        if basis_key not in basis_tensors:
            raise ValueError(f"missing basis tensor: {basis_key}")
        if prototype_key not in prototype_tensors:
            raise ValueError(f"missing prototype tensor: {prototype_key}")
        basis = basis_tensors[basis_key]
        if rank < 1 or rank > basis.shape[0]:
            raise ValueError("rank is outside the stored basis")
        return cls(
            basis[:rank],
            prototype_tensors[prototype_key],
            scale=scale,
            norm_cap=norm_cap,
        )


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    sha256: str
    key: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArtifactReference:
        required = ("path", "sha256", "key")
        if any(name not in value for name in required):
            raise ValueError("artifact reference requires path, sha256, and key")
        digest = str(value["sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("artifact SHA-256 must be lowercase hexadecimal")
        return cls(
            path=str(value["path"]),
            sha256=digest,
            key=str(value["key"]),
        )

    def verify(self, root: Path) -> Path:
        resolved = root / self.path
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if file_sha256(resolved) != self.sha256:
            raise ValueError(f"artifact hash mismatch: {self.path}")
        return resolved


@dataclass(frozen=True)
class NativeCoordinatePosition:
    answer_position: int
    hidden_state_index: int
    scale: float
    norm_cap: float
    rank: int
    prototypes: ArtifactReference


@dataclass(frozen=True)
class NativeCoordinateManifest:
    name: str
    model_id: str
    model_revision: str
    residual_width: int
    basis: ArtifactReference
    positions: dict[int, NativeCoordinatePosition]
    evidence: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NativeCoordinateManifest:
        if value.get("schema_version") != "oli.native-coordinate-interface/v1":
            raise ValueError("unsupported native-coordinate manifest schema")
        model = value["model"]
        representation = value["representation"]
        positions = {}
        for key, row in value["positions"].items():
            position = int(key)
            if position != int(row["answer_position"]):
                raise ValueError("position key and answer_position disagree")
            positions[position] = NativeCoordinatePosition(
                answer_position=position,
                hidden_state_index=int(row["hidden_state_index"]),
                scale=float(row["scale"]),
                norm_cap=float(row["norm_cap"]),
                rank=int(row["rank"]),
                prototypes=ArtifactReference.from_dict(row["prototypes"]),
            )
        if not positions:
            raise ValueError("manifest requires at least one coordinate position")
        return cls(
            name=str(value["name"]),
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            residual_width=int(representation["residual_width"]),
            basis=ArtifactReference.from_dict(representation["basis"]),
            positions=positions,
            evidence=dict(value["evidence"]),
        )

    @classmethod
    def load(cls, path: Path) -> NativeCoordinateManifest:
        return cls.from_dict(json.loads(path.read_text()))

    def verify(self, root: Path) -> None:
        basis_path = self.basis.verify(root)
        tensors = load_file(str(basis_path))
        if self.basis.key not in tensors:
            raise ValueError(f"missing basis tensor: {self.basis.key}")
        basis = tensors[self.basis.key]
        if basis.ndim != 2 or basis.shape[1] != self.residual_width:
            raise ValueError("basis tensor does not match manifest residual width")
        for position in self.positions.values():
            prototype_path = position.prototypes.verify(root)
            prototype_tensors = load_file(str(prototype_path))
            if position.prototypes.key not in prototype_tensors:
                raise ValueError(
                    f"missing prototype tensor: {position.prototypes.key}"
                )
            prototypes = prototype_tensors[position.prototypes.key]
            if position.rank > basis.shape[0]:
                raise ValueError("position rank exceeds stored basis")
            if prototypes.shape != (10, position.rank):
                raise ValueError("prototype tensor must have shape [10, rank]")
        if self.evidence.get("audit_gate_passed") is not True:
            raise ValueError("manifest evidence must record a passing audit gate")
        if self.evidence.get("audit_runs") != 1:
            raise ValueError("manifest evidence must record exactly one audit run")
        for name in ("development_result", "audit_config", "audit_result"):
            reference = self.evidence.get(name)
            if not isinstance(reference, dict):
                raise ValueError(f"missing evidence reference: {name}")
            path = root / str(reference["path"])
            expected = str(reference["sha256"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if file_sha256(path) != expected:
                raise ValueError(f"evidence hash mismatch: {name}")

    def load_writer(
        self,
        answer_position: int,
        *,
        root: Path,
    ) -> NativeCoordinateWriter:
        if answer_position not in self.positions:
            raise KeyError(f"no interface for answer position {answer_position}")
        self.verify(root)
        position = self.positions[answer_position]
        return NativeCoordinateWriter.from_artifacts(
            basis_path=root / self.basis.path,
            prototype_path=root / position.prototypes.path,
            rank=position.rank,
            scale=position.scale,
            norm_cap=position.norm_cap,
            basis_key=self.basis.key,
            prototype_key=position.prototypes.key,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a native-coordinate interface manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or args.manifest.parent.parent
    manifest = NativeCoordinateManifest.load(args.manifest)
    manifest.verify(root)
    positions = ", ".join(str(value) for value in sorted(manifest.positions))
    print(
        f"valid native-coordinate interface: {manifest.name} "
        f"(positions: {positions})"
    )


if __name__ == "__main__":
    main()
