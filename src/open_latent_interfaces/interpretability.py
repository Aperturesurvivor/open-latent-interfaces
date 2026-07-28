from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

SCHEMA_VERSION = "oli.interpretability-artifact/v1"
EvidenceStatus = Literal["hypothesis", "corroborated", "contradicted", "inconclusive"]


def _float32_array(value: Any) -> np.ndarray:
    array = np.asarray(
        value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else value,
        dtype="<f4",
    )
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("vectors must be non-empty and finite")
    return np.ascontiguousarray(array)


def vector_sha256(value: Any) -> str:
    """Hash the canonical little-endian float32 representation of a vector."""

    return hashlib.sha256(_float32_array(value).tobytes()).hexdigest()


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _artifact_digest(
    *,
    example_id: str,
    site: LatentSite,
    method: MethodProvenance,
    observation: dict[str, Any],
    reconstruction: dict[str, Any] | None,
) -> str:
    core = {
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "site": asdict(site),
        "method": asdict(method),
        "observation": observation,
        "reconstruction": reconstruction,
    }
    encoded = json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VectorRecord:
    sha256: str
    dtype: str
    shape: tuple[int, ...]
    values: list[float] | None = None

    @classmethod
    def from_value(cls, value: Any, *, include_values: bool = False) -> VectorRecord:
        array = _float32_array(value)
        return cls(
            sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            dtype="float32-le",
            shape=tuple(array.shape),
            values=array.reshape(-1).tolist() if include_values else None,
        )


@dataclass(frozen=True)
class LatentSite:
    target_model: str
    target_model_revision: str
    hidden_state_index: int
    token_position: int
    activation_sha256: str

    def __post_init__(self) -> None:
        if not self.target_model or not self.target_model_revision:
            raise ValueError("target model and immutable revision are required")
        if self.hidden_state_index < 0:
            raise ValueError("hidden-state index must be non-negative")
        if len(self.activation_sha256) != 64:
            raise ValueError("activation_sha256 must be a SHA-256 hex digest")
        int(self.activation_sha256, 16)

    @classmethod
    def from_activation(
        cls,
        activation: Any,
        *,
        target_model: str,
        target_model_revision: str,
        hidden_state_index: int,
        token_position: int,
    ) -> LatentSite:
        return cls(
            target_model=target_model,
            target_model_revision=target_model_revision,
            hidden_state_index=hidden_state_index,
            token_position=token_position,
            activation_sha256=vector_sha256(activation),
        )


@dataclass(frozen=True)
class MethodProvenance:
    family: str
    implementation: str
    repository: str
    revision: str
    license: str
    checkpoint: str | None = None
    checkpoint_revision: str | None = None

    def __post_init__(self) -> None:
        if not all((self.family, self.implementation, self.repository, self.license)):
            raise ValueError("complete method provenance is required")
        if len(self.revision) != 40:
            raise ValueError("method revision must be an exact 40-character commit")
        int(self.revision, 16)


@dataclass(frozen=True)
class Corroboration:
    status: EvidenceStatus = "hypothesis"
    artifact_ids: tuple[str, ...] = ()
    method_families: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True)
class InterpretabilityArtifact:
    schema_version: str
    artifact_id: str
    created_at: str
    example_id: str
    site: LatentSite
    method: MethodProvenance
    observation: dict[str, Any]
    reconstruction: dict[str, Any] | None
    corroboration: Corroboration
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if len(self.artifact_id) != 64:
            raise ValueError("artifact_id must be a SHA-256 hex digest")
        int(self.artifact_id, 16)
        json.dumps(self.observation, allow_nan=False)
        json.dumps(self.reconstruction, allow_nan=False)
        expected_id = _artifact_digest(
            example_id=self.example_id,
            site=self.site,
            method=self.method,
            observation=self.observation,
            reconstruction=self.reconstruction,
        )
        if self.artifact_id != expected_id:
            raise ValueError("artifact_id does not match the scientific payload")
        if self.corroboration.status == "corroborated":
            if not self.corroboration.artifact_ids:
                raise ValueError("corroborated artifacts require independent artifact IDs")
            if not any(
                family != self.method.family
                for family in self.corroboration.method_families
            ):
                raise ValueError("corroboration must include a different method family")

    @classmethod
    def create(
        cls,
        *,
        example_id: str,
        site: LatentSite,
        method: MethodProvenance,
        observation: dict[str, Any],
        reconstruction: dict[str, Any] | None = None,
        limitations: tuple[str, ...] = (),
        created_at: str | None = None,
    ) -> InterpretabilityArtifact:
        normalized_observation = _json_normalize(observation)
        normalized_reconstruction = _json_normalize(reconstruction)
        return cls(
            schema_version=SCHEMA_VERSION,
            artifact_id=_artifact_digest(
                example_id=example_id,
                site=site,
                method=method,
                observation=normalized_observation,
                reconstruction=normalized_reconstruction,
            ),
            created_at=created_at or datetime.now(UTC).isoformat(),
            example_id=example_id,
            site=site,
            method=method,
            observation=normalized_observation,
            reconstruction=normalized_reconstruction,
            corroboration=Corroboration(),
            limitations=limitations,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InterpretabilityArtifact:
        return cls(
            schema_version=value["schema_version"],
            artifact_id=value["artifact_id"],
            created_at=value["created_at"],
            example_id=value["example_id"],
            site=LatentSite(**value["site"]),
            method=MethodProvenance(**value["method"]),
            observation=value["observation"],
            reconstruction=value.get("reconstruction"),
            corroboration=Corroboration(
                status=value["corroboration"]["status"],
                artifact_ids=tuple(value["corroboration"].get("artifact_ids", ())),
                method_families=tuple(
                    value["corroboration"].get("method_families", ())
                ),
                note=value["corroboration"].get("note"),
            ),
            limitations=tuple(value.get("limitations", ())),
        )


def corroborate(
    primary: InterpretabilityArtifact,
    independent: InterpretabilityArtifact,
    *,
    status: Literal["corroborated", "contradicted", "inconclusive"],
    note: str,
) -> InterpretabilityArtifact:
    """Attach a judgment from a genuinely different method at the exact same site."""

    if primary.site != independent.site:
        raise ValueError("corroborating artifacts must refer to the exact same activation")
    if primary.method.family == independent.method.family:
        raise ValueError("corroboration requires a different method family")
    return replace(
        primary,
        corroboration=Corroboration(
            status=status,
            artifact_ids=(independent.artifact_id,),
            method_families=(independent.method.family,),
            note=note,
        ),
    )


def write_jsonl(path: Path, artifacts: list[InterpretabilityArtifact]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for artifact in artifacts:
            handle.write(json.dumps(artifact.to_dict(), sort_keys=True, allow_nan=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[InterpretabilityArtifact]:
    with path.open(encoding="utf-8") as handle:
        return [
            InterpretabilityArtifact.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        ]
