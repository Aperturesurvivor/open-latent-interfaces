from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


@dataclass(frozen=True)
class OperandTokenPositions:
    operand_a: tuple[int, ...]
    operand_b: tuple[int, ...]


@dataclass(frozen=True)
class NearestCentroidDigitReader:
    """Decode decimal digits with a frozen linear nearest-centroid rule."""

    classes: torch.Tensor
    centroids: torch.Tensor

    def __post_init__(self) -> None:
        classes = self.classes.detach().long().cpu()
        centroids = self.centroids.detach().float().cpu()
        if classes.ndim != 1 or centroids.ndim != 2:
            raise ValueError("classes and centroids must be a vector and matrix")
        if centroids.shape[0] != classes.shape[0]:
            raise ValueError("one centroid is required per class")
        if len(set(classes.tolist())) != classes.shape[0]:
            raise ValueError("reader classes must be unique")
        if not bool(torch.isfinite(centroids).all()):
            raise ValueError("reader centroids must be finite")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "centroids", centroids)

    @property
    def residual_width(self) -> int:
        return self.centroids.shape[1]

    def scores(self, states: torch.Tensor) -> torch.Tensor:
        states = states.detach().float().cpu()
        if states.ndim != 2 or states.shape[1] != self.residual_width:
            raise ValueError("states do not match reader residual width")
        # Negative squared distance with the state-only term removed.
        return (
            2.0 * states @ self.centroids.T
            - self.centroids.square().sum(dim=1)
        )

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        return self.classes[self.scores(states).argmax(dim=1)]


def fit_nearest_centroid_digit_reader(
    states: torch.Tensor,
    digits: torch.Tensor,
    *,
    classes: tuple[int, ...] = tuple(range(10)),
) -> tuple[NearestCentroidDigitReader, torch.Tensor]:
    """Fit one native-state centroid per decimal digit."""

    states = states.detach().float().cpu()
    digits = digits.detach().long().cpu()
    if states.ndim != 2 or digits.shape != (states.shape[0],):
        raise ValueError("states and digit labels must align")
    if len(set(classes)) != len(classes):
        raise ValueError("classes must be unique")
    centroids = []
    counts = []
    for digit in classes:
        selected = states[digits == digit]
        if selected.shape[0] == 0:
            raise ValueError(f"no fit state for digit class {digit}")
        centroids.append(selected.mean(dim=0))
        counts.append(selected.shape[0])
    return (
        NearestCentroidDigitReader(
            classes=torch.tensor(classes, dtype=torch.int64),
            centroids=torch.stack(centroids),
        ),
        torch.tensor(counts, dtype=torch.int64),
    )


def locate_operand_digit_tokens(
    tokenizer: Any,
    rendered_prompt: str,
    prompt_content: str,
    operand_a: int,
    operand_b: int,
) -> OperandTokenPositions:
    """Resolve one-token decimal digits inside the exact user-prompt content."""

    content_start = rendered_prompt.find(prompt_content)
    if content_start < 0:
        raise ValueError("rendered prompt does not contain the source content")
    a_text = str(operand_a)
    b_text = str(operand_b)
    a_relative = prompt_content.find(a_text)
    if a_relative < 0:
        raise ValueError("prompt content does not contain operand A")
    b_relative = prompt_content.find(b_text, a_relative + len(a_text))
    if b_relative < 0:
        raise ValueError("prompt content does not contain operand B after operand A")
    spans = (
        (
            content_start + a_relative,
            content_start + a_relative + len(a_text),
            a_text,
        ),
        (
            content_start + b_relative,
            content_start + b_relative + len(b_text),
            b_text,
        ),
    )
    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(value) for value in encoded["offset_mapping"]]
    token_ids = encoded["input_ids"]
    located = []
    for start, stop, expected in spans:
        positions = [
            index
            for index, (left, right) in enumerate(offsets)
            if left >= start and right <= stop and right > left
        ]
        if len(positions) != len(expected):
            raise ValueError("operand digits must each occupy one token")
        decoded = "".join(
            tokenizer.decode([token_ids[position]]) for position in positions
        )
        if decoded != expected:
            raise ValueError("operand token span does not decode exactly")
        located.append(tuple(positions))
    return OperandTokenPositions(
        operand_a=located[0],
        operand_b=located[1],
    )


def reconstruct_decimal_digits(digits: list[int]) -> int:
    if not digits:
        raise ValueError("at least one digit is required")
    if any(digit < 0 or digit > 9 for digit in digits):
        raise ValueError("decimal digits must be between zero and nine")
    return int("".join(str(digit) for digit in digits))


@dataclass(frozen=True)
class OperandReaderManifest:
    schema_version: str
    name: str
    model_id: str
    model_revision: str
    residual_width: int
    hidden_state_index: int
    artifact_path: str
    artifact_sha256: str
    classes_key: str
    centroids_key: str
    counts_key: str
    locator: dict[str, Any]
    evidence: dict[str, Any]
    claim_boundary: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OperandReaderManifest:
        supported = {
            "oli.operand-reader-interface/v1",
            "oli.operand-reader-interface/v2",
        }
        if value.get("schema_version") not in supported:
            raise ValueError("unsupported operand-reader manifest schema")
        model = value["model"]
        representation = value["representation"]
        tensors = value["reader"]["tensors"]
        digest = str(tensors["sha256"])
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ValueError("artifact SHA-256 must be lowercase hexadecimal")
        return cls(
            schema_version=str(value["schema_version"]),
            name=str(value["name"]),
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            residual_width=int(representation["residual_width"]),
            hidden_state_index=int(value["reader"]["hidden_state_index"]),
            artifact_path=str(tensors["path"]),
            artifact_sha256=digest,
            classes_key=str(tensors["classes_key"]),
            centroids_key=str(tensors["centroids_key"]),
            counts_key=str(tensors["counts_key"]),
            locator=dict(value["locator"]),
            evidence=dict(value["evidence"]),
            claim_boundary=tuple(str(row) for row in value["claim_boundary"]),
        )

    @classmethod
    def load(cls, path: Path) -> OperandReaderManifest:
        return cls.from_dict(json.loads(path.read_text()))

    def verify(self, root: Path) -> None:
        if self.hidden_state_index < 1 or self.residual_width < 1:
            raise ValueError("reader boundary and width must be positive")
        artifact = root / self.artifact_path
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        observed = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if observed != self.artifact_sha256:
            raise ValueError("operand-reader artifact hash mismatch")
        tensors = load_file(str(artifact))
        required = (self.classes_key, self.centroids_key, self.counts_key)
        if any(key not in tensors for key in required):
            raise ValueError("operand-reader artifact is missing tensors")
        classes = tensors[self.classes_key]
        centroids = tensors[self.centroids_key]
        counts = tensors[self.counts_key]
        if classes.shape != (10,) or classes.tolist() != list(range(10)):
            raise ValueError("reader classes must be decimal digits")
        if centroids.shape != (10, self.residual_width):
            raise ValueError("reader centroids have the wrong shape")
        if counts.shape != (10,) or bool((counts < 1).any()):
            raise ValueError("reader fit counts are invalid")
        if self.locator.get("type") != "external_semantic_operand_spans":
            raise ValueError("manifest must declare the external locator")
        if self.evidence.get("reader_audit_gate_passed") is not True:
            raise ValueError("manifest must record a passing reader audit")
        if self.evidence.get("audit_runs") != 1:
            raise ValueError("manifest must record exactly one audit run")
        if self.schema_version == "oli.operand-reader-interface/v1":
            evidence_sources = self.evidence
            required_sources = {
                "selection_config",
                "selection_result",
                "development_result",
                "development_metric_correction",
                "audit_config",
                "audit_result",
            }
        else:
            evidence_sources = self.evidence.get("sources")
            if not isinstance(evidence_sources, dict):
                raise ValueError("v2 reader evidence must contain sources")
            required_sources = {
                "selection_config",
                "selection_result",
                "audit_config",
                "audit_result",
            }
        if not required_sources <= set(evidence_sources):
            missing = sorted(required_sources - set(evidence_sources))
            raise ValueError(f"missing evidence reference: {missing[0]}")
        source_names = (
            required_sources
            if self.schema_version == "oli.operand-reader-interface/v1"
            else set(evidence_sources)
        )
        for name in source_names:
            reference = evidence_sources[name]
            if not isinstance(reference, dict):
                raise ValueError(f"invalid evidence reference: {name}")
            path = root / str(reference["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != str(reference["sha256"]):
                raise ValueError(f"evidence hash mismatch: {name}")
        selection = json.loads(
            (root / evidence_sources["selection_result"]["path"]).read_text()
        )
        if selection.get("passes") is not True:
            raise ValueError("reader selection result did not pass")
        expected_model = {
            "id": self.model_id,
            "revision": self.model_revision,
        }
        if self.schema_version == "oli.operand-reader-interface/v2":
            if selection.get("model") != expected_model:
                raise ValueError("reader selection/model mismatch")
            if int(selection["selection"]["hidden_state_index"]) != (
                self.hidden_state_index
            ):
                raise ValueError("reader selection boundary mismatch")
        audit = json.loads(
            (root / evidence_sources["audit_result"]["path"]).read_text()
        )
        if (
            self.schema_version == "oli.operand-reader-interface/v2"
            and audit.get("audit_runs") != 1
        ):
            raise ValueError("reader evidence is not a one-shot audit")
        if self.schema_version == "oli.operand-reader-interface/v2":
            if audit.get("model") != expected_model:
                raise ValueError("reader audit/model mismatch")
            if int(audit["reader"]["hidden_state_index"]) != (
                self.hidden_state_index
            ):
                raise ValueError("reader audit boundary mismatch")
        observed_pair_accuracy = audit["reader"]["metrics"]["pair_accuracy"]
        if observed_pair_accuracy < float(
            self.evidence["minimum_audit_pair_accuracy"]
        ):
            raise ValueError("reader audit metric does not pass")

    def load_reader(self, root: Path) -> NearestCentroidDigitReader:
        self.verify(root)
        tensors = load_file(str(root / self.artifact_path))
        return NearestCentroidDigitReader(
            classes=tensors[self.classes_key],
            centroids=tensors[self.centroids_key],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an operand-reader interface manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root or args.manifest.parent.parent
    manifest = OperandReaderManifest.load(args.manifest)
    manifest.verify(root)
    print(
        f"valid operand-reader interface: {manifest.name} "
        f"(hidden-state index: {manifest.hidden_state_index})"
    )


if __name__ == "__main__":
    main()
