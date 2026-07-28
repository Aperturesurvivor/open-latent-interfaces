from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


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
