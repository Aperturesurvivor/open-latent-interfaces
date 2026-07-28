from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DigitSubspace:
    """Low-rank native state prototypes for one answer position."""

    classes: tuple[int, ...]
    centroids: torch.Tensor
    basis: torch.Tensor

    def __post_init__(self) -> None:
        if self.centroids.ndim != 2 or self.basis.ndim != 2:
            raise ValueError("centroids and basis must be matrices")
        if self.centroids.shape[0] != len(self.classes):
            raise ValueError("one centroid is required per class")
        if self.centroids.shape[1] != self.basis.shape[1]:
            raise ValueError("centroids and basis must share a hidden width")

    def write_delta(
        self,
        states: torch.Tensor,
        target_digits: torch.Tensor,
        *,
        rank: int,
        scale: float,
    ) -> torch.Tensor:
        """Replace the target-subspace coordinates with a digit prototype."""

        if states.ndim != 2 or states.shape[1] != self.centroids.shape[1]:
            raise ValueError("states must have shape [examples, hidden_width]")
        if target_digits.shape != (states.shape[0],):
            raise ValueError("one target digit is required per state")
        if rank < 1 or rank > self.basis.shape[0]:
            raise ValueError("rank is outside the fitted basis")
        class_rows = {value: index for index, value in enumerate(self.classes)}
        try:
            rows = torch.tensor(
                [class_rows[int(value)] for value in target_digits.tolist()],
                dtype=torch.long,
            )
        except KeyError as error:
            raise ValueError(f"target digit {error.args[0]} was not fitted") from error
        target_states = self.centroids[rows]
        basis = self.basis[:rank]
        return ((target_states - states) @ basis.T) @ basis * scale


def fit_digit_subspace(
    states: torch.Tensor,
    digits: torch.Tensor,
) -> DigitSubspace:
    """Fit the between-class centroid subspace for native next-digit states."""

    if states.ndim != 2 or digits.shape != (states.shape[0],):
        raise ValueError("states and digits must align by example")
    classes = tuple(sorted({int(value) for value in digits.tolist()}))
    if len(classes) < 2:
        raise ValueError("at least two digit classes are required")
    centroids = torch.stack(
        [states[digits == value].mean(dim=0) for value in classes]
    )
    centered = centroids - centroids.mean(dim=0)
    _, _, basis = torch.linalg.svd(centered, full_matrices=False)
    return DigitSubspace(classes=classes, centroids=centroids, basis=basis)
