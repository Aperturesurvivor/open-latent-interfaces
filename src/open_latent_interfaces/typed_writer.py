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


@dataclass(frozen=True)
class TransportSubspace:
    """Low-rank class-conditioned recipient-to-native transport deltas."""

    classes: tuple[int, ...]
    class_deltas: torch.Tensor
    basis: torch.Tensor

    def write_delta(
        self,
        target_digits: torch.Tensor,
        *,
        rank: int,
        scale: float,
    ) -> torch.Tensor:
        if target_digits.ndim != 1:
            raise ValueError("target digits must be a vector")
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
        prototype_deltas = self.class_deltas[rows]
        basis = self.basis[:rank]
        return (prototype_deltas @ basis.T) @ basis * scale


def fit_transport_subspace(
    deltas: torch.Tensor,
    digits: torch.Tensor,
) -> TransportSubspace:
    """Fit class-conditioned mean transports and their uncentered subspace."""

    if deltas.ndim != 2 or digits.shape != (deltas.shape[0],):
        raise ValueError("deltas and digits must align by example")
    classes = tuple(sorted({int(value) for value in digits.tolist()}))
    if len(classes) < 2:
        raise ValueError("at least two digit classes are required")
    class_deltas = torch.stack(
        [deltas[digits == value].mean(dim=0) for value in classes]
    )
    _, _, basis = torch.linalg.svd(class_deltas, full_matrices=False)
    return TransportSubspace(
        classes=classes,
        class_deltas=class_deltas,
        basis=basis,
    )


def _conditional_features(
    states: torch.Tensor,
    digits: torch.Tensor,
    *,
    classes: tuple[int, ...],
    state_mean: torch.Tensor,
    state_basis: torch.Tensor,
    score_scale: torch.Tensor,
) -> torch.Tensor:
    class_rows = {value: index for index, value in enumerate(classes)}
    try:
        rows = torch.tensor(
            [class_rows[int(value)] for value in digits.tolist()],
            dtype=torch.long,
        )
    except KeyError as error:
        raise ValueError(f"target digit {error.args[0]} was not fitted") from error
    scores = ((states - state_mean) @ state_basis.T) / score_scale
    one_hot = torch.nn.functional.one_hot(
        rows,
        num_classes=len(classes),
    ).float()
    interactions = (scores[:, :, None] * one_hot[:, None, :]).flatten(start_dim=1)
    bias = torch.ones(states.shape[0], 1)
    return torch.cat((scores, one_hot, interactions, bias), dim=1)


@dataclass(frozen=True)
class ConditionalTransportModel:
    """Reduced-rank transport predicted from recipient state and target digit."""

    classes: tuple[int, ...]
    state_mean: torch.Tensor
    state_basis: torch.Tensor
    score_scale: torch.Tensor
    delta_basis: torch.Tensor
    weights: torch.Tensor

    def predict(self, states: torch.Tensor, target_digits: torch.Tensor) -> torch.Tensor:
        features = _conditional_features(
            states,
            target_digits,
            classes=self.classes,
            state_mean=self.state_mean,
            state_basis=self.state_basis,
            score_scale=self.score_scale,
        )
        return (features @ self.weights) @ self.delta_basis


@dataclass(frozen=True)
class ConditionalTransportDesign:
    """Reusable training design for a conditional reduced-rank bridge."""

    classes: tuple[int, ...]
    state_mean: torch.Tensor
    state_basis: torch.Tensor
    score_scale: torch.Tensor
    delta_basis: torch.Tensor
    features: torch.Tensor
    deltas: torch.Tensor

    def fit(self, *, transport_rank: int, ridge: float) -> ConditionalTransportModel:
        if transport_rank < 1 or transport_rank > self.delta_basis.shape[0]:
            raise ValueError("transport rank is outside the fitted basis")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        basis = self.delta_basis[:transport_rank]
        targets = self.deltas @ basis.T
        gram = self.features.T @ self.features
        penalty = torch.eye(gram.shape[0]) * ridge
        penalty[-1, -1] = 0.0
        weights = torch.linalg.solve(
            gram + penalty,
            self.features.T @ targets,
        )
        return ConditionalTransportModel(
            classes=self.classes,
            state_mean=self.state_mean,
            state_basis=self.state_basis,
            score_scale=self.score_scale,
            delta_basis=basis,
            weights=weights,
        )


def build_conditional_transport_design(
    states: torch.Tensor,
    deltas: torch.Tensor,
    digits: torch.Tensor,
    *,
    state_rank: int,
    max_transport_rank: int,
) -> ConditionalTransportDesign:
    """Prepare state/digit interaction features and transport output basis."""

    if states.ndim != 2 or deltas.shape != states.shape:
        raise ValueError("states and deltas must be aligned matrices")
    if digits.shape != (states.shape[0],):
        raise ValueError("one target digit is required per row")
    if state_rank < 1 or state_rank > min(states.shape):
        raise ValueError("state rank is outside the available state matrix")
    if max_transport_rank < 1 or max_transport_rank > min(deltas.shape):
        raise ValueError("transport rank is outside the available delta matrix")
    classes = tuple(sorted({int(value) for value in digits.tolist()}))
    state_mean = states.mean(dim=0)
    _, _, full_state_basis = torch.linalg.svd(
        states - state_mean,
        full_matrices=False,
    )
    state_basis = full_state_basis[:state_rank]
    raw_scores = (states - state_mean) @ state_basis.T
    score_scale = raw_scores.std(dim=0).clamp_min(1e-6)
    features = _conditional_features(
        states,
        digits,
        classes=classes,
        state_mean=state_mean,
        state_basis=state_basis,
        score_scale=score_scale,
    )
    _, _, full_delta_basis = torch.linalg.svd(deltas, full_matrices=False)
    return ConditionalTransportDesign(
        classes=classes,
        state_mean=state_mean,
        state_basis=state_basis,
        score_scale=score_scale,
        delta_basis=full_delta_basis[:max_transport_rank],
        features=features,
        deltas=deltas,
    )


def encode_three_digit_results(results: torch.Tensor) -> torch.Tensor:
    """Encode a three-digit result as position-specific categorical features."""

    if results.ndim != 1:
        raise ValueError("results must be a vector")
    if bool(((results < 100) | (results > 999)).any()):
        raise ValueError("results must contain three-digit integers")
    hundreds = results // 100
    tens = (results // 10) % 10
    ones = results % 10
    return torch.cat(
        (
            torch.nn.functional.one_hot(hundreds - 1, num_classes=9),
            torch.nn.functional.one_hot(tens, num_classes=10),
            torch.nn.functional.one_hot(ones, num_classes=10),
        ),
        dim=1,
    ).float()


def _full_result_features(
    states: torch.Tensor,
    results: torch.Tensor,
    *,
    state_mean: torch.Tensor,
    state_basis: torch.Tensor,
    score_scale: torch.Tensor,
) -> torch.Tensor:
    scores = ((states - state_mean) @ state_basis.T) / score_scale
    result_features = encode_three_digit_results(results)
    interactions = (
        scores[:, :, None] * result_features[:, None, :]
    ).flatten(start_dim=1)
    bias = torch.ones(states.shape[0], 1)
    return torch.cat((scores, result_features, interactions, bias), dim=1)


@dataclass(frozen=True)
class FullResultTransportModel:
    """Reduced-rank transport conditioned on state and a complete result."""

    state_mean: torch.Tensor
    state_basis: torch.Tensor
    score_scale: torch.Tensor
    delta_basis: torch.Tensor
    weights: torch.Tensor

    def predict(self, states: torch.Tensor, results: torch.Tensor) -> torch.Tensor:
        features = _full_result_features(
            states,
            results,
            state_mean=self.state_mean,
            state_basis=self.state_basis,
            score_scale=self.score_scale,
        )
        return (features @ self.weights) @ self.delta_basis


@dataclass(frozen=True)
class FullResultTransportDesign:
    """Reusable training design for a full-result conditional bridge."""

    state_mean: torch.Tensor
    state_basis: torch.Tensor
    score_scale: torch.Tensor
    delta_basis: torch.Tensor
    features: torch.Tensor
    deltas: torch.Tensor

    def fit(self, *, transport_rank: int, ridge: float) -> FullResultTransportModel:
        if transport_rank < 1 or transport_rank > self.delta_basis.shape[0]:
            raise ValueError("transport rank is outside the fitted basis")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        basis = self.delta_basis[:transport_rank]
        targets = self.deltas @ basis.T
        gram = self.features.T @ self.features
        penalty = torch.eye(gram.shape[0]) * ridge
        penalty[-1, -1] = 0.0
        weights = torch.linalg.solve(
            gram + penalty,
            self.features.T @ targets,
        )
        return FullResultTransportModel(
            state_mean=self.state_mean,
            state_basis=self.state_basis,
            score_scale=self.score_scale,
            delta_basis=basis,
            weights=weights,
        )


def build_full_result_transport_design(
    states: torch.Tensor,
    deltas: torch.Tensor,
    results: torch.Tensor,
    *,
    state_rank: int,
    max_transport_rank: int,
) -> FullResultTransportDesign:
    """Prepare state/full-result interactions and a transport output basis."""

    if states.ndim != 2 or deltas.shape != states.shape:
        raise ValueError("states and deltas must be aligned matrices")
    if results.shape != (states.shape[0],):
        raise ValueError("one target result is required per row")
    encode_three_digit_results(results)
    if state_rank < 1 or state_rank > min(states.shape):
        raise ValueError("state rank is outside the available state matrix")
    if max_transport_rank < 1 or max_transport_rank > min(deltas.shape):
        raise ValueError("transport rank is outside the available delta matrix")
    state_mean = states.mean(dim=0)
    _, _, full_state_basis = torch.linalg.svd(
        states - state_mean,
        full_matrices=False,
    )
    state_basis = full_state_basis[:state_rank]
    raw_scores = (states - state_mean) @ state_basis.T
    score_scale = raw_scores.std(dim=0).clamp_min(1e-6)
    features = _full_result_features(
        states,
        results,
        state_mean=state_mean,
        state_basis=state_basis,
        score_scale=score_scale,
    )
    _, _, full_delta_basis = torch.linalg.svd(deltas, full_matrices=False)
    return FullResultTransportDesign(
        state_mean=state_mean,
        state_basis=state_basis,
        score_scale=score_scale,
        delta_basis=full_delta_basis[:max_transport_rank],
        features=features,
        deltas=deltas,
    )
