from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> Standardizer:
        if values.ndim != 2:
            raise ValueError("values must have shape [examples, features]")
        mean = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)
        return cls(mean=mean, scale=scale)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale

    def inverse_direction(self, standardized_direction: torch.Tensor) -> torch.Tensor:
        return standardized_direction / self.scale


def _ridge_weights(
    values: torch.Tensor,
    targets: torch.Tensor,
    *,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor, Standardizer]:
    if values.ndim != 2:
        raise ValueError("values must have shape [examples, features]")
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2 or targets.shape[0] != values.shape[0]:
        raise ValueError("targets must align with values")
    if l2 <= 0:
        raise ValueError("l2 must be positive")

    standardizer = Standardizer.fit(values)
    x = standardizer.transform(values)
    y_mean = targets.mean(dim=0)
    centered_y = targets - y_mean

    # Solve in sample space when width is larger than the number of examples.
    # This is both faster and numerically more stable for residual streams.
    if x.shape[1] > x.shape[0]:
        gram = x @ x.T
        dual = torch.linalg.solve(
            gram + l2 * torch.eye(gram.shape[0], dtype=x.dtype, device=x.device),
            centered_y,
        )
        weights = x.T @ dual
    else:
        gram = x.T @ x
        weights = torch.linalg.solve(
            gram + l2 * torch.eye(gram.shape[0], dtype=x.dtype, device=x.device),
            x.T @ centered_y,
        )
    return weights, y_mean, standardizer


@dataclass(frozen=True)
class ScalarRidgeProbe:
    weights: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    standardizer: Standardizer

    @classmethod
    def fit(
        cls,
        values: torch.Tensor,
        targets: torch.Tensor,
        *,
        l2: float = 10.0,
    ) -> ScalarRidgeProbe:
        targets = targets.float().reshape(-1)
        target_mean = targets.mean()
        target_scale = targets.std(unbiased=False).clamp_min(1e-6)
        normalized = (targets - target_mean) / target_scale
        weights, bias, standardizer = _ridge_weights(values.float(), normalized, l2=l2)
        return cls(
            weights=weights[:, 0],
            target_mean=target_mean,
            target_scale=target_scale,
            standardizer=standardizer,
        )

    def predict_normalized(self, values: torch.Tensor) -> torch.Tensor:
        x = self.standardizer.transform(values.float())
        return x @ self.weights

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        return self.predict_normalized(values) * self.target_scale + self.target_mean

    def raw_direction(self) -> torch.Tensor:
        """Return the activation-space covector used by the scalar decoder."""

        return self.standardizer.inverse_direction(self.weights)

    def minimal_shift(
        self,
        values: torch.Tensor,
        desired_targets: torch.Tensor,
        *,
        strength: float = 1.0,
        max_relative_norm: float | None = 0.25,
    ) -> torch.Tensor:
        """Return the minimum-L2 raw activation shift under the linear probe.

        The shift is diagnostic, not proof that the decoded scalar is a
        monosemantic causal variable. ``max_relative_norm`` prevents extreme
        off-manifold interventions.
        """

        values = values.float()
        desired = desired_targets.float().reshape(-1)
        current = self.predict(values)
        raw_direction = self.raw_direction()
        denominator = raw_direction.square().sum().clamp_min(1e-12)
        deltas = (
            strength
            * ((desired - current) / self.target_scale)[:, None]
            * raw_direction[None, :]
            / denominator
        )
        if max_relative_norm is not None:
            if max_relative_norm <= 0:
                raise ValueError("max_relative_norm must be positive or None")
            maximum = max_relative_norm * values.norm(dim=1).clamp_min(1e-12)
            norms = deltas.norm(dim=1).clamp_min(1e-12)
            scales = torch.minimum(torch.ones_like(norms), maximum / norms)
            deltas = deltas * scales[:, None]
        return deltas


@dataclass(frozen=True)
class BinaryRidgeProbe:
    weights: torch.Tensor
    bias: torch.Tensor
    threshold: torch.Tensor
    standardizer: Standardizer

    @classmethod
    def fit(
        cls,
        values: torch.Tensor,
        labels: torch.Tensor,
        *,
        l2: float = 10.0,
    ) -> BinaryRidgeProbe:
        signed = labels.float().reshape(-1) * 2 - 1
        weights, bias, standardizer = _ridge_weights(values.float(), signed, l2=l2)
        raw_scores = standardizer.transform(values.float()) @ weights[:, 0] + bias[0]
        sorted_scores = torch.sort(torch.unique(raw_scores)).values
        if len(sorted_scores) == 1:
            candidates = sorted_scores
        else:
            candidates = torch.cat(
                (
                    sorted_scores[:1] - 1e-6,
                    (sorted_scores[:-1] + sorted_scores[1:]) / 2,
                    sorted_scores[-1:] + 1e-6,
                )
            )
        positives = labels.reshape(-1) == 1
        negatives = ~positives
        best_score: tuple[float, float, float] | None = None
        best_threshold: torch.Tensor | None = None
        for candidate in candidates:
            predictions = raw_scores >= candidate
            true_positive_rate = (predictions[positives]).float().mean()
            true_negative_rate = (~predictions[negatives]).float().mean()
            balanced = float((true_positive_rate + true_negative_rate) / 2)
            accuracy = float((predictions == positives).float().mean())
            score = (balanced, accuracy, -float(candidate.abs()))
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = candidate
        assert best_threshold is not None
        return cls(
            weights=weights[:, 0],
            bias=bias[0],
            threshold=best_threshold,
            standardizer=standardizer,
        )

    def score(self, values: torch.Tensor) -> torch.Tensor:
        raw = self.standardizer.transform(values.float()) @ self.weights + self.bias
        return raw - self.threshold

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        return (self.score(values) >= 0).long()


@dataclass(frozen=True)
class CategoricalRidgeProbe:
    weights: torch.Tensor
    bias: torch.Tensor
    standardizer: Standardizer
    number_of_classes: int

    @classmethod
    def fit(
        cls,
        values: torch.Tensor,
        labels: torch.Tensor,
        *,
        number_of_classes: int,
        l2: float = 10.0,
    ) -> CategoricalRidgeProbe:
        labels = labels.long().reshape(-1)
        if number_of_classes < 2:
            raise ValueError("number_of_classes must be at least two")
        if bool(((labels < 0) | (labels >= number_of_classes)).any()):
            raise ValueError("categorical label outside configured classes")
        targets = torch.nn.functional.one_hot(
            labels,
            num_classes=number_of_classes,
        ).float()
        weights, bias, standardizer = _ridge_weights(values.float(), targets, l2=l2)
        return cls(
            weights=weights,
            bias=bias,
            standardizer=standardizer,
            number_of_classes=number_of_classes,
        )

    def score(self, values: torch.Tensor) -> torch.Tensor:
        return self.standardizer.transform(values.float()) @ self.weights + self.bias

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        return self.score(values).argmax(dim=1)

    def raw_weights(self) -> torch.Tensor:
        return self.weights / self.standardizer.scale[:, None]

    def minimal_margin_shift(
        self,
        values: torch.Tensor,
        desired_labels: torch.Tensor,
        *,
        margin: float = 1.0,
        strength: float = 1.0,
        max_relative_norm: float | None = 0.25,
    ) -> torch.Tensor:
        """Move each row toward the desired class against its best competitor."""

        values = values.float()
        desired_labels = desired_labels.long().reshape(-1)
        if desired_labels.shape[0] != values.shape[0]:
            raise ValueError("one desired label is required per activation")
        scores = self.score(values)
        rows = torch.arange(values.shape[0])
        competitors = scores.clone()
        competitors[rows, desired_labels] = -torch.inf
        competitor_labels = competitors.argmax(dim=1)
        current_margin = (
            scores[rows, desired_labels] - scores[rows, competitor_labels]
        )
        required = (margin - current_margin).clamp_min(0) * strength
        raw = self.raw_weights()
        directions = raw[:, desired_labels].T - raw[:, competitor_labels].T
        denominator = directions.square().sum(dim=1).clamp_min(1e-12)
        deltas = required[:, None] * directions / denominator[:, None]
        if max_relative_norm is not None:
            if max_relative_norm <= 0:
                raise ValueError("max_relative_norm must be positive or None")
            maximum = max_relative_norm * values.norm(dim=1).clamp_min(1e-12)
            norms = deltas.norm(dim=1).clamp_min(1e-12)
            scales = torch.minimum(torch.ones_like(norms), maximum / norms)
            deltas = deltas * scales[:, None]
        return deltas


def regression_metrics(predicted: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predicted = predicted.float().reshape(-1)
    targets = targets.float().reshape(-1)
    residual = predicted - targets
    denominator = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - residual.square().sum() / denominator.clamp_min(1e-12)
    return {
        "r2": float(r2),
        "mae": float(residual.abs().mean()),
        "rounded_exact": float((predicted.round() == targets).float().mean()),
    }


def binary_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    scores = scores.float().reshape(-1)
    labels = labels.long().reshape(-1)
    predictions = (scores >= 0).long()
    positives = labels == 1
    negatives = labels == 0
    true_positive_rate = ((predictions == 1) & positives).sum() / positives.sum().clamp_min(1)
    true_negative_rate = ((predictions == 0) & negatives).sum() / negatives.sum().clamp_min(1)

    # Rank-based AUC with average ranks for ties.
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    positive_count = positives.sum()
    negative_count = negatives.sum()
    auc = (
        ranks[positives].sum() - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count).clamp_min(1)
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "balanced_accuracy": float((true_positive_rate + true_negative_rate) / 2),
        "auc": float(auc),
    }


def categorical_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    number_of_classes: int,
) -> dict[str, object]:
    labels = labels.long().reshape(-1)
    predictions = scores.argmax(dim=1)
    per_class: dict[str, float | None] = {}
    recalls: list[torch.Tensor] = []
    for label in range(number_of_classes):
        selected = labels == label
        if not bool(selected.any()):
            per_class[str(label)] = None
            continue
        recall = (predictions[selected] == label).float().mean()
        per_class[str(label)] = float(recall)
        recalls.append(recall)
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "macro_recall_present_classes": float(torch.stack(recalls).mean()) if recalls else 0.0,
        "per_class_recall": per_class,
    }
