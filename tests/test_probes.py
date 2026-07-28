import torch

from open_latent_interfaces.probes import (
    BinaryRidgeProbe,
    CategoricalRidgeProbe,
    ScalarRidgeProbe,
    binary_metrics,
    regression_metrics,
)


def test_scalar_probe_recovers_synthetic_signal() -> None:
    generator = torch.Generator().manual_seed(4)
    values = torch.randn(80, 12, generator=generator)
    targets = 7 * values[:, 2] - 3 * values[:, 7] + 11
    probe = ScalarRidgeProbe.fit(values[:60], targets[:60], l2=0.01)
    metrics = regression_metrics(probe.predict(values[60:]), targets[60:])
    assert metrics["r2"] > 0.999
    assert metrics["mae"] < 0.02


def test_minimal_shift_reaches_probe_target_without_clipping() -> None:
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(64, 10, generator=generator)
    targets = 4 * values[:, 1] + 2 * values[:, 4]
    probe = ScalarRidgeProbe.fit(values, targets, l2=0.001)
    source = values[:8]
    desired = probe.predict(source) + 3.0
    delta = probe.minimal_shift(
        source,
        desired,
        max_relative_norm=None,
    )
    moved = probe.predict(source + delta)
    assert torch.allclose(moved, desired, atol=1e-3)


def test_binary_probe_separates_synthetic_classes() -> None:
    generator = torch.Generator().manual_seed(6)
    values = torch.randn(100, 8, generator=generator)
    labels = (values[:, 3] - values[:, 5] > 0).long()
    probe = BinaryRidgeProbe.fit(values[:80], labels[:80], l2=0.1)
    metrics = binary_metrics(probe.score(values[80:]), labels[80:])
    assert metrics["accuracy"] > 0.85
    assert metrics["auc"] > 0.9


def test_binary_probe_calibrates_imbalanced_training_threshold() -> None:
    values = torch.tensor(
        [[-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]]
    )
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    probe = BinaryRidgeProbe.fit(values, labels, l2=0.1)
    predictions = probe.predict(values)
    assert torch.equal(predictions, labels)


def test_categorical_probe_shift_reaches_requested_margin() -> None:
    generator = torch.Generator().manual_seed(7)
    labels = torch.arange(120) % 3
    prototypes = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    values = prototypes[labels] + 0.05 * torch.randn(120, 4, generator=generator)
    probe = CategoricalRidgeProbe.fit(
        values,
        labels,
        number_of_classes=3,
        l2=0.01,
    )
    source = values[:9]
    desired = (labels[:9] + 1) % 3
    deltas = probe.minimal_margin_shift(
        source,
        desired,
        margin=0.5,
        max_relative_norm=None,
    )
    scores = probe.score(source + deltas)
    rows = torch.arange(len(source))
    competitors = scores.clone()
    competitors[rows, desired] = -torch.inf
    achieved = scores[rows, desired] - competitors.max(dim=1).values
    assert bool((achieved >= 0.499).all())
