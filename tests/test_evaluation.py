import torch

from open_latent_interfaces.evaluation import (
    norm_match,
    random_norm_matched,
    select_bounded_candidate,
    token_metrics,
    wrong_digit_labels,
)


def test_norm_matching_preserves_requested_row_norms() -> None:
    directions = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    target = torch.tensor([2.0, 7.0])
    matched = norm_match(directions, target)
    assert torch.allclose(matched.norm(dim=1), target)


def test_random_norm_matching_is_deterministic() -> None:
    target = torch.tensor([1.0, 2.0, 3.0])
    first = random_norm_matched((3, 5), target, seed=11)
    second = random_norm_matched((3, 5), target, seed=11)
    assert torch.equal(first, second)
    assert torch.allclose(first.norm(dim=1), target)


def test_wrong_digit_mapping_never_preserves_label() -> None:
    labels = torch.arange(1, 10)
    wrong = wrong_digit_labels(labels)
    assert bool((wrong != labels).all())
    assert set(wrong.tolist()) == set(range(1, 10))


def test_token_metrics_reports_count_rank_and_margin() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0], [2.0, 1.0, 4.0]])
    targets = torch.tensor([1, 0])
    metrics = token_metrics(logits, targets)
    assert metrics["top1_count"] == 1
    assert metrics["top1_exact"] == 0.5
    assert metrics["mean_target_rank"] == 1.5


def test_bounded_candidate_selection_excludes_excess_norm() -> None:
    rows = [
        {
            "scale": 1.0,
            "minimum_accuracy": 0.4,
            "target_token_accuracy": 0.5,
            "identity_token_accuracy": 0.4,
            "mean_target_relative_norm": 0.4,
            "mean_identity_relative_norm": 0.3,
        },
        {
            "scale": 2.0,
            "minimum_accuracy": 0.8,
            "target_token_accuracy": 0.8,
            "identity_token_accuracy": 0.9,
            "mean_target_relative_norm": 1.1,
            "mean_identity_relative_norm": 0.6,
        },
    ]
    selected = select_bounded_candidate(rows, max_relative_norm=1.0)
    assert selected["scale"] == 1.0
