from __future__ import annotations

from typing import Any

import torch


def first_digit_token_id(tokenizer: Any, value: int) -> int | None:
    expected = str(value)[0]
    token_ids = tokenizer.encode(expected, add_special_tokens=False)
    if len(token_ids) != 1 or tokenizer.decode(token_ids[0]) != expected:
        return None
    return int(token_ids[0])


def token_metrics(logits: torch.Tensor, target_ids: torch.Tensor) -> dict[str, float]:
    rows = torch.arange(logits.shape[0])
    target_logits = logits[rows, target_ids]
    ranks = 1 + (logits > target_logits[:, None]).sum(dim=1)
    other = logits.clone()
    other[rows, target_ids] = -torch.inf
    margins = target_logits - other.max(dim=1).values
    return {
        "top1_exact": float((logits.argmax(dim=1) == target_ids).float().mean()),
        "top1_count": int((logits.argmax(dim=1) == target_ids).sum()),
        "mean_target_logit": float(target_logits.mean()),
        "mean_target_rank": float(ranks.float().mean()),
        "median_target_rank": float(ranks.float().median()),
        "mean_target_margin": float(margins.mean()),
    }


def margin_delta(
    before: torch.Tensor,
    after: torch.Tensor,
    target_ids: torch.Tensor,
) -> float:
    return token_metrics(after, target_ids)["mean_target_margin"] - token_metrics(
        before,
        target_ids,
    )["mean_target_margin"]


def wrong_digit_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map each leading digit 1–9 to a deterministic different digit."""

    labels = labels.long()
    return labels.remainder(9) + 1


def norm_match(directions: torch.Tensor, target_norms: torch.Tensor) -> torch.Tensor:
    directions = directions.float()
    target_norms = target_norms.float().reshape(-1)
    if directions.ndim != 2 or directions.shape[0] != target_norms.shape[0]:
        raise ValueError("directions and target norms must align by row")
    norms = directions.norm(dim=1).clamp_min(1e-12)
    return directions * (target_norms / norms)[:, None]


def random_norm_matched(
    shape: tuple[int, int],
    target_norms: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(shape, generator=generator)
    return norm_match(directions, target_norms)


def select_bounded_candidate(
    rows: list[dict[str, Any]],
    *,
    max_relative_norm: float,
) -> dict[str, Any]:
    """Select the strongest accuracy row that satisfies both norm ceilings."""
    eligible = [
        row
        for row in rows
        if row["mean_target_relative_norm"] <= max_relative_norm
        and row["mean_identity_relative_norm"] <= max_relative_norm
    ]
    if not eligible:
        raise ValueError("no candidate satisfies the relative-norm ceiling")
    return max(
        eligible,
        key=lambda row: (
            row["minimum_accuracy"],
            row["target_token_accuracy"],
            row["identity_token_accuracy"],
            -row["mean_target_relative_norm"],
            -row["mean_identity_relative_norm"],
        ),
    )


def phase2_advancement_gate_passes(report: dict[str, Any]) -> bool:
    """Check every conjunctive Phase 2 closed-loop advancement criterion."""
    gate = report["advancement_gate"]
    conditions = report["conditions"]
    targeted = conditions["hybrid"]
    identity = conditions["identity_hard_gated"]
    controls = [
        row["target_full_result_accuracy"]
        for name, row in conditions.items()
        if name not in ("hybrid", "identity_hard_gated")
    ]
    return (
        targeted["target_full_result_accuracy"]
        >= gate["exact_target_minimum"]
        and min(targeted["step_target_token_accuracy"])
        >= gate["per_position_target_minimum"]
        and targeted["target_full_result_accuracy"] - max(controls)
        >= gate["control_advantage_minimum"]
        and identity["original_full_result_accuracy"]
        >= gate["identity_preservation_minimum"]
        and max(targeted["mean_relative_norm_by_step"])
        <= gate["relative_norm_maximum"]
        and targeted["parse_rate"] == gate["parse_rate_required"]
    )
