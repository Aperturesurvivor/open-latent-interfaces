from __future__ import annotations

import pytest
import torch

from open_latent_interfaces.causal_compiler import LocalMarginPlan


def make_plan(*, hard_gate: bool = False) -> LocalMarginPlan:
    return LocalMarginPlan(
        recipient_states=torch.tensor([[3.0, 4.0]]),
        base_logits=torch.zeros((1, 8)),
        target_token_ids=torch.tensor([2]),
        competitor_token_ids=torch.tensor([3]),
        current_margins=torch.tensor([-1.0]),
        margin_gradients=torch.tensor([[3.0, 4.0]]),
        hard_gate=torch.tensor([hard_gate]),
    )


def test_minimum_norm_delta_reaches_linearized_margin() -> None:
    plan = make_plan()
    delta = plan.deltas(desired_margin=2.0, max_relative_norm=None)
    achieved_change = (delta * plan.margin_gradients).sum(dim=1)
    assert torch.allclose(achieved_change, torch.tensor([3.0]))
    assert torch.allclose(delta, torch.tensor([[0.36, 0.48]]))


def test_hard_gate_is_exact_zero() -> None:
    delta = make_plan(hard_gate=True).deltas(desired_margin=20.0)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_relative_norm_cap_is_enforced() -> None:
    delta = make_plan().deltas(
        desired_margin=100.0,
        max_relative_norm=0.1,
    )
    assert delta.norm().item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"desired_margin": -1.0}, "nonnegative"),
        ({"desired_margin": 1.0, "strength": 0.0}, "positive"),
        (
            {"desired_margin": 1.0, "max_relative_norm": 0.0},
            "positive or None",
        ),
    ],
)
def test_invalid_delta_parameters(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_plan().deltas(**kwargs)
