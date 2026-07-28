from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import open_latent_interfaces.compiler_writer as compiler_writer
from open_latent_interfaces.compiler_writer import (
    PositionCompilerSpec,
    sequential_compiler_condition,
)


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return "".join(str(token_id - 10) for token_id in token_ids)


class FakePlan:
    def __init__(self, target_token_ids: torch.Tensor) -> None:
        self.target_token_ids = target_token_ids
        self.recipient_states = torch.ones((len(target_token_ids), 2))

    def deltas(
        self,
        *,
        desired_margin: float,
        max_relative_norm: float,
    ) -> torch.Tensor:
        del desired_margin, max_relative_norm
        return torch.stack(
            (
                self.target_token_ids.float(),
                torch.zeros_like(self.target_token_ids, dtype=torch.float32),
            ),
            dim=1,
        )


def test_position_compiler_spec_validation() -> None:
    assert PositionCompilerSpec(24, 8.0, 0.25).hidden_state_index == 24
    with pytest.raises(ValueError, match="block output"):
        PositionCompilerSpec(0, 8.0, 0.25)
    with pytest.raises(ValueError, match="nonnegative"):
        PositionCompilerSpec(24, -1.0, 0.25)
    with pytest.raises(ValueError, match="positive"):
        PositionCompilerSpec(24, 8.0, 0.0)


def test_sequential_target_compiler_emits_three_requested_digits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_compile(
        _model: object,
        _tokenizer: object,
        _prompts: list[str],
        *,
        target_token_ids: torch.Tensor,
        **_kwargs: object,
    ) -> FakePlan:
        return FakePlan(target_token_ids)

    def fake_predict(
        _model: object,
        _tokenizer: object,
        _prompts: list[str],
        delta: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        logits = torch.zeros((len(delta), 20))
        rows = torch.arange(len(delta))
        logits[rows, delta[:, 0].long()] = 1
        return logits

    monkeypatch.setattr(
        compiler_writer,
        "compile_local_margin_plan",
        fake_compile,
    )
    monkeypatch.setattr(
        compiler_writer,
        "_predict_with_delta",
        fake_predict,
    )
    targets = [123, 456]
    result = sequential_compiler_condition(
        "target",
        model=object(),
        tokenizer=FakeTokenizer(),
        capture=SimpleNamespace(),
        example_ids=["a", "b"],
        original_results=targets,
        rendered_prompts=["p=", "q="],
        writer_targets=targets,
        evaluation_targets=targets,
        true_targets=targets,
        reference_targets=None,
        digit_token_ids={digit: 10 + digit for digit in range(10)},
        candidate_token_ids=torch.arange(10, 20),
        position_specs={
            position: PositionCompilerSpec(24, 8.0, 0.25)
            for position in range(3)
        },
        plan_cache={},
        compiler_batch_size=2,
        base_model_batch_size=2,
        random_seed=7,
        device=torch.device("cpu"),
    )
    assert result["evaluation_target_correct"] == 2
    assert result["writer_target_correct"] == 2
    assert result["true_result_correct"] == 2
    assert result["step_evaluation_target_correct"] == [2, 2, 2]
    assert result["digit_token_rate"] == 1.0
    assert [row["parsed"] for row in result["outputs"]] == targets
