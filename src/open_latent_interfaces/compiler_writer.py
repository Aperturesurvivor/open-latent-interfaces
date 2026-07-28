from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from open_latent_interfaces.activations import ActivationCapture
from open_latent_interfaces.capability import parse_first_integer
from open_latent_interfaces.causal_compiler import (
    LocalMarginPlan,
    compile_local_margin_plan,
)
from open_latent_interfaces.evaluation import norm_match, random_norm_matched
from open_latent_interfaces.interventions import intervened_next_token_logits

CompilerCondition = Literal["base", "target", "wrong", "random"]


@dataclass(frozen=True)
class PositionCompilerSpec:
    hidden_state_index: int
    desired_margin: float
    norm_cap: float

    def __post_init__(self) -> None:
        if self.hidden_state_index < 1:
            raise ValueError("hidden-state index must be a block output")
        if self.desired_margin < 0:
            raise ValueError("desired margin must be nonnegative")
        if self.norm_cap <= 0:
            raise ValueError("norm cap must be positive")


def _predict_with_delta(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    delta: torch.Tensor,
    *,
    hidden_state_index: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(prompts), batch_size):
        chunks.append(
            intervened_next_token_logits(
                model,
                tokenizer,
                prompts[start : start + batch_size],
                hidden_state_index=hidden_state_index,
                deltas=delta[start : start + batch_size],
                device=device,
            )
        )
    return torch.cat(chunks)


def sequential_compiler_condition(
    condition: CompilerCondition,
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    *,
    example_ids: list[str],
    original_results: list[int],
    rendered_prompts: list[str],
    writer_targets: list[int],
    evaluation_targets: list[int],
    true_targets: list[int],
    reference_targets: list[int] | None,
    digit_token_ids: dict[int, int],
    candidate_token_ids: torch.Tensor,
    position_specs: dict[int, PositionCompilerSpec],
    plan_cache: dict[
        tuple[tuple[str, ...], tuple[int, ...], int],
        LocalMarginPlan,
    ],
    compiler_batch_size: int,
    base_model_batch_size: int,
    random_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Generate exactly three tokens under a sequential compiler condition."""

    n = len(rendered_prompts)
    aligned = (
        example_ids,
        original_results,
        writer_targets,
        evaluation_targets,
        true_targets,
    )
    if any(len(values) != n for values in aligned):
        raise ValueError("condition inputs must align")
    if condition == "wrong" and reference_targets is None:
        raise ValueError("wrong-target control requires reference targets")
    if reference_targets is not None and len(reference_targets) != n:
        raise ValueError("reference targets must align")
    if sorted(position_specs) != [0, 1, 2]:
        raise ValueError("exactly three position compiler specs are required")

    prefixes = ["" for _ in range(n)]
    predicted_ids: list[list[int]] = [[] for _ in range(n)]
    evaluation_step_correct = []
    writer_step_correct = []
    true_step_correct = []
    relative_norms = []
    gate_rates = []
    cache_hits = 0
    cache_misses = 0

    def plan(
        prompts: list[str],
        target_ids: torch.Tensor,
        position: int,
    ) -> LocalMarginPlan:
        nonlocal cache_hits, cache_misses
        key = (tuple(prompts), tuple(target_ids.tolist()), position)
        if key in plan_cache:
            cache_hits += 1
            return plan_cache[key]
        spec = position_specs[position]
        compiled = compile_local_margin_plan(
            model,
            tokenizer,
            prompts,
            hidden_state_index=spec.hidden_state_index,
            target_token_ids=target_ids,
            candidate_token_ids=candidate_token_ids,
            device=device,
            batch_size=compiler_batch_size,
        )
        plan_cache[key] = compiled
        cache_misses += 1
        return compiled

    for position in range(3):
        spec = position_specs[position]
        prompts = [
            prompt + prefix
            for prompt, prefix in zip(
                rendered_prompts,
                prefixes,
                strict=True,
            )
        ]
        writer_expected = torch.tensor(
            [
                digit_token_ids[int(str(value)[position])]
                for value in writer_targets
            ],
            dtype=torch.long,
        )
        evaluation_expected = [
            digit_token_ids[int(str(value)[position])]
            for value in evaluation_targets
        ]
        true_expected = [
            digit_token_ids[int(str(value)[position])]
            for value in true_targets
        ]
        if condition == "base":
            logits = capture.next_token_logits(
                prompts,
                batch_size=base_model_batch_size,
            )
            relative_norms.append(0.0)
            gate_rates.append(1.0)
        else:
            target_plan = plan(prompts, writer_expected, position)
            targeted = target_plan.deltas(
                desired_margin=spec.desired_margin,
                max_relative_norm=spec.norm_cap,
            )
            if condition == "target":
                delta = targeted
                reference_states = target_plan.recipient_states
            elif condition == "random":
                delta = random_norm_matched(
                    tuple(targeted.shape),
                    targeted.norm(dim=1),
                    seed=random_seed + position,
                )
                reference_states = target_plan.recipient_states
            else:
                assert reference_targets is not None
                reference_expected = torch.tensor(
                    [
                        digit_token_ids[int(str(value)[position])]
                        for value in reference_targets
                    ],
                    dtype=torch.long,
                )
                reference_plan = plan(
                    prompts,
                    reference_expected,
                    position,
                )
                reference_delta = reference_plan.deltas(
                    desired_margin=spec.desired_margin,
                    max_relative_norm=spec.norm_cap,
                )
                delta = norm_match(
                    targeted,
                    reference_delta.norm(dim=1),
                )
                reference_states = reference_plan.recipient_states
            logits = _predict_with_delta(
                model,
                tokenizer,
                prompts,
                delta,
                hidden_state_index=spec.hidden_state_index,
                batch_size=base_model_batch_size,
                device=device,
            )
            relative_norms.append(
                float(
                    (
                        delta.norm(dim=1)
                        / reference_states.norm(dim=1).clamp_min(1e-12)
                    ).mean()
                )
            )
            gate_rates.append(
                float((delta.norm(dim=1) == 0).float().mean())
            )

        next_ids = logits.argmax(dim=1).tolist()
        evaluation_step_correct.append(
            sum(
                actual == expected
                for actual, expected in zip(
                    next_ids,
                    evaluation_expected,
                    strict=True,
                )
            )
        )
        writer_step_correct.append(
            sum(
                actual == expected
                for actual, expected in zip(
                    next_ids,
                    writer_expected.tolist(),
                    strict=True,
                )
            )
        )
        true_step_correct.append(
            sum(
                actual == expected
                for actual, expected in zip(
                    next_ids,
                    true_expected,
                    strict=True,
                )
            )
        )
        for index, token_id in enumerate(next_ids):
            predicted_ids[index].append(int(token_id))
            prefixes[index] += tokenizer.decode([int(token_id)])

    text = [tokenizer.decode(token_ids) for token_ids in predicted_ids]
    parsed = [parse_first_integer(value) for value in text]
    digit_ids = set(digit_token_ids.values())
    outputs = []
    for index in range(n):
        outputs.append(
            {
                "example_id": example_ids[index],
                "original_result": original_results[index],
                "writer_target": writer_targets[index],
                "evaluation_target": evaluation_targets[index],
                "true_target": true_targets[index],
                "predicted_token_ids": predicted_ids[index],
                "text": text[index],
                "parsed": parsed[index],
            }
        )
    evaluation_correct = sum(
        actual == target
        for actual, target in zip(
            parsed,
            evaluation_targets,
            strict=True,
        )
    )
    writer_correct = sum(
        actual == target
        for actual, target in zip(
            parsed,
            writer_targets,
            strict=True,
        )
    )
    true_correct = sum(
        actual == target
        for actual, target in zip(parsed, true_targets, strict=True)
    )
    return {
        "n": n,
        "evaluation_target_correct": evaluation_correct,
        "evaluation_target_accuracy": evaluation_correct / n,
        "writer_target_correct": writer_correct,
        "writer_target_accuracy": writer_correct / n,
        "true_result_correct": true_correct,
        "true_result_accuracy": true_correct / n,
        "step_evaluation_target_correct": evaluation_step_correct,
        "step_evaluation_target_accuracy": [
            value / n for value in evaluation_step_correct
        ],
        "step_writer_target_correct": writer_step_correct,
        "step_writer_target_accuracy": [
            value / n for value in writer_step_correct
        ],
        "step_true_target_correct": true_step_correct,
        "step_true_target_accuracy": [
            value / n for value in true_step_correct
        ],
        "parse_rate": sum(value is not None for value in parsed) / n,
        "digit_token_rate": (
            sum(
                token_id in digit_ids
                for row in predicted_ids
                for token_id in row
            )
            / (3 * n)
        ),
        "mean_relative_norm_by_position": relative_norms,
        "hard_gate_rate_by_position": gate_rates,
        "plan_cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "entries_after_condition": len(plan_cache),
        },
        "outputs": outputs,
    }
