from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from open_latent_interfaces.activations import (
    last_nonpadding_positions,
    resolve_decoder_blocks,
)


def _replace_hidden(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    raise TypeError(f"unsupported decoder block output: {type(output)!r}")


@dataclass(frozen=True)
class LocalMarginPlan:
    """A frozen-model, prompt-local linearization of a token margin."""

    recipient_states: torch.Tensor
    base_logits: torch.Tensor
    target_token_ids: torch.Tensor
    competitor_token_ids: torch.Tensor
    current_margins: torch.Tensor
    margin_gradients: torch.Tensor
    hard_gate: torch.Tensor

    def deltas(
        self,
        *,
        desired_margin: float,
        strength: float = 1.0,
        max_relative_norm: float | None = 1.0,
    ) -> torch.Tensor:
        """Return the minimum-L2 first-order update for the requested margin."""

        if desired_margin < 0:
            raise ValueError("desired_margin must be nonnegative")
        if strength <= 0:
            raise ValueError("strength must be positive")
        required = (
            (desired_margin - self.current_margins).clamp_min(0) * strength
        )
        required = torch.where(
            self.hard_gate,
            torch.zeros_like(required),
            required,
        )
        denominator = self.margin_gradients.square().sum(dim=1).clamp_min(1e-12)
        deltas = (
            required[:, None] * self.margin_gradients / denominator[:, None]
        )
        if max_relative_norm is not None:
            if max_relative_norm <= 0:
                raise ValueError("max_relative_norm must be positive or None")
            maximum = (
                max_relative_norm
                * self.recipient_states.norm(dim=1).clamp_min(1e-12)
            )
            norms = deltas.norm(dim=1).clamp_min(1e-12)
            factors = torch.minimum(torch.ones_like(norms), maximum / norms)
            deltas = deltas * factors[:, None]
        return deltas


@dataclass(frozen=True)
class IterativeMarginTrace:
    """Cumulative residual updates after each prompt-local relinearization."""

    cumulative_deltas: tuple[torch.Tensor, ...]
    base_recipient_states: torch.Tensor


def project_relative_norm(
    deltas: torch.Tensor,
    reference_states: torch.Tensor,
    *,
    max_relative_norm: float,
) -> torch.Tensor:
    """Project row-wise updates onto a relative-L2 ball."""

    if max_relative_norm <= 0:
        raise ValueError("max_relative_norm must be positive")
    if deltas.shape != reference_states.shape or deltas.ndim != 2:
        raise ValueError("deltas and reference_states must share a 2D shape")
    maximum = max_relative_norm * reference_states.norm(dim=1).clamp_min(1e-12)
    norms = deltas.norm(dim=1).clamp_min(1e-12)
    factors = torch.minimum(torch.ones_like(norms), maximum / norms)
    return deltas * factors[:, None]


def compile_local_margin_plan(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    hidden_state_index: int,
    target_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    device: torch.device | str,
    batch_size: int = 4,
    preexisting_deltas: torch.Tensor | None = None,
) -> LocalMarginPlan:
    """Differentiate a requested next-token margin at one residual boundary.

    The returned plan contains no trained parameters. Each gradient is taken
    through the frozen suffix of the model, from ``hidden_state_index`` to the
    next-token logits, at the supplied prompt.
    """

    if hidden_state_index < 1:
        raise ValueError("hidden_state_index must be at least one")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(prompts) != int(target_token_ids.numel()):
        raise ValueError("one target token id is required per prompt")
    if preexisting_deltas is not None:
        if preexisting_deltas.ndim != 2:
            raise ValueError("preexisting_deltas must have shape [prompts, width]")
        if preexisting_deltas.shape[0] != len(prompts):
            raise ValueError("one preexisting delta is required per prompt")
    if candidate_token_ids.ndim != 1 or candidate_token_ids.numel() < 2:
        raise ValueError("candidate_token_ids must contain at least two ids")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("model parameters must be frozen before compilation")

    device = torch.device(device)
    targets = target_token_ids.long().reshape(-1)
    candidates = candidate_token_ids.long().reshape(-1)
    if not bool((targets[:, None] == candidates[None, :]).any(dim=1).all()):
        raise ValueError("every target token id must be a candidate")
    blocks = resolve_decoder_blocks(model)
    block_index = hidden_state_index - 1
    if block_index >= len(blocks):
        raise IndexError(
            f"block index {block_index} outside model with {len(blocks)} blocks"
        )

    state_rows = []
    logit_rows = []
    competitor_rows = []
    margin_rows = []
    gradient_rows = []
    gate_rows = []
    for start in range(0, len(prompts), batch_size):
        stop = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:stop]
        batch_targets = targets[start:stop].to(device)
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        positions = last_nonpadding_positions(encoded["attention_mask"])
        rows = torch.arange(stop - start, device=device)
        captured: dict[str, torch.Tensor] = {}
        batch_preexisting = (
            None
            if preexisting_deltas is None
            else preexisting_deltas[start:stop]
        )

        def capture_leaf(
            _module: Any,
            _inputs: tuple[Any, ...],
            output: Any,
            captured_store: dict[str, torch.Tensor] = captured,
            row_indices: torch.Tensor = rows,
            token_positions: torch.Tensor = positions,
            existing: torch.Tensor | None = batch_preexisting,
        ) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            leaf = hidden.detach().clone()
            if existing is not None:
                leaf[row_indices, token_positions] += existing.to(
                    device=leaf.device,
                    dtype=leaf.dtype,
                )
            leaf.requires_grad_(True)
            captured_store["hidden"] = leaf
            return _replace_hidden(output, leaf)

        handle = blocks[block_index].register_forward_hook(capture_leaf)
        try:
            with torch.enable_grad():
                outputs = model(
                    **encoded,
                    use_cache=False,
                    return_dict=True,
                )
                logits = outputs.logits[rows, positions]
                digit_logits = logits[:, candidates.to(device)]
                target_columns = (
                    batch_targets[:, None] == candidates.to(device)[None, :]
                ).long().argmax(dim=1)
                alternatives = digit_logits.clone()
                alternatives[rows, target_columns] = -torch.inf
                competitor_columns = alternatives.argmax(dim=1)
                competitor_ids = candidates.to(device)[competitor_columns]
                margins = (
                    logits[rows, batch_targets]
                    - logits[rows, competitor_ids]
                )
                gradient = torch.autograd.grad(
                    margins.sum(),
                    captured["hidden"],
                    retain_graph=False,
                    create_graph=False,
                )[0][rows, positions]
        finally:
            handle.remove()

        states = captured["hidden"][rows, positions]
        state_rows.append(states.detach().float().cpu())
        logit_rows.append(logits.detach().float().cpu())
        competitor_rows.append(competitor_ids.detach().long().cpu())
        margin_rows.append(margins.detach().float().cpu())
        gradient_rows.append(gradient.detach().float().cpu())
        gate_rows.append(
            (logits.detach().argmax(dim=1) == batch_targets).cpu()
        )

    return LocalMarginPlan(
        recipient_states=torch.cat(state_rows),
        base_logits=torch.cat(logit_rows),
        target_token_ids=targets.cpu(),
        competitor_token_ids=torch.cat(competitor_rows),
        current_margins=torch.cat(margin_rows),
        margin_gradients=torch.cat(gradient_rows),
        hard_gate=torch.cat(gate_rows).bool(),
    )


def compile_iterative_margin_deltas(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    hidden_state_index: int,
    target_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    desired_margin: float,
    iterations: int,
    max_relative_norm: float,
    device: torch.device | str,
    batch_size: int = 4,
) -> IterativeMarginTrace:
    """Relinearize and update until the frozen iteration budget is exhausted."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    cumulative: torch.Tensor | None = None
    base_states: torch.Tensor | None = None
    trace = []
    for _ in range(iterations):
        plan = compile_local_margin_plan(
            model,
            tokenizer,
            prompts,
            hidden_state_index=hidden_state_index,
            target_token_ids=target_token_ids,
            candidate_token_ids=candidate_token_ids,
            device=device,
            batch_size=batch_size,
            preexisting_deltas=cumulative,
        )
        if base_states is None:
            base_states = plan.recipient_states
            cumulative = torch.zeros_like(base_states)
        assert cumulative is not None
        step = plan.deltas(
            desired_margin=desired_margin,
            max_relative_norm=None,
        )
        cumulative = project_relative_norm(
            cumulative + step,
            base_states,
            max_relative_norm=max_relative_norm,
        )
        trace.append(cumulative.clone())
    assert base_states is not None
    return IterativeMarginTrace(
        cumulative_deltas=tuple(trace),
        base_recipient_states=base_states,
    )
