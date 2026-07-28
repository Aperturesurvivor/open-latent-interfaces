from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

from open_latent_interfaces.activations import (
    last_nonpadding_positions,
    resolve_decoder_blocks,
)


class OnlineAdapterHook:
    """Apply a trainable adapter to the final active token in each row."""

    def __init__(
        self,
        adapter: Any,
        target_digits: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        scale: float,
        norm_cap: float,
    ) -> None:
        self.adapter = adapter
        self.target_digits = target_digits
        self.attention_mask = attention_mask
        self.scale = scale
        self.norm_cap = norm_cap
        self.applied_delta: torch.Tensor | None = None
        self.recipient_states: torch.Tensor | None = None

    def __call__(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        positions = last_nonpadding_positions(self.attention_mask)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        recipient = hidden[rows, positions]
        delta = self.adapter(recipient, self.target_digits.to(hidden.device)) * self.scale
        maximum = recipient.float().norm(dim=1) * self.norm_cap
        factor = (maximum / delta.norm(dim=1).clamp_min(1e-12)).clamp(max=1.0)
        delta = delta * factor[:, None]
        modified = hidden.clone()
        modified[rows, positions] += delta.to(hidden.dtype)
        self.applied_delta = delta
        self.recipient_states = recipient
        return _replace_hidden(output, modified)


@contextmanager
def online_adapter_intervention(
    model: Any,
    *,
    hidden_state_index: int,
    adapter: Any,
    target_digits: torch.Tensor,
    attention_mask: torch.Tensor,
    scale: float,
    norm_cap: float,
):
    blocks = resolve_decoder_blocks(model)
    hook = OnlineAdapterHook(
        adapter,
        target_digits,
        attention_mask,
        scale=scale,
        norm_cap=norm_cap,
    )
    handle = blocks[hidden_state_index - 1].register_forward_hook(hook)
    try:
        yield hook
    finally:
        handle.remove()


def _replace_hidden(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    raise TypeError(f"unsupported decoder block output: {type(output)!r}")


def last_token_addition_hook(
    deltas: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Callable[[Any, tuple[Any, ...], Any], Any]:
    """Build a hook that adds one vector per row at its last prompt token."""

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != deltas.shape[0]:
            raise ValueError("hook batch does not match intervention batch")
        modified = hidden.clone()
        positions = last_nonpadding_positions(attention_mask)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        modified[rows, positions] += deltas.to(device=hidden.device, dtype=hidden.dtype)
        return _replace_hidden(output, modified)

    return hook


def one_shot_last_token_addition_hook(
    deltas: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Callable[[Any, tuple[Any, ...], Any], Any]:
    """Add the prompt-boundary delta once, then leave generated tokens untouched."""

    applied = False

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        nonlocal applied
        if applied:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != deltas.shape[0]:
            raise ValueError("hook batch does not match intervention batch")
        modified = hidden.clone()
        positions = last_nonpadding_positions(attention_mask)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        modified[rows, positions] += deltas.to(device=hidden.device, dtype=hidden.dtype)
        applied = True
        return _replace_hidden(output, modified)

    return hook


def one_shot_sequence_addition_hook(
    deltas: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Callable[[Any, tuple[Any, ...], Any], Any]:
    """Add a padded sequence delta once during the prompt forward pass."""

    if deltas.ndim != 3:
        raise ValueError("sequence deltas must have shape [batch, tokens, width]")
    if attention_mask.ndim != 2 or deltas.shape[:2] != attention_mask.shape:
        raise ValueError("sequence deltas and attention mask must share batch/token shape")
    applied = False

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        nonlocal applied
        if applied:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape != deltas.shape:
            raise ValueError("hook hidden state does not match sequence intervention")
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        modified = hidden + deltas.to(device=hidden.device, dtype=hidden.dtype) * mask
        applied = True
        return _replace_hidden(output, modified)

    return hook


@contextmanager
def residual_intervention(
    model: Any,
    *,
    hidden_state_index: int,
    deltas: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Temporarily add residual deltas at a HF hidden-state boundary.

    Hidden-state index 0 is the embedding output and is not supported by this
    block hook. Index ``k`` patches the output of decoder block ``k - 1``.
    """

    if hidden_state_index < 1:
        raise ValueError("block intervention requires hidden_state_index >= 1")
    blocks = resolve_decoder_blocks(model)
    block_index = hidden_state_index - 1
    if block_index >= len(blocks):
        raise IndexError(f"block index {block_index} outside model with {len(blocks)} blocks")
    handle = blocks[block_index].register_forward_hook(
        last_token_addition_hook(deltas, attention_mask)
    )
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def one_shot_residual_intervention(
    model: Any,
    *,
    hidden_state_index: int,
    deltas: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Patch one prompt boundary during cached autoregressive generation."""

    if hidden_state_index < 1:
        raise ValueError("block intervention requires hidden_state_index >= 1")
    blocks = resolve_decoder_blocks(model)
    block_index = hidden_state_index - 1
    if block_index >= len(blocks):
        raise IndexError(f"block index {block_index} outside model with {len(blocks)} blocks")
    handle = blocks[block_index].register_forward_hook(
        one_shot_last_token_addition_hook(deltas, attention_mask)
    )
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def one_shot_sequence_residual_intervention(
    model: Any,
    *,
    hidden_state_index: int,
    deltas: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Patch all active prompt tokens once during cached generation."""

    if hidden_state_index < 1:
        raise ValueError("block intervention requires hidden_state_index >= 1")
    blocks = resolve_decoder_blocks(model)
    block_index = hidden_state_index - 1
    if block_index >= len(blocks):
        raise IndexError(f"block index {block_index} outside model with {len(blocks)} blocks")
    handle = blocks[block_index].register_forward_hook(
        one_shot_sequence_addition_hook(deltas, attention_mask)
    )
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def intervened_next_token_logits(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    hidden_state_index: int,
    deltas: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    device = torch.device(device)
    encoded = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }
    if deltas.shape[0] != len(prompts):
        raise ValueError("one intervention delta is required per prompt")
    with residual_intervention(
        model,
        hidden_state_index=hidden_state_index,
        deltas=deltas,
        attention_mask=encoded["attention_mask"],
    ):
        outputs = model(**encoded, use_cache=False, return_dict=True)
    positions = last_nonpadding_positions(encoded["attention_mask"])
    rows = torch.arange(len(prompts), device=device)
    return outputs.logits[rows, positions].detach().float().cpu()


@torch.inference_mode()
def intervened_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    hidden_state_index: int,
    deltas: torch.Tensor,
    device: torch.device | str,
    max_new_tokens: int = 8,
) -> list[str]:
    """Greedily generate after applying one prompt-boundary residual delta."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    finally:
        tokenizer.padding_side = previous_padding_side
    device = torch.device(device)
    encoded = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }
    if deltas.shape[0] != len(prompts):
        raise ValueError("one intervention delta is required per prompt")
    with one_shot_residual_intervention(
        model,
        hidden_state_index=hidden_state_index,
        deltas=deltas,
        attention_mask=encoded["attention_mask"],
    ):
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuations = generated[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(continuations, skip_special_tokens=True)


@torch.inference_mode()
def intervened_generate_sequence(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    hidden_state_index: int,
    deltas: torch.Tensor,
    device: torch.device | str,
    max_new_tokens: int = 8,
) -> list[str]:
    """Greedily generate after one full-prompt residual intervention."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    finally:
        tokenizer.padding_side = previous_padding_side
    if deltas.ndim != 3 or deltas.shape[:2] != encoded["input_ids"].shape:
        raise ValueError(
            "sequence deltas must match the tokenizer-padded prompt batch"
        )
    device = torch.device(device)
    encoded = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }
    with one_shot_sequence_residual_intervention(
        model,
        hidden_state_index=hidden_state_index,
        deltas=deltas,
        attention_mask=encoded["attention_mask"],
    ):
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuations = generated[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(continuations, skip_special_tokens=True)
