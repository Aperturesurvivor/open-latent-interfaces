from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
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
