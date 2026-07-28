from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch


def last_nonpadding_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2 or bool((attention_mask.sum(dim=1) == 0).any()):
        raise ValueError("attention mask must contain at least one token per row")
    return attention_mask.shape[1] - 1 - attention_mask.flip(dims=(1,)).argmax(dim=1)


@dataclass(frozen=True)
class CapturedLayer:
    """Last-nonpadding-token residual states for one hidden-state index."""

    hidden_state_index: int
    values: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("captured values must have shape [examples, width]")


@dataclass(frozen=True)
class CapturedSequences:
    """Unpadded residual sequences for one hidden-state index."""

    hidden_state_index: int
    values: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("captured sequences cannot be empty")
        if any(value.ndim != 2 for value in self.values):
            raise ValueError("each captured sequence must have shape [tokens, width]")
        widths = {value.shape[1] for value in self.values}
        if len(widths) != 1:
            raise ValueError("captured sequence widths must match")


class ActivationCapture:
    """Capture residual-stream states from Hugging Face causal language models.

    Hugging Face returns the embedding state at index 0 and the output of block
    ``i`` at hidden-state index ``i + 1``. This class preserves that convention
    explicitly so intervention code cannot silently confuse layer numbering.
    """

    def __init__(self, model: Any, tokenizer: Any, *, device: torch.device | str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @torch.inference_mode()
    def capture_last_token(
        self,
        prompts: list[str],
        *,
        hidden_state_indices: Iterable[int] | None = None,
        batch_size: int = 8,
    ) -> dict[int, CapturedLayer]:
        if not prompts:
            raise ValueError("at least one prompt is required")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        requested = None if hidden_state_indices is None else tuple(hidden_state_indices)
        chunks: dict[int, list[torch.Tensor]] = {}
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in encoded.items()
            }
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            indices = tuple(range(len(hidden_states))) if requested is None else requested
            lengths = last_nonpadding_positions(encoded["attention_mask"])
            rows = torch.arange(len(batch_prompts), device=self.device)
            for index in indices:
                if index < 0 or index >= len(hidden_states):
                    raise IndexError(
                        f"hidden-state index {index} outside [0, {len(hidden_states) - 1}]"
                    )
                selected = hidden_states[index][rows, lengths]
                chunks.setdefault(index, []).append(selected.detach().float().cpu())

        return {
            index: CapturedLayer(index, torch.cat(values, dim=0))
            for index, values in chunks.items()
        }

    @torch.inference_mode()
    def capture_sequences(
        self,
        prompts: list[str],
        *,
        hidden_state_indices: Iterable[int] | None = None,
        batch_size: int = 8,
    ) -> dict[int, CapturedSequences]:
        """Capture every active prompt token, excluding tokenizer padding."""

        if not prompts:
            raise ValueError("at least one prompt is required")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        requested = None if hidden_state_indices is None else tuple(hidden_state_indices)
        chunks: dict[int, list[torch.Tensor]] = {}
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in encoded.items()
            }
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            indices = tuple(range(len(hidden_states))) if requested is None else requested
            active = encoded["attention_mask"].bool()
            for index in indices:
                if index < 0 or index >= len(hidden_states):
                    raise IndexError(
                        f"hidden-state index {index} outside [0, {len(hidden_states) - 1}]"
                    )
                state = hidden_states[index]
                chunks.setdefault(index, []).extend(
                    state[row, active[row]].detach().float().cpu()
                    for row in range(len(batch_prompts))
                )

        return {
            index: CapturedSequences(index, tuple(values))
            for index, values in chunks.items()
        }

    @torch.inference_mode()
    def next_token_logits(self, prompts: list[str], *, batch_size: int = 8) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(prompts), batch_size):
            encoded = self.tokenizer(
                prompts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in encoded.items()
            }
            outputs = self.model(**encoded, use_cache=False, return_dict=True)
            lengths = last_nonpadding_positions(encoded["attention_mask"])
            rows = torch.arange(lengths.shape[0], device=self.device)
            chunks.append(outputs.logits[rows, lengths].detach().float().cpu())
        return torch.cat(chunks, dim=0)


def resolve_decoder_blocks(model: Any) -> Any:
    """Find the repeated decoder block collection on common HF causal LMs."""

    candidates = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for path in candidates:
        value = model
        try:
            for name in path:
                value = getattr(value, name)
        except AttributeError:
            continue
        if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
            return value
    raise TypeError("could not locate repeated decoder blocks on this model")
