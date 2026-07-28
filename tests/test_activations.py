from types import SimpleNamespace

import torch
from torch import nn

from open_latent_interfaces.activations import (
    ActivationCapture,
    last_nonpadding_positions,
)
from open_latent_interfaces.interventions import intervened_next_token_logits


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, prompts, **_kwargs):
        rows = [[int(piece) for piece in prompt.split()] for prompt in prompts]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class AddOneBlock(nn.Module):
    def forward(self, hidden):
        return (hidden + 1,)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([AddOneBlock(), AddOneBlock()])
        self.head = nn.Linear(4, 32, bias=False)

    def forward(
        self,
        input_ids,
        attention_mask,
        output_hidden_states=False,
        **_kwargs,
    ):
        hidden = self.embed(input_ids)
        states = [hidden]
        for block in self.model.layers:
            hidden = block(hidden)[0]
            states.append(hidden)
        return SimpleNamespace(
            hidden_states=tuple(states) if output_hidden_states else None,
            logits=self.head(hidden),
        )


def test_last_nonpadding_positions_supports_left_and_right_padding() -> None:
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    assert last_nonpadding_positions(mask).tolist() == [1, 3]


def test_activation_capture_uses_hf_hidden_state_indexing() -> None:
    model = FakeModel()
    capture = ActivationCapture(model, FakeTokenizer(), device="cpu")
    captured = capture.capture_last_token(
        ["1 2 3", "4 5"],
        hidden_state_indices=[1, 2],
        batch_size=2,
    )
    difference = captured[2].values - captured[1].values
    assert torch.allclose(difference, torch.ones_like(difference))


def test_intervention_adds_delta_at_selected_boundary() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    prompts = ["1 2 3", "4 5"]
    zero = torch.zeros(2, 4)
    delta = torch.ones(2, 4)
    base = intervened_next_token_logits(
        model,
        tokenizer,
        prompts,
        hidden_state_index=1,
        deltas=zero,
        device="cpu",
    )
    moved = intervened_next_token_logits(
        model,
        tokenizer,
        prompts,
        hidden_state_index=1,
        deltas=delta,
        device="cpu",
    )
    expected = model.head(delta)
    assert torch.allclose(moved - base, expected, atol=1e-6)
