from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from open_latent_interfaces.interpretability import (
    LatentSite,
    corroborate,
    read_jsonl,
    vector_sha256,
    write_jsonl,
)
from open_latent_interfaces.interpretability_backends import (
    JACOBIAN_LENS_REVISION,
    NLA_INFERENCE_REVISION,
    JacobianLensAdapter,
    NLAAdapter,
)


class FakeNLAClient:
    def generate(self, activation: np.ndarray, **_: object) -> str:
        assert activation.shape == (3,)
        return "a positive arithmetic-result direction"


class FakeNLACritic:
    def reconstruct(self, explanation: str) -> torch.Tensor:
        assert "arithmetic" in explanation
        return torch.tensor([1.0, 2.0, 3.0])


class FakeLens:
    d_model = 3
    n_prompts = 100

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        assert layer == 1
        return residual * 2


class FakeLensModel:
    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return torch.tensor([residual[0], residual[1], residual[2], -1.0])


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return ("zero", "one", "two", "three")[token_ids[0]]


def site_kwargs() -> dict[str, object]:
    return {
        "example_id": "addition-7-plus-8",
        "target_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "target_model_revision": "a" * 40,
        "hidden_state_index": 2,
        "token_position": -1,
    }


def test_vector_hash_is_canonical_float32() -> None:
    assert vector_sha256([1, 2, 3]) == vector_sha256(
        np.array([1, 2, 3], dtype=np.float64)
    )


def test_nla_adapter_records_explanation_reconstruction_and_provenance() -> None:
    adapter = NLAAdapter(
        FakeNLAClient(),
        critic=FakeNLACritic(),
        actor_checkpoint="kitft/fake-av",
        actor_checkpoint_revision="actor-revision",
        critic_checkpoint="kitft/fake-ar",
        critic_checkpoint_revision="critic-revision",
    )
    artifact = adapter.readout(np.array([1, 2, 3]), **site_kwargs())

    assert artifact.method.revision == NLA_INFERENCE_REVISION
    assert artifact.method.license == "Apache-2.0"
    assert artifact.observation["interpretation_role"] == "hypothesis_only"
    assert artifact.reconstruction is not None
    assert artifact.reconstruction["cosine_similarity"] == pytest.approx(1.0)
    assert artifact.reconstruction["vector"]["values"] is None
    assert artifact.corroboration.status == "hypothesis"


def test_jacobian_adapter_records_tokens_and_transported_direction() -> None:
    adapter = JacobianLensAdapter(
        FakeLens(),
        FakeLensModel(),
        FakeTokenizer(),
        lens_checkpoint="local/fake-lens.pt",
        lens_checkpoint_revision="lens-revision",
    )
    artifact = adapter.readout(np.array([1, 2, 3]), top_k=2, **site_kwargs())

    assert artifact.method.revision == JACOBIAN_LENS_REVISION
    assert artifact.observation["tokens"][0]["token"] == "two"
    assert artifact.observation["tokens"][0]["rank"] == 1
    assert artifact.observation["transported_direction"]["values"] is None


def test_corroboration_requires_same_site_and_independent_family() -> None:
    nla = NLAAdapter(
        FakeNLAClient(),
        actor_checkpoint="fake-av",
        actor_checkpoint_revision="revision",
    ).readout(np.array([1, 2, 3]), **site_kwargs())
    lens_adapter = JacobianLensAdapter(
        FakeLens(),
        FakeLensModel(),
        FakeTokenizer(),
        lens_checkpoint="fake-lens",
        lens_checkpoint_revision="revision",
    )
    lens = lens_adapter.readout(np.array([1, 2, 3]), **site_kwargs())

    reviewed = corroborate(
        nla, lens, status="corroborated", note="J-lens top tokens match the hypothesis."
    )
    assert reviewed.corroboration.status == "corroborated"
    assert reviewed.corroboration.artifact_ids == (lens.artifact_id,)
    with pytest.raises(ValueError, match="different method"):
        corroborate(nla, nla, status="corroborated", note="self-confirmation")


def test_jsonl_round_trip(tmp_path) -> None:
    artifact = NLAAdapter(
        FakeNLAClient(),
        critic=FakeNLACritic(),
        actor_checkpoint="fake-av",
        actor_checkpoint_revision="revision",
    ).readout(np.array([1, 2, 3]), **site_kwargs())
    path = tmp_path / "readouts.jsonl"
    write_jsonl(path, [artifact])
    loaded = read_jsonl(path)

    assert loaded == [artifact]
    assert json.loads(path.read_text())["schema_version"].endswith("/v1")


def test_site_hash_distinguishes_activations() -> None:
    first = LatentSite.from_activation(
        [1, 2],
        target_model="model",
        target_model_revision="b" * 40,
        hidden_state_index=1,
        token_position=3,
    )
    second = LatentSite.from_activation(
        [1, 3],
        target_model="model",
        target_model_revision="b" * 40,
        hidden_state_index=1,
        token_position=3,
    )
    assert first != second
