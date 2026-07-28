from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from open_latent_interfaces.interpretability import (
    InterpretabilityArtifact,
    LatentSite,
    MethodProvenance,
    VectorRecord,
)

NLA_INFERENCE_REVISION = "38b802a33d1d317f21b6825a9116f388c2141f86"
JACOBIAN_LENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


class NLAAdapter:
    """Optional, duck-typed adapter for kitft/nla-inference.

    The caller owns the heavyweight NLAClient/NLACritic processes. Keeping the
    adapter duck typed avoids making SGLang, Transformers 5, or NLA checkpoints
    hard dependencies of the core instrumentation package.
    """

    def __init__(
        self,
        client: Any,
        *,
        critic: Any | None = None,
        actor_checkpoint: str,
        actor_checkpoint_revision: str,
        critic_checkpoint: str | None = None,
        critic_checkpoint_revision: str | None = None,
    ) -> None:
        self.client = client
        self.critic = critic
        self.method = MethodProvenance(
            family="natural_language_autoencoder",
            implementation="kitft/nla-inference",
            repository="https://github.com/kitft/nla-inference",
            revision=NLA_INFERENCE_REVISION,
            license="Apache-2.0",
            checkpoint=actor_checkpoint,
            checkpoint_revision=actor_checkpoint_revision,
        )
        self.critic_checkpoint = critic_checkpoint
        self.critic_checkpoint_revision = critic_checkpoint_revision

    def readout(
        self,
        activation: Any,
        *,
        example_id: str,
        target_model: str,
        target_model_revision: str,
        hidden_state_index: int,
        token_position: int,
        include_reconstruction_values: bool = False,
        generation: dict[str, Any] | None = None,
    ) -> InterpretabilityArtifact:
        vector = np.asarray(activation, dtype=np.float32).reshape(-1)
        site = LatentSite.from_activation(
            vector,
            target_model=target_model,
            target_model_revision=target_model_revision,
            hidden_state_index=hidden_state_index,
            token_position=token_position,
        )
        explanation = self.client.generate(vector, **(generation or {}))
        observation = {
            "kind": "activation_verbalization",
            "explanation": explanation,
            "interpretation_role": "hypothesis_only",
        }
        reconstruction = None
        if self.critic is not None:
            predicted = np.asarray(
                self.critic.reconstruct(explanation), dtype=np.float32
            ).reshape(-1)
            if predicted.shape != vector.shape:
                raise ValueError(
                    f"NLA reconstruction shape {predicted.shape} != activation {vector.shape}"
                )
            pred_norm = predicted / max(float(np.linalg.norm(predicted)), 1e-12)
            gold_norm = vector / max(float(np.linalg.norm(vector)), 1e-12)
            cosine = float(np.dot(pred_norm, gold_norm))
            reconstruction = {
                "kind": "activation_reconstruction",
                "vector": asdict(
                    VectorRecord.from_value(
                        predicted, include_values=include_reconstruction_values
                    )
                ),
                "direction_mse": float(np.mean((pred_norm - gold_norm) ** 2) * vector.size),
                "cosine_similarity": cosine,
                "critic_checkpoint": self.critic_checkpoint,
                "critic_checkpoint_revision": self.critic_checkpoint_revision,
                "interpretation": (
                    "Round-trip fidelity is dependent evidence from the same NLA "
                    "pair; it does not independently establish semantic faithfulness."
                ),
            }
        return InterpretabilityArtifact.create(
            example_id=example_id,
            site=site,
            method=self.method,
            observation=observation,
            reconstruction=reconstruction,
            limitations=(
                "Natural-language output can hallucinate or confabulate.",
                "An AV/AR round trip is not independent semantic corroboration.",
                "Treat the explanation as a hypothesis until another method agrees.",
            ),
        )


class JacobianLensAdapter:
    """Optional, duck-typed adapter for anthropics/jacobian-lens."""

    def __init__(
        self,
        lens: Any,
        model: Any,
        tokenizer: Any,
        *,
        lens_checkpoint: str,
        lens_checkpoint_revision: str,
    ) -> None:
        self.lens = lens
        self.model = model
        self.tokenizer = tokenizer
        self.method = MethodProvenance(
            family="jacobian_lens",
            implementation="anthropics/jacobian-lens",
            repository="https://github.com/anthropics/jacobian-lens",
            revision=JACOBIAN_LENS_REVISION,
            license="Apache-2.0",
            checkpoint=lens_checkpoint,
            checkpoint_revision=lens_checkpoint_revision,
        )

    @torch.no_grad()
    def readout(
        self,
        activation: Any,
        *,
        example_id: str,
        target_model: str,
        target_model_revision: str,
        hidden_state_index: int,
        token_position: int,
        top_k: int = 10,
        include_transport_values: bool = False,
    ) -> InterpretabilityArtifact:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if hidden_state_index < 1:
            raise ValueError(
                "J-lens reads decoder-block outputs; hidden-state index 0 is embeddings"
            )
        vector = torch.as_tensor(np.asarray(activation, dtype=np.float32)).reshape(-1)
        if vector.numel() != self.lens.d_model:
            raise ValueError(
                f"activation width {vector.numel()} != lens width {self.lens.d_model}"
            )
        site = LatentSite.from_activation(
            vector,
            target_model=target_model,
            target_model_revision=target_model_revision,
            hidden_state_index=hidden_state_index,
            token_position=token_position,
        )
        # HF hidden_states[i + 1] is the output of decoder block i. J-lens
        # indexes those blocks directly, so translate the public site convention.
        jacobian_source_layer = hidden_state_index - 1
        transported = self.lens.transport(vector, jacobian_source_layer)
        logits = self.model.unembed(transported).float().cpu().reshape(-1)
        count = min(top_k, logits.numel())
        scores, token_ids = torch.topk(logits, count)
        tokens = [
            {
                "token_id": int(token_id),
                "token": self.tokenizer.decode([int(token_id)]),
                "logit": float(score),
                "rank": rank,
            }
            for rank, (token_id, score) in enumerate(
                zip(token_ids.tolist(), scores.tolist(), strict=True), start=1
            )
        ]
        observation = {
            "kind": "jacobian_transport_unembedding",
            "tokens": tokens,
            "transported_direction": asdict(
                VectorRecord.from_value(
                    transported, include_values=include_transport_values
                )
            ),
            "jacobian_source_layer": jacobian_source_layer,
            "lens_n_prompts": int(self.lens.n_prompts),
            "interpretation_role": "model_disposition_readout",
        }
        return InterpretabilityArtifact.create(
            example_id=example_id,
            site=site,
            method=self.method,
            observation=observation,
            limitations=(
                "Token readout reflects average-Jacobian disposition, not a transcript.",
                "Top tokens do not by themselves establish causal use by the model.",
                "Corpus choice and lens-fit quality can change the readout.",
            ),
        )
