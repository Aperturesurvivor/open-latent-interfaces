# Prior Art and Claims Boundary

Last reviewed: 2026-07-27

This is a working map, not an exhaustive novelty review.

## Direct foundations

- **Natural Language Autoencoders** (Fraser-Taliente, Kantamneni, Ong et al.,
  2026) jointly train an activation verbalizer and activation reconstructor
  through a natural-language bottleneck. The authors released Apache-2.0
  training code and checkpoints for Qwen2.5-7B, Gemma 3, and Llama 3.3.
  Explanations can confabulate, are not mechanistically grounded, and are
  expensive. Their edit-and-reconstruct intervention is direct prior art for
  converting text edits into residual-stream steering vectors.
- **Jacobian Lens / J-space** (Gurnee, Sofroniew et al., 2026) maps intermediate
  residual directions to their average causal effect on current and future
  verbalization. The work demonstrates reading, ablation, concept swapping, and
  a sparse workspace-like subframe. Its public implementation is a reference
  release rather than a maintained library.
- **Circuit tracing** (Lindsey et al., 2025) uses cross-layer transcoders and
  attribution graphs to approximate causal feature interactions. Anthropic
  released open-model tooling and a Neuronpedia interface.
- **Activation Oracles / LatentQA** (Karvonen et al., 2025) train language
  models to answer natural-language questions about activations and show
  out-of-distribution decoding. Current evidence also documents sensitivity to
  question phrasing and hallucination.
- **Patchscopes** (Ghandeharioun et al., 2024) patches hidden states into a
  target prompt/model for natural-language inspection and demonstrates
  reasoning-error correction. Later work identifies decoder-prior bias as a
  faithfulness failure mode.

## Read/write and deterministic-computation neighbors

- **Activation addition and contrastive activation addition** establish
  inference-time steering in frozen models using residual directions.
- **HyperSteer** (Sun et al., 2025) trains hypernetworks to generate
  prompt-conditioned residual steering vectors.
- **Integrated Gated Calculator** (Dietz and Klakow, 2025) integrates a
  calculator into a fine-tuned Llama model and is direct prior art for internal
  deterministic arithmetic.
- **Neural Firmware Arithmetic** (Wilson with Codex implementation assistance,
  2026) tests frozen deterministic arithmetic mechanisms inside native
  transformer activation pathways, with causal ablations and explicit
  routing/typing/decoding failure boundaries.

## What this repository may claim now

- It defines an open experimental program connecting activation
  interpretation, causal latent mapping, and frozen deterministic grafts.
- It provides model-agnostic Hugging Face instrumentation and a reproducible
  Phase 0 pilot protocol.
- It can report only results actually present in committed result artifacts.

## What it may not claim

- Historical invention of activation interpretation, steering, latent
  interfaces, internal calculators, or model editing.
- That NLA prose is a faithful transcript of thought.
- That a probe-discovered direction is used by the model.
- That a successful activation intervention identifies the full underlying
  circuit.
- That a workflow works for any model before multi-family replication.
- That deterministic reasoning is achieved until route, typed read,
  deterministic execution, write, multi-token output, and preservation gates
  pass together.

## Primary sources

- https://transformer-circuits.pub/2026/nla/
- https://github.com/kitft/natural_language_autoencoders
- https://transformer-circuits.pub/2026/workspace/
- https://github.com/anthropics/jacobian-lens
- https://transformer-circuits.pub/2025/attribution-graphs/
- https://arxiv.org/abs/2512.15674
- https://arxiv.org/abs/2401.06102
- https://arxiv.org/abs/2506.03292
- https://arxiv.org/abs/2501.00684

