# Interpretability Backends

This project treats an interpretability readout as an observation about a
specific activation—not as a transcript of hidden thought. Natural-language
explanations begin with `hypothesis` status. A readout can become
`corroborated` only when an artifact from a different method family refers to
the exact same model revision, hidden-state index, token position, and
activation hash.

## Pinned upstream implementations

| Method | Repository | Pinned commit | License |
|---|---|---|---|
| Natural Language Autoencoder inference | `kitft/nla-inference` | `38b802a33d1d317f21b6825a9116f388c2141f86` | Apache-2.0 |
| Jacobian lens | `anthropics/jacobian-lens` | `581d398613e5602a5af361e1c34d3a92ea82ba8e` | Apache-2.0 |

Checkpoint identifiers and immutable checkpoint revisions are recorded
separately from code provenance. Model checkpoints, corpora, and downloaded
weights retain their own licenses.

On 2026-07-27, the pinned J-lens commit passed its full upstream test suite
(32 tests) in an isolated environment. The OLI adapter was then exercised with
that release's real `JacobianLens` class. The pinned NLA module was imported
directly and its `NLAClient.generate` and `NLACritic.reconstruct` signatures
were checked against the adapter contract. No claim of real-checkpoint NLA
inference is made by those interface checks.

## Artifact contract

`schemas/interpretability-artifact-v1.schema.json` is the portable JSON
contract. The Python implementation is in
`open_latent_interfaces.interpretability`.

Each record contains:

- immutable target-model and method revisions;
- an explicit hidden-state index and token position;
- the SHA-256 of the canonical little-endian float32 activation;
- a method-specific observation;
- an optional vector reconstruction with its own hash and round-trip metrics;
- limitations and a machine-checkable corroboration state.

Vectors are hash-only by default. Callers may opt into inline values for an
auditable transfer artifact, but large activation collections should remain in
a separately checksummed tensor store.

An artifact ID hashes the scientific payload but not its creation timestamp.
Rerunning an identical deterministic backend therefore yields the same ID.

## NLA adapter

`NLAAdapter` accepts the released `NLAClient` and optional `NLACritic` by duck
typing. The AV explanation is stored as an `activation_verbalization`. When an
AR critic is supplied, its reconstructed vector, direction-normalized MSE, and
cosine similarity are stored under `reconstruction`.

The AV and AR were trained as one autoencoding system. A strong round trip is
useful fidelity evidence, but it is not independent semantic corroboration.
The record remains `hypothesis` until a separate method agrees.

The adapter does not import NLA, SGLang, or its server stack. Run the upstream
client in an environment pinned to its commit, then pass its client objects to
the adapter or transfer the resulting JSONL records.

## Jacobian-lens adapter

`JacobianLensAdapter` uses the released bare interfaces:

```python
transported = lens.transport(activation, hidden_state_index)
logits = model.unembed(transported)
```

It records ranked token IDs, decoded token strings, logits, and a checksummed
transported direction. This direct path ensures that NLA and J-lens can inspect
the exact same activation. The prompt-level `lens.apply()` API remains useful
for visualization, but a separately captured activation is preferable for
cross-method identity.

The schema uses Hugging Face's hidden-state convention: index 0 is the embedding
state and index `i + 1` is decoder block `i`'s output. J-lens indexes decoder
blocks directly, so the adapter performs this translation explicitly and
records `jacobian_source_layer` in every observation.

The current reference release declares `transformers>=5.5`, while this
repository's tested Phase 0 harness is intentionally pinned below 5. Run
J-lens in an isolated environment rather than silently changing the tested
activation-capture stack.

## Compute requirements

The lightweight artifact/adaptor layer runs in the normal local environment
and its offline fake-backend tests require no downloaded weights.

Released NLA inference is not a small-model local test: the smallest published
pair targets Qwen2.5-7B at layer 20. The AV requires a full SGLang-served actor;
the AR loads a truncated backbone plus reconstruction head. Plan a GPU
environment for a real paired run, pin the actor and critic checkpoint
revisions, and disable SGLang's radix cache as required upstream. This project
does not claim that the released pair fits the 16 GB local machine.

J-lens application requires the target model plus fitted Jacobian matrices.
Fitting additionally requires backward passes over a corpus. The upstream
reference used 1,000 sequences of 128 tokens and reports that roughly 100
prompts is usable. Storage scales as one `d_model × d_model` matrix per fitted
layer: float32 is approximately `4 × layers × d_model²` bytes before checkpoint
compression. Start with Qwen2.5-0.5B and a small development corpus locally;
reserve the frozen audit for a separately fitted lens and held-out prompts.

## Minimal use

```python
from open_latent_interfaces.interpretability_backends import (
    JacobianLensAdapter,
    NLAAdapter,
)

nla = NLAAdapter(
    client,
    critic=critic,
    actor_checkpoint="kitft/nla-qwen2.5-7b-L20-av",
    actor_checkpoint_revision="<immutable Hugging Face revision>",
    critic_checkpoint="kitft/nla-qwen2.5-7b-L20-ar",
    critic_checkpoint_revision="<immutable Hugging Face revision>",
)
nla_artifact = nla.readout(activation, **site_metadata)

jlens_backend = JacobianLensAdapter(
    lens,
    lens_model,
    tokenizer,
    lens_checkpoint="path-or-hub-id/lens.pt",
    lens_checkpoint_revision="<immutable lens revision>",
)
jlens_artifact = jlens_backend.readout(activation, **site_metadata)
```

Only a deliberate review step should call `corroborate(...)`; superficial token
overlap is not automatically converted into scientific evidence.
