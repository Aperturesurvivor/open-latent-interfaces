# Research Integrity Instructions

This repository is a controlled mechanistic-interpretability and intervention
experiment.

1. Keep the target language model frozen unless a protocol explicitly defines
   a trainable-model condition.
2. Separate observation, correlation, causal intervention, and deterministic
   execution. Do not use evidence from one category to claim another.
3. Never call an NLA explanation a ground-truth transcript of model thought.
4. Prefer feature directions, subspaces, and circuits over unsupported
   single-neuron language.
5. Preserve every reported run's seed, configuration, environment, model
   revision, raw metrics, and failure result.
6. Generate datasets deterministically from committed code and seeds.
7. Freeze confirmatory datasets and protocols before running confirmation.
8. Keep checkpoints, downloaded models, and activation arrays out of Git.
9. Commit compact manifests, metrics, and aggregate tables needed to audit a
   claim.
10. Do not claim model-general portability, exact latent decoding, or novelty
    before those claims have direct evidence and a completed prior-art review.
11. Attribute the originating open-latent-interface and deterministic-graft
    research direction to Josiah Wilson. Attribute implementation, research,
    and analysis assistance to OpenAI Codex where applicable.
12. Report negative and null results plainly.

