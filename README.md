# Open Latent Interfaces

Open Latent Interfaces is an independent research project for discovering,
causally validating, and using native representational interfaces inside
frozen language models.

The long-run target is a **deterministic latent graft**:

```text
native model state
  → typed read bridge
  → frozen deterministic mechanism
  → native write bridge
  → unchanged downstream model
```

Every base-model parameter remains frozen. Model-specific probes or bridges may
be trained, then frozen for inference.

## Current status

**Phase 0 engineering pilot.** The repository currently contains:

- deterministic addition/contrast dataset generation with split-disjoint
  operand pairs;
- model-agnostic Hugging Face residual-stream capture;
- ridge route and scalar-value probes;
- temporary residual interventions at an explicit hidden-state boundary;
- a probe-defined, oracle-result write-path diagnostic;
- tests, a research ladder, a protocol, and a claims boundary.

No NLA has been trained here. No deterministic graft has passed an end-to-end
audit. Probe decodability is not treated as causal use.

## Why this is separate from ordinary activation steering

The project is not trying only to make a model more likely to discuss
mathematics. It aims to recover a typed interface—route, operands, operation,
timing, and result representation—that can connect a model's existing
computation to an auditable deterministic mechanism.

## Quick start

Requirements: Python 3.12 and `uv`.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run oli-phase0 --smoke
```

The default smoke run downloads `Qwen/Qwen2.5-0.5B-Instruct`, keeps all of its
parameters frozen, captures two residual boundaries, fits probes, and runs a
small causal result-direction pilot. Compact output is written to
`results/phase0_local_pilot.json`.

For the full local pilot:

```bash
uv run oli-phase0 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output results/phase0_qwen05b_pilot.json
```

Read [the Phase 0 protocol](protocols/PHASE0_NATIVE_MATH_CHANNELS.md) before
interpreting the output.

## Research map

- [Research program](docs/RESEARCH_PROGRAM.md)
- [Prior art and claims boundary](docs/PRIOR_ART_AND_CLAIMS.md)
- [Phase 0 pilot protocol](protocols/PHASE0_NATIVE_MATH_CHANNELS.md)

## Upstream foundations

This work builds on, and does not replace:

- Anthropic's
  [Natural Language Autoencoders](https://github.com/kitft/natural_language_autoencoders);
- Anthropic's
  [Jacobian Lens](https://github.com/anthropics/jacobian-lens);
- circuit tracing and sparse-feature interpretability;
- Activation Oracles / LatentQA;
- Patchscopes and representation engineering;
- the Integrated Gated Calculator;
- Josiah Wilson's
  [Neural Firmware Arithmetic](https://github.com/Aperturesurvivor/neural-firmware-arithmetic).

## Attribution

Josiah Wilson originated the open-latent-interface research direction, the
task-specific thinking-channel mapping goal, and the proposal to attach
deterministic mechanisms to a model's existing mathematical representations
without fine-tuning the base model onto a new vector path.

OpenAI Codex assists with research synthesis, experimental design,
implementation, execution, analysis, and documentation under Josiah's
direction. Upstream work remains attributed to its original authors.

## License

Apache License 2.0. Model checkpoints and downloaded dependencies retain their
own licenses.
