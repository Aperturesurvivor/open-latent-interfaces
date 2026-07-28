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

**Phase 0.1 completed; Phase 1 infrastructure underway.** The repository
currently contains:

- deterministic addition/contrast dataset generation with split-disjoint
  operand pairs;
- model-agnostic Hugging Face residual-stream capture;
- ridge route and scalar-value probes;
- temporary residual interventions at an explicit hidden-state boundary;
- a probe-defined, oracle-result write-path diagnostic;
- optional, dependency-light adapters for the released NLA and Jacobian-lens
  systems;
- a provenance-rich interpretability artifact schema that requires independent
  corroboration before an explanation can be marked as evidence;
- tests, a research ladder, a protocol, and a claims boundary.

No NLA has been trained here. No deterministic graft has passed an end-to-end
audit. Probe decodability is not treated as causal use.

The first Qwen2.5-0.5B pilot was an infrastructure success and a scientific
non-pass. Phase 0.1 then removed digit imbalance and template leakage, committed
its selected configuration before audit, and produced a clearer negative
result: leading-digit decoding stayed near chance, exact scalar recovery was
0/36, and the selected internal digit-probe direction lost to its random
control on aggregate margin. See the
[Phase 0.1 executive summary](PHASE01_EXECUTIVE_SUMMARY.md).

Phase 1A has now fit and exercised a real 24-prompt Jacobian lens across all
internal layers. It sharply improved recognition that a numeric token belongs
next, but its best selected layer identified the correct leading digit only
12/72 times (16.7% versus 11.1% balanced chance, exploratory and uncorrected).
The untouched model itself scored 7/72 at that boundary, so the next gate is a
capability sweep before causal cartography. See the
[Phase 1A executive summary](PHASE1A_EXECUTIVE_SUMMARY.md).

The follow-up capability gate moved to frozen Qwen2.5-1.5B-Instruct and an
instruction-aligned prompt contract. Its preselected mixed three-digit regime
passed 48/48 exact frozen-audit conditions across three templates and raw/chat
presentation. This establishes the behavioral envelope for Phase 1, not a
latent-mechanism claim. See the
[capability audit](CAPABILITY_GATE_V2_AUDIT.md).

The first 1.5B J-lens timing map then found a late transition: teacher-forced
three-digit readout stayed at 0% through block 22 and reached 91.7–97.2% only at
blocks 25–26. Vanilla logit lens behaved almost identically. Those layers are
too output-adjacent for the frozen Phase 1 interface gate, so probes and donor
patches—not interpretation prose—must now determine whether earlier
non-vocabulary state exists. See the
[Phase 1B J-lens summary](PHASE1B_JLENS_SUMMARY.md).

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

Phase 0.1 uses a separate development/frozen-audit workflow with balanced
leading digits and template families that never cross splits:

```bash
uv run oli-phase01 develop
# Review and commit configs/phase01_frozen.json before opening the audit.
uv run oli-phase01 audit --config configs/phase01_frozen.json
```

Read the
[Phase 0.1 protocol](protocols/PHASE01_BALANCED_CARTOGRAPHY.md) before running
either stage.

## Research map

- [Research program](docs/RESEARCH_PROGRAM.md)
- [Prior art and claims boundary](docs/PRIOR_ART_AND_CLAIMS.md)
- [Phase 0 pilot protocol](protocols/PHASE0_NATIVE_MATH_CHANNELS.md)
- [Phase 0.1 balanced-cartography protocol](protocols/PHASE01_BALANCED_CARTOGRAPHY.md)
- [Phase 1A J-lens development protocol](protocols/PHASE1_JLENS_DEVELOPMENT.md)
- [Phase 1 triangulated-cartography protocol](protocols/PHASE1_TRIANGULATED_CARTOGRAPHY.md)
- [Arithmetic capability gate](protocols/CAPABILITY_GATE.md)
- [Capability gate v1 results](CAPABILITY_GATE_V1_RESULTS.md)
- [Capability gate v2 development](CAPABILITY_GATE_V2_DEVELOPMENT.md)
- [Capability gate v2 frozen audit](CAPABILITY_GATE_V2_AUDIT.md)
- [Phase 0 lab notebook](PHASE0_LAB_NOTEBOOK.md)
- [Phase 0.1 lab notebook](PHASE01_LAB_NOTEBOOK.md)
- [Phase 1A executive summary](PHASE1A_EXECUTIVE_SUMMARY.md)
- [Phase 1A lab notebook](PHASE1A_LAB_NOTEBOOK.md)
- [Phase 1B J-lens development summary](PHASE1B_JLENS_SUMMARY.md)
- [Interpretability backends and evidence contract](docs/INTERPRETABILITY_BACKENDS.md)
- [Interpretability artifact JSON schema](schemas/interpretability-artifact-v1.schema.json)

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
