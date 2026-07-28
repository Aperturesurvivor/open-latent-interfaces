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

Layer-wide probes reproduced that timing: exact pre-output result decoding
peaked at 6.7%, while teacher-forced decoding reached 97.8% only at the final
hidden state. A tens-carry probe was 82.5% balanced-accurate at an internal
boundary, but flipping its probe score on 100% of examples produced no
counterfactual hundreds-digit effect and matched random controls. See the
[probe and causality summary](PHASE1B_PROBE_CAUSAL_SUMMARY.md).

A native full-residual donor patch then found the first causal write boundary:
at hidden state 23 / decoder block 22, targeted donors changed the next leading
digit on 42/45 examples versus 1/45 random and 2/45 shuffled controls.
Generated answers became the donor's leading digit plus the recipient's
original suffix on 38/45 examples. This is a causal, internal, sequential
digit-write interface. See the
[native donor-write summary](PHASE1C_NATIVE_WRITE_SUMMARY.md).

A frozen closed-loop follow-up reapplied matched native states at each of the
three answer positions. It generated the complete counterfactual donor result
on 38/45 development examples (84.4%), versus 0/45 for base, same-leading, and
random controls and 1/45 for a shuffled-donor control. This establishes a
causal native sequential write path, but it remains a large full-residual
transplant rather than a compact deterministic graft. See the
[stepwise native-write summary](PHASE1D_STEPWISE_NATIVE_WRITE_SUMMARY.md).

A donor-free compression then learned position-specific digit subspaces using
training data only. Its rank-8 leading-digit writer retained 36/45 target
transfer (80.0%) with clean control separation, but the closed-loop writer
produced only 2/45 complete results because the second-position interface did
not generalize. This partial pass localizes the next problem: counterfactual
prefixes require contextual state transport in addition to digit encoding. See
the [typed-writer summary](PHASE1E_TYPED_WRITER_SUMMARY.md).

A paired low-rank transport writer then learned from recipient/donor states
that already shared the target prefix. It improved second-digit transfer from
7/45 to 17/45 and complete targets from 2/45 to 5/45, but required later
interventions of 135–241% of residual norm and remained far below the 38/45
full-donor result. This supports contextual repair while showing that the
remaining transport is recipient-conditioned rather than a class-mean digit
vector. See the
[paired-transport summary](PHASE1F_PAIRED_TRANSPORT_SUMMARY.md).

A 480-pair conditional bridge then predicted low-rank transport from the
recipient state and desired digit. It reached 41/45 leading-digit transfer,
18/45 and 22/45 at later positions, and 6/45 exact results. Shuffling recipient
states reduced exact transfer to 0/45, while conditional transport cut later
intervention norms from 135–241% to 54–59%. Recipient conditioning is therefore
a causal part of the write path, but next-digit input alone remains
under-specified for reliable composition. See the
[conditional-transport summary](PHASE1G_CONDITIONAL_TRANSPORT_SUMMARY.md).

A full-result variant then supplied all three target digits at every position.
It regressed: teacher-forced transfer was only 19/45, 9/45, and 10/45, and
closed-loop exact writing fell to 1/45. Raw one-hot result interactions
over-specified the 480-pair regression and did not generalize to unseen result
combinations. This architecture was rejected without opening the audit. See the
[full-result transport summary](PHASE1H_FULL_RESULT_TRANSPORT_SUMMARY.md).

A digit-restricted nearest-state dictionary then tested whether transport was
locally nonlinear. It improved third-digit transfer to 27/45 but produced only
4/45 exact results, below the 6/45 linear bridge, while later norms exceeded
100%. Retrieval was rejected. The compression bottleneck now points to the
60-example fit set's coverage rather than another small-model variant. See the
[local-transport summary](PHASE1I_LOCAL_TRANSPORT_SUMMARY.md).

Phase 2 therefore freezes a new 720-example corpus: 450 fit, 90 selection, 90
development, and 90 sealed audit examples. It is balanced across leading
digits, uses a distinct template family per split, and excludes every
capability-gate and Phase 1 operand pair. The next bridge is a bottlenecked
nonlinear adapter trained from multiple native transports per fit recipient.
See the [Phase 2 scaled-adapter protocol](protocols/PHASE2_SCALED_ADAPTER.md).

The first scaled adapter used 1,800 targeted transports plus 450 identity rows
per position and a three-seed nonlinear ensemble. On development it reached
68/90 leading digits and 12/90 exact target results with all intervention norms
below one residual norm. It nevertheless failed exactness, later-position,
control-advantage, and preservation gates, so the 90-example audit remains
sealed. The evidence now favors training against causal downstream token loss
rather than imitating full donor states. See the
[Phase 2 development summary](PHASE2_SCALED_ADAPTER_SUMMARY.md).

Direct causal fine-tuning through the frozen downstream model then improved
leading-digit transfer to 84/90, exact targets to 17/90, and identity
preservation to 72/90 while keeping norms at 36%, 23%, and 30%. Later-position
accuracy remained 36/90 and 31/90, so four advancement gates still failed.
Selection behavior localizes the next problem to template-conditioned suffix
geometry; the audit remains sealed. See the
[causal-adapter summary](PHASE2_CAUSAL_ADAPTER_SUMMARY.md).

Fit-only training across four paraphrase families then improved suffix transfer
to 38/90 and 34/90, exact targets to 18/90, and identity preservation to 80/90.
This confirms template diversity helps, but does not close the gap. The next
constraint to remove is the frozen donor-PCA output basis, which may exclude
low-variance causal suffix directions. See the
[multitemplate causal summary](PHASE2_MULTITEMPLATE_CAUSAL_SUMMARY.md).

Jointly learning that output basis under causal loss improved suffix transfer
to 44/90 and 39/90, exact targets to 21/90, and identity preservation to the
90% gate, while keeping intervention norms below 36%. This is the strongest
compressed writer so far, but it still cannot advance to audit. The next bias
to remove is donor target selection, which deliberately minimized suffix
differences and leaves causal suffix labels insufficiently counterfactual. See
the [learned-basis summary](PHASE2_LEARNED_BASIS_SUMMARY.md).

Replacing matched-donor labels with exactly balanced synthetic targets, where
every answer digit must change, exposed the remaining bottleneck. The learned
writer controlled leading digits on 87/90 examples and ones digits on 56/90,
but tens digits on only 20/90. It retained the original tens digit on 51/90
examples while using 24% of residual norm, versus roughly 70% for the prior
native-donor upper bound. This motivates a frozen strength sweep before any
larger architecture change. See the
[balanced-counterfactual summary](PHASE2_BALANCED_COUNTERFACTUAL_SUMMARY.md).

A fixed-weight, norm-bounded scale sweep then doubled the tens intervention
amplitude. Tens accuracy improved only from 20/90 to 25/90 before declining at
larger scales, while identity preservation fell from 82/90 to 53/90. This
rules out insufficient amplitude as the sole bottleneck and moves the work to
a tens-specific representation and residual-boundary study. See the
[scale-sweep summary](PHASE2_ADAPTER_SCALE_SWEEP_SUMMARY.md).

A tens-only native boundary map then found a sharp causal transition: targeted
fit-donor states rose from 8/90 tens digits at hidden index 21 to 86/90 at
index 23. The selected index 27 reached 90/90 on development, versus 10/90 for
the strongest norm-matched control. Because the compressed adapter's existing
index 23 is already sufficient for native control, its bottleneck is now
localized to residual representation or coefficient prediction rather than
the write boundary itself. See the
[tens native-boundary summary](PHASE2_TENS_NATIVE_BOUNDARY_SUMMARY.md).

Projecting exact donor transports into a fit-only PCA basis then showed that
only 16 dimensions are required at index 27: rank 16 retained 89/90 selection
and 89/90 development tens digits, while rank 32 reached 90/90 selection.
Matched controls remained at or below 11/90. This separates the remaining
problem from output dimensionality: the next writer must predict the correct
16 coefficients without a donor. See the
[tens delta-rank summary](PHASE2_TENS_DELTA_RANK_SUMMARY.md).

A donor-free coordinate prototype then closed that gap at the tens position.
Replacing only the 16 causal coordinates with a fit-derived tens-digit
prototype achieved 90/90 target digits and 90/90 hard-gated identity digits on
development, versus at most 14/90 for norm-matched controls. It requires no
donor execution, model-weight update, or neural coefficient predictor at
inference. This is the first deterministic native-coordinate implant in the
project. See the
[tens prototype-writer summary](PHASE2_TENS_PROTOTYPE_WRITER_SUMMARY.md).

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
- [Phase 2 scaled-adapter protocol](protocols/PHASE2_SCALED_ADAPTER.md)
- [Arithmetic capability gate](protocols/CAPABILITY_GATE.md)
- [Capability gate v1 results](CAPABILITY_GATE_V1_RESULTS.md)
- [Capability gate v2 development](CAPABILITY_GATE_V2_DEVELOPMENT.md)
- [Capability gate v2 frozen audit](CAPABILITY_GATE_V2_AUDIT.md)
- [Phase 0 lab notebook](PHASE0_LAB_NOTEBOOK.md)
- [Phase 0.1 lab notebook](PHASE01_LAB_NOTEBOOK.md)
- [Phase 1A executive summary](PHASE1A_EXECUTIVE_SUMMARY.md)
- [Phase 1A lab notebook](PHASE1A_LAB_NOTEBOOK.md)
- [Phase 1B J-lens development summary](PHASE1B_JLENS_SUMMARY.md)
- [Phase 1B probe and carry-causality summary](PHASE1B_PROBE_CAUSAL_SUMMARY.md)
- [Phase 1C native donor-write summary](PHASE1C_NATIVE_WRITE_SUMMARY.md)
- [Phase 1D stepwise native-write summary](PHASE1D_STEPWISE_NATIVE_WRITE_SUMMARY.md)
- [Phase 1E donor-free typed-writer summary](PHASE1E_TYPED_WRITER_SUMMARY.md)
- [Phase 1F paired-transport writer summary](PHASE1F_PAIRED_TRANSPORT_SUMMARY.md)
- [Phase 1G conditional-transport bridge summary](PHASE1G_CONDITIONAL_TRANSPORT_SUMMARY.md)
- [Phase 1H full-result transport summary](PHASE1H_FULL_RESULT_TRANSPORT_SUMMARY.md)
- [Phase 1I local-transport dictionary summary](PHASE1I_LOCAL_TRANSPORT_SUMMARY.md)
- [Phase 2 scaled-adapter development summary](PHASE2_SCALED_ADAPTER_SUMMARY.md)
- [Phase 2 causal-adapter development summary](PHASE2_CAUSAL_ADAPTER_SUMMARY.md)
- [Phase 2 multitemplate causal summary](PHASE2_MULTITEMPLATE_CAUSAL_SUMMARY.md)
- [Phase 2 learned-basis causal summary](PHASE2_LEARNED_BASIS_SUMMARY.md)
- [Phase 2 balanced-counterfactual summary](PHASE2_BALANCED_COUNTERFACTUAL_SUMMARY.md)
- [Phase 2 fixed-weight scale-sweep summary](PHASE2_ADAPTER_SCALE_SWEEP_SUMMARY.md)
- [Phase 2 tens native-boundary summary](PHASE2_TENS_NATIVE_BOUNDARY_SUMMARY.md)
- [Phase 2 tens native-delta rank summary](PHASE2_TENS_DELTA_RANK_SUMMARY.md)
- [Phase 2 donor-free tens prototype summary](PHASE2_TENS_PROTOTYPE_WRITER_SUMMARY.md)
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
