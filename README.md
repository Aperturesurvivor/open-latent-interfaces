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

**A first deterministic native-coordinate interface has passed a sealed
one-shot audit; cross-family replication is underway.** The repository
currently contains:

- deterministic addition/contrast dataset generation with split-disjoint
  operand pairs;
- model-agnostic Hugging Face residual-stream capture;
- ridge route and scalar-value probes;
- temporary residual interventions at an explicit hidden-state boundary;
- a probe-defined, oracle-result write-path diagnostic;
- optional, dependency-light adapters for the released NLA and Jacobian-lens
  systems;
- a donor-free rank-16 digit-write interface with fit-derived coordinate
  prototypes;
- a model-agnostic interface manifest, runtime API, and validation schema;
- a provenance-rich interpretability artifact schema that requires independent
  corroboration before an explanation can be marked as evidence;
- tests, a research ladder, a protocol, and a claims boundary.

No NLA has been trained here. One narrow answer-channel graft has passed an
end-to-end audit on one model and task; this is not yet a model-general
reasoning implant. Probe decodability is not treated as causal use.

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

Composing that implant with the existing donor-free leading and ones writers
then produced 49/90 exact balanced counterfactual results in closed loop.
Leading and tens digits reached 87/90 and 90/90; identity preservation reached
90/90; exact matched controls reached at most 5/90. Exactness, control,
preservation, norm, and parse gates all passed. Only the existing ones writer,
at 51/90, remains below the per-position gate. See the
[closed-loop hybrid summary](PHASE2_CLOSED_LOOP_HYBRID_SUMMARY.md).

The same rank-16 basis then transferred unchanged to the ones position.
Fit-derived ones prototypes achieved 90/90 target and 90/90 identity digits on
development, versus at most 14/90 for matched controls. Only the ten prototype
vectors changed; the causal basis, boundary, and deterministic controller were
shared. This identifies a reusable late-layer next-digit interface spanning
multiple autoregressive positions. See the
[ones transfer summary](PHASE2_ONES_PROTOTYPE_TRANSFER_SUMMARY.md).

Substituting both suffix dictionaries into closed-loop generation produced
87/90 exact balanced counterfactual results. Leading, tens, and ones control
reached 87/90, 90/90, and 90/90; identity reached 90/90; the strongest exact
matched control reached 10/90. Every frozen development gate passed. The
project then froze and executed its first one-shot audit package. On the held-
out audit prompt family, exact control reached 88/90, both suffix positions
reached 90/90, identity reached 90/90, and the strongest exact control reached
12/90. Every frozen audit gate passed. See the
[one-shot audit summary](PHASE2_DUAL_PROTOTYPE_AUDIT_SUMMARY.md).

For cross-family replication, Phi-3.5-mini-instruct then passed a frozen
behavioral screen on 179/180 native-chat development conditions. All five
addition regimes cleared the precommitted aggregate and worst-template gates.
This establishes only that Phi is a valid experimental target; its latent
interface has not yet been mapped. See the
[Phi-3.5 capability summary](CROSS_MODEL_PHI35_CAPABILITY_SUMMARY.md).

On a new 720-example corpus excluding every earlier operand pair, Phi also
passed the model-specific `Answer=` prefill gate: 428/450 fit, 82/90 selection,
and 85/90 development examples were exact, with three aligned digit tokens on
all 630 evaluated rows. The audit split remains sealed. See the
[Phase 3 prefill summary](PHASE3_PHI_PREFILL_BEHAVIOR_SUMMARY.md).

A selection-only native-boundary map then found complete counterfactual
next-digit control in Phi: 90/90 at hidden index 24 for the leading digit and
90/90 at index 30 for both suffix positions. Strongest matched controls reached
9/90, 9/90, and 11/90. This is a full-donor causal upper bound; rank compression
and donor removal remain open. See the
[Phase 3 native-boundary summary](PHASE3_PHI_NATIVE_BOUNDARY_SUMMARY.md).

Fit-only SVD then compressed Phi's native transports to rank 8 for the leading
position and a shared rank-32 suffix basis. The suffix basis was learned only
from tens transports yet reached 90/90 target digits at the ones position,
versus at most 12/90 for matched controls. This retains donor-dependent
coefficients and is not yet a donor-free controller. See the
[Phase 3 causal-rank summary](PHASE3_PHI_DELTA_RANK_SUMMARY.md).

Donor-free class prototypes then produced a partial pass. The shared rank-32
suffix interface reached 90/90 target and 90/90 identity digits at both tens
and ones, but the rank-8 leading prototype reached only 54/90 target digits.
The suffix configuration is locked while a bounded leading-rank follow-up
tests whether class averaging needs more coordinates than donor transport. See
the [Phase 3 prototype summary](PHASE3_PHI_PROTOTYPE_SELECTION_SUMMARY.md).

The bounded leading-rank follow-up selected rank 32 and scale 1.0, reaching
75/90 target and 89/90 identity leading digits with a 100% digit-token rate.
An exact-count correction showed that rank 16 reached the 63/90 accuracy floor
but emitted one non-digit token, so rank 32 remained the smallest complete
pass. See the
[leading-rank summary](PHASE3_PHI_LEADING_PROTOTYPE_RANK_SUMMARY.md).

The resulting three-position controller then passed every closed-loop
development gate on its first run: 73/90 exact counterfactual results, position
accuracies of 73/90, 89/90, and 90/90, identity preservation of 89/90, and at
most 1/90 exact for any norm-matched control. The audit remains sealed pending
a committed one-shot package. See the
[Phase 3 closed-loop development summary](PHASE3_PHI_CLOSED_LOOP_DEVELOPMENT_SUMMARY.md).

The sealed Phi audit then passed every gate on its single authorized run:
70/90 exact counterfactual results, position accuracies of 70/90, 90/90, and
90/90, identity preservation of 89/90, and 0/90 exact for every matched
control. This independently reproduces the donor-free native-coordinate
workflow in a second model family, while using different boundaries and ranks.
See the [Phase 3 audit summary](PHASE3_PHI_AUDIT_SUMMARY.md).

Phase 4 now begins internal carry cartography with matched +1 arithmetic
quartets. Its first behavior gate was a documented non-pass on fit
(641/720 rows and 127/180 complete quartets), while unfiltered selection and
development passed. The corpus was retained; fit-only estimation may use the
127 fully correct quartets, but selection and development cannot be filtered.
See the [Phase 4 behavior summary](PHASE4_CARRY_BEHAVIOR_SUMMARY.md).

The first causal carry map then rejected a single-vector prompt bottleneck:
full carry-pair, difference-in-differences, matched +1, shuffled, and random
interventions at the final `Answer=` token all reached at most 2/45 target tens
digits. Carry must therefore be traced across prompt-token states and their
cached downstream effects rather than assumed to reside in one terminal prompt
vector. See the
[Phase 4 prompt-boundary summary](PHASE4_CARRY_BOUNDARY_SUMMARY.md).

Full-prompt transport then revealed a strong generic operand-update route:
both the carry-pair and matched no-carry `+1` sequence deltas produced up to
39/45 correct target results, while the random control remained at 1/45. The
carry difference-in-differences residual reached 29/45 target tens digits but
did not outperform the matched `+1` control. This is causal evidence for an
early operand-update interface, not yet for an isolated carry coordinate. See
the [Phase 4 sequence-boundary summary](PHASE4_CARRY_SEQUENCE_BOUNDARY_SUMMARY.md).

Token-region localization then separated two causal interfaces. A single
changed first-operand digit at hidden-state index 1 produced 39/45 exact
carried answers versus 0/45 for its random control. A single contextualized
second-operand ones digit at index 13 produced 32/45 target tens digits versus
17/45 for the matched no-carry update and 1/45 random. This is a
donor-dependent localization of carry-specific computation; compact
donor-free estimation is the next gate. See the
[Phase 4 token-region summary](PHASE4_CARRY_TOKEN_REGION_SUMMARY.md).

Fit-only donor-free prototypes then yielded a 41/45 exact operand-edit writer.
The carry prototype reached 37/45 exact and beat matched no-carry and random
controls, but a wrong source-digit class worked equally well. The
class-specific carry gate therefore failed while motivating a simpler
class-invariant carry direction. See the
[Phase 4 donor-free prototype summary](PHASE4_DONOR_FREE_PROTOTYPE_SUMMARY.md).

A single class-invariant carry vector then reached 30/45 exact at scale 1.0
against 16/45 for matched no-carry and 1/45 random. Although that scale met
every gate, the original scorer selected a higher-accuracy scale whose control
advantage failed; the original run is therefore preserved as a non-pass
pending a bounded no-rerun rule correction. See the
[Phase 4 universal carry summary](PHASE4_UNIVERSAL_CARRY_SUMMARY.md).

The bounded correction then found that scale 1.0—and only scale 1.0—passed all
original gates. No model inference or new candidate was introduced. That
single universal direction is therefore fixed for untouched development.

On untouched development, the universal carry direction passed at 29/45 exact
against 16/45 matched no-carry and 4/45 random. The operand writer reached
41/45 exact against 0/45 wrong-class exact, but its precommitted tens-only
margin missed by one quartet because wrong digits often preserve the target
tens character. The original combined result is preserved as a non-pass
pending a bounded metric correction. See the
[Phase 4 donor-free development summary](PHASE4_DONOR_FREE_DEVELOPMENT_SUMMARY.md).

The no-rerun correction then applied exact-result discrimination to the
operand interface: 41/45 target exact versus 4/45 strongest control. Together
with the unchanged carry pass, the corrected development package clears every
gate and authorizes a separately frozen one-shot audit.

The sealed one-shot audit passed both interfaces. The operand writer reached
37/45 exact against 0/45 wrong-class and 2/45 random exact. The universal
rank-one carry direction reached 26/45 target tens digits against 11/45
matched no-carry and 2/45 random. Every output parsed, and a repeat audit
invocation was refused. See the
[Phase 4 donor-free audit summary](PHASE4_DONOR_FREE_AUDIT_SUMMARY.md).

Those audited coordinates are now packaged behind a typed manifest and Python
API. The 74 KB artifact contains four source-digit operand vectors and one
universal rank-one carry vector, with model revision, semantic token selectors,
scales, and evidence hashes declared separately. See the
[arithmetic-coordinate interface](docs/ARITHMETIC_COORDINATE_INTERFACE.md) and
[audited manifest](manifests/phi35-mini-arithmetic-coordinates-v1.json). The
verified tensor package is published in the
[`phase4-phi-arithmetic-coordinate-audit-v1` release](https://github.com/Aperturesurvivor/open-latent-interfaces/releases/tag/phase4-phi-arithmetic-coordinate-audit-v1).

Phase 5 then began an independent workflow replication in Qwen2.5-1.5B.
Untouched behavior passed on every non-audit split, including 158/180
complete-correct fit quartets, 39/45 selection quartets, and 41/45 development
quartets. No Phi coordinate is transferred. See the
[Phase 5 Qwen behavior summary](PHASE5_QWEN_CARRY_BEHAVIOR_SUMMARY.md).

The Qwen token-region scan then rediscovered an early operand edit at
hidden-state index 12: 43/45 exact versus 0/45 random. A later carry-context
effect appeared at index 16 but remained a unit-scale non-pass at 11/45 target
tens versus 2/45 matched no-carry. See the
[Phase 5 Qwen token-region summary](PHASE5_QWEN_TOKEN_REGION_SUMMARY.md).

A bounded scale-only follow-up then passed the fixed Qwen carry context at
scale 2.0: 26/45 target tens versus 7/45 matched no-carry and 0/45 random.
Scale 2.0 was the smallest passing candidate. See the
[Phase 5 Qwen carry-scale summary](PHASE5_QWEN_CARRY_SCALE_SUMMARY.md).

Qwen-only donor-free fitting then produced a 40/45 exact operand writer at
scale 1.0. Its digit-conditioned carry writer failed because wrong source-digit
classes transferred equally well, independently favoring a universal carry
direction. See the
[Phase 5 Qwen donor-free prototype summary](PHASE5_QWEN_DONOR_FREE_PROTOTYPE_SUMMARY.md).

The first universal-Qwen scale grid did not pass: scale 1.5 was specific but
three quartets below the absolute gate, while scale 2.0 was accurate but moved
the matched no-carry control too strongly. A single interpolation-only
follow-up is permitted before rejecting the universal-vector hypothesis. See
the [Phase 5 Qwen universal carry summary](PHASE5_QWEN_UNIVERSAL_CARRY_SUMMARY.md).

The one authorized interpolation then passed at scale 1.6: 24/45 target tens
versus 10/45 matched no-carry and 0/45 random. The universal tensor hash was
unchanged, and scale 1.6 is fixed for untouched Qwen development.

Qwen then passed one-shot untouched development without correction. The
operand writer reached 43/45 exact versus 1/45 wrong-class; the universal carry
writer reached 34/45 target tens versus 22/45 matched no-carry and 0/45
random. See the
[Phase 5 Qwen donor-free development summary](PHASE5_QWEN_DONOR_FREE_DEVELOPMENT_SUMMARY.md).

The sealed Qwen audit then passed operand editing at 43/45 exact versus 0/45
wrong-class, but the universal carry vector failed specificity: 31/45 target
tens versus 21/45 matched no-carry, a 22.22-point margin below the fixed
25-point gate. The audit cannot be reused for tuning. See the
[Phase 5 Qwen audit summary](PHASE5_QWEN_DONOR_FREE_AUDIT_SUMMARY.md).

The independently passing Qwen operand coordinate is packaged on its own.
Fresh pair-disjoint recovery work then rejected two further Qwen carry
hypotheses: recipient-conditioned full transport failed selection specificity,
and a matched difference-in-differences carry interaction failed one-shot
development. Carry audit remained sealed. See the
[Phase 6 conditional-carry summary](PHASE6_QWEN_CONDITIONAL_CARRY_SELECTION_SUMMARY.md)
and [Phase 6B interaction summary](PHASE6B_QWEN_CARRY_INTERACTION_SUMMARY.md).

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
- [Phase 2 closed-loop hybrid summary](PHASE2_CLOSED_LOOP_HYBRID_SUMMARY.md)
- [Phase 2 cross-position ones prototype summary](PHASE2_ONES_PROTOTYPE_TRANSFER_SUMMARY.md)
- [Phase 2 closed-loop dual-prototype summary](PHASE2_CLOSED_LOOP_DUAL_PROTOTYPE_SUMMARY.md)
- [Phase 2 one-shot audit summary](PHASE2_DUAL_PROTOTYPE_AUDIT_SUMMARY.md)
- [Phi-3.5 cross-model capability summary](CROSS_MODEL_PHI35_CAPABILITY_SUMMARY.md)
- [Phase 3 Phi prefill behavior summary](PHASE3_PHI_PREFILL_BEHAVIOR_SUMMARY.md)
- [Phase 3 Phi native-boundary summary](PHASE3_PHI_NATIVE_BOUNDARY_SUMMARY.md)
- [Phase 3 Phi causal-rank summary](PHASE3_PHI_DELTA_RANK_SUMMARY.md)
- [Phase 3 Phi prototype-selection summary](PHASE3_PHI_PROTOTYPE_SELECTION_SUMMARY.md)
- [Phase 3 Phi leading-prototype rank summary](PHASE3_PHI_LEADING_PROTOTYPE_RANK_SUMMARY.md)
- [Phase 3 Phi closed-loop development summary](PHASE3_PHI_CLOSED_LOOP_DEVELOPMENT_SUMMARY.md)
- [Phase 3 Phi one-shot audit summary](PHASE3_PHI_AUDIT_SUMMARY.md)
- [Phase 4 carry-quartet behavior summary](PHASE4_CARRY_BEHAVIOR_SUMMARY.md)
- [Phase 4 carry prompt-boundary summary](PHASE4_CARRY_BOUNDARY_SUMMARY.md)
- [Phase 4 carry sequence-boundary summary](PHASE4_CARRY_SEQUENCE_BOUNDARY_SUMMARY.md)
- [Phase 4 carry token-region summary](PHASE4_CARRY_TOKEN_REGION_SUMMARY.md)
- [Phase 4 donor-free prototype summary](PHASE4_DONOR_FREE_PROTOTYPE_SUMMARY.md)
- [Phase 4 universal carry summary](PHASE4_UNIVERSAL_CARRY_SUMMARY.md)
- [Phase 4 donor-free development summary](PHASE4_DONOR_FREE_DEVELOPMENT_SUMMARY.md)
- [Phase 4 donor-free audit summary](PHASE4_DONOR_FREE_AUDIT_SUMMARY.md)
- [Native-coordinate interface API and manifest](docs/NATIVE_COORDINATE_INTERFACE.md)
- [Arithmetic-coordinate interface and deterministic graft](docs/ARITHMETIC_COORDINATE_INTERFACE.md)
- [Audited Phi arithmetic-coordinate manifest](manifests/phi35-mini-arithmetic-coordinates-v1.json)
- [Arithmetic-coordinate interface schema](schemas/arithmetic-coordinate-interface-v1.schema.json)
- [Phase 5 Qwen arithmetic-state replication protocol](protocols/PHASE5_QWEN_ARITHMETIC_STATE_REPLICATION.md)
- [Phase 5 Qwen carry-behavior summary](PHASE5_QWEN_CARRY_BEHAVIOR_SUMMARY.md)
- [Phase 5 Qwen token-region summary](PHASE5_QWEN_TOKEN_REGION_SUMMARY.md)
- [Phase 5 Qwen carry-context scale summary](PHASE5_QWEN_CARRY_SCALE_SUMMARY.md)
- [Phase 5 Qwen donor-free prototype summary](PHASE5_QWEN_DONOR_FREE_PROTOTYPE_SUMMARY.md)
- [Phase 5 Qwen universal carry summary](PHASE5_QWEN_UNIVERSAL_CARRY_SUMMARY.md)
- [Phase 5 Qwen donor-free development summary](PHASE5_QWEN_DONOR_FREE_DEVELOPMENT_SUMMARY.md)
- [Phase 5 Qwen donor-free audit summary](PHASE5_QWEN_DONOR_FREE_AUDIT_SUMMARY.md)
- [Phase 6 Qwen conditional-carry protocol](protocols/PHASE6_QWEN_CONDITIONAL_CARRY.md)
- [Phase 6 Qwen conditional-carry selection summary](PHASE6_QWEN_CONDITIONAL_CARRY_SELECTION_SUMMARY.md)
- [Phase 6B Qwen carry-interaction protocol](protocols/PHASE6B_QWEN_CARRY_INTERACTION.md)
- [Phase 6B Qwen carry-interaction summary](PHASE6B_QWEN_CARRY_INTERACTION_SUMMARY.md)
- [Audited Qwen operand-coordinate manifest](manifests/qwen25-15b-operand-coordinate-v1.json)
- [Audited Phi native-coordinate manifest](manifests/phi35-mini-next-digit-interface-v1.json)
- [Native-coordinate interface v2 schema](schemas/native-coordinate-interface-v2.schema.json)
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
