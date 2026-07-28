# Phase 1 Protocol: Triangulated Mathematical Cartography

## Objective

Map where and when a frozen language model represents arithmetic routing,
operand roles, operation, carry state, and exact result. Distinguish:

1. generic arithmetic/numeric context;
2. decodable task variables;
3. variables the model causally uses;
4. a typed latent interface suitable for a deterministic read/write bridge.

No single probe, lens, explanation, or steering result can pass Phase 1.

## Target tiers

### Tier A — local causal cartography

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Behavioral envelope: mixed three-digit addition, passed 48/48 frozen
  capability-audit conditions.
- Required methods: Jacobian lens, vanilla logit lens, linear/nonlinear probes,
  contrastive directions, donor patching, necessity interventions, sufficiency
  interventions, and preservation controls.

Tier A cannot make an NLA-complete triangulation claim because no released NLA
pair targets this architecture and width.

### Tier B — NLA-complete triangulation

- Initial target: Qwen2.5-7B at the released NLA extraction layer, subject to a
  fresh behavioral capability gate at an immutable model revision.
- Required methods: every Tier A method plus released AV verbalization and AR
  reconstruction on the exact same activation hashes.

A future project-trained 1.5B AV/AR pair may replace Tier B only after it passes
the open NLA-factory reproduction and held-out faithfulness protocol.
Architecture-mismatched NLA output is prohibited.

## Layer and position conventions

Hugging Face hidden-state index 0 is the embedding state; index `i + 1` is the
output of decoder block `i`. J-lens source layer `i` therefore corresponds to
HF hidden-state index `i + 1`.

Positions are recorded as absolute token indices plus semantic landmarks:

- after first operand;
- after operator/relation phrase;
- after second operand;
- final user token;
- assistant generation tokens 0, 1, and 2.

Chat-template tokens are part of the sequence and must not be silently removed.
Every artifact records the rendered-prompt hash, token ID, decoded token, and
activation hash.

## Variables are separate endpoints

| Variable | Typed target | Required contrasts |
|---|---|---|
| Route | compute / quote / list / compare | same numerals and template length |
| Operand A | hundreds, tens, ones; exact integer | operand B held or permuted |
| Operand B | hundreds, tens, ones; exact integer | operand A held or permuted |
| Operation | addition / subtraction / multiplication / no-op | operands held |
| Carry | ones carry, tens carry | result magnitude matched where possible |
| Timing | first layer/position with stable information and causal effect | earlier/later boundaries |
| Result | hundreds, tens, ones; exact integer | same leading digit, different suffix; same result, different operands |

Leading-digit decoding is a diagnostic only. It cannot substitute for exact
three-token result recovery.

## Data separation

Development and audit must be disjoint by:

- canonical unordered operand pair;
- exact rendered string;
- prompt template family;
- donor/recipient pairing;
- J-lens fitting corpus;
- probe-fitting examples.

Capability-gate operands are excluded from Phase 1. Development uses direct and
symbolic families. Audit uses separately authored paraphrase and word-problem
families. Each result digit and carry condition is balanced or inverse-weighted
with weights frozen before analysis.

Matched negative routes reuse the same operands:

- quote the expression without solving;
- list the two identifiers;
- compare the values without combining them;
- perform a different operation.

## Stage 1 — observational cartography

At every layer and semantic position:

1. fit train-only linear probes for each typed variable;
2. compare with shuffled-label, majority, and prompt-only baselines;
3. compute J-lens and vanilla logit-lens token ranks;
4. record NLA AV/AR artifacts where a compatible pair exists;
5. measure cross-template and cross-pair generalization;
6. measure whether results are stable across at least three seeds.

Natural-language explanations remain `hypothesis`. AR round-trip quality is
dependent evidence and cannot change that status.

## Stage 2 — causal necessity

For each development-selected candidate channel:

- mean-ablate or project out the candidate subspace;
- patch a matched wrong-value donor;
- patch a same-value/different-operands donor;
- patch a same-operands/different-operation donor;
- compare with norm-matched random, shuffled-label, wrong-digit, scalar, and
  final-logit directions.

A necessary result channel should selectively damage exact arithmetic while
preserving route recognition, operands not under test, and unrelated tasks.
Output-adjacent direct logit suppression is reported but cannot satisfy
necessity on its own.

## Stage 3 — causal sufficiency

Write a counterfactual typed result at an internal boundary while keeping the
prompt fixed. Require:

- the intended full multi-token answer, not only its first digit;
- correct carry-dependent suffixes;
- stronger effect than every norm-matched control;
- success before the final residual boundary;
- evidence that downstream layers propagate the change;
- no base-model weight update.

Native donor-state patches are the first sufficiency test. Learned bridges may
follow, but development and audit bridge parameters are frozen at inference.

## Stage 4 — preservation

Run unchanged and intervened models on:

- quote/copy controls with the same numerals;
- factual recall;
- short code completion;
- non-arithmetic instruction following;
- the alternate arithmetic operations.

Report exact accuracy, token-level divergence, intervention norm relative to
native-state norm, and any generation-format failure. A gain achieved by
turning every numeric prompt into addition is a failure.

## Development selection

Development may select:

- one contiguous layer band;
- one semantic position or generation step per variable;
- probe family and regularization;
- intervention strength from a fixed grid;
- one write construction.

Selection maximizes the minimum advantage over controls, not raw target score.
No layer deeper than 80% of model depth may be the sole candidate.

The selected configuration, code commit, model revision, dataset hash, J-lens
hash, probe weights hash, and all thresholds must be committed before audit.

## Frozen audit gates

### Behavioral

- untouched model: at least 90% aggregate exact accuracy;
- every primary template cell: at least 80%.

### Read interface

- every exact result digit: at least 80% held-out accuracy;
- exact full result: at least 70%;
- at least 20 percentage points above shuffled and prompt-only controls;
- stable direction of effect across all seeds and audit templates.

### Causal necessity

- candidate removal or wrong-value patch causes at least a 20-point exact
  arithmetic drop;
- same-value/different-operands patch causes less than a 5-point drop;
- targeted-minus-random effect has a paired 95% bootstrap interval above zero.

### Causal sufficiency

- counterfactual full-result exactness improves at least 20 points over base;
- targeted intervention beats every matched control by at least 10 points;
- correct-token margin improves at every generated result position;
- the effect exists at an internal boundary, not only the last block.

### Preservation

- no unrelated evaluation loses more than 2 absolute points;
- median intervention norm is no more than one native residual norm unless a
  larger preregistered bound is justified before audit.

### NLA-complete status

- AV explanation refers to the typed variable;
- AR cosine/normalized MSE passes a threshold calibrated on held-out vectors;
- at least one non-NLA method independently agrees at the exact activation;
- a causal intervention confirms the same variable.

Only then may an NLA explanation artifact be marked `corroborated`.

## Stop conditions

Phase 1 does not pass if:

- value readouts stay at chance after the behavioral gate passes;
- methods recover only generic numeric context;
- exact suffix digits appear only after they are already emitted;
- interventions work solely through final logits;
- causal effects disappear on audit paraphrases;
- preservation fails;
- NLA text conflicts with causal evidence;
- the audit is used to reselect layers, strengths, prompts, or methods.

## Claims allowed after each tier

- Tier A pass: a causally supported native arithmetic interface candidate in
  one frozen model.
- Tier B pass: the candidate is independently triangulated with a compatible
  NLA pair.
- Neither tier: a deterministic latent graft, universal interface, or
  model-independent NLA recipe. Those require later phases.
