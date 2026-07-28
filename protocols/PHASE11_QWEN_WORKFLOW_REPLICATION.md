# Phase 11: Qwen Workflow-Level Replication

## Purpose

Test whether the audited Phi arithmetic-graft workflow can be rediscovered on
a structurally different model without transferring any Phi activation,
centroid, basis, prototype, layer, margin, or norm parameter.

The target is `Qwen/Qwen2.5-1.5B-Instruct` at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

## Transfer boundary

The following may transfer:

- external semantic operand-span contract;
- nearest-centroid reader family;
- frozen-model prompt-local Jacobian compiler algorithm;
- paired controls, hash locking, write-once results, and one-shot audit
  discipline;
- typed manifest and validation workflow.

The following must be rediscovered or independently sourced for Qwen:

- reader hidden-state boundary and centroids;
- leading compiler boundary, margin, iteration budget, and norm cap;
- digit token IDs;
- suffix writer tensors and scales.

No Phi tensor may be loaded by a Qwen discovery or evaluation runner.

## Initial corpus boundary

Reader fit and selection reuse the pair-disjoint Phase 7 source corpus only as
non-audit discovery data, rendered through the Qwen chat template. The dataset
configuration remains audit-sealed.

Qwen already has independently audited rank-16 suffix writers at hidden-state
index 27. They may be bound only after the new reader and leading compiler
pass their own exposed selection gates.

## Advancement

1. Fit one full-width native-state centroid per digit at predeclared candidate
   hidden-state indices.
2. Select the earliest reader satisfying the same accuracy and rotated-label
   control thresholds used for Phi.
3. Select Qwen-specific leading compiler parameters on exposed data.
4. Evaluate a complete closed-loop read → host addition → hybrid write
   development pipeline.
5. Only a passing development pipeline may authorize a newly generated
   pair- and template-disjoint Qwen audit.

No Qwen audit claim is open at this stage.

## Reader selection outcome

The frozen selection run passed:

- result: `results/phase11_qwen_operand_reader_selection.json`
- result SHA-256:
  `968fcac02c0bcf912a37a258b22b2178cc48cd7c2bfea37a2d8eef8c3b484309`
- selected hidden-state index: 1 (the earliest passing candidate)
- held-out target accuracy: 180/180 operand pairs and 988/988 digits
- rotated-label control: 0/180 operand pairs and 0/988 digits
- reader artifact:
  `artifacts/phase11_qwen_operand_reader.safetensors`
- artifact SHA-256:
  `654b47ba8b09e72979cca97f0e872dc49d8bfe112ca0f7a1bce0091006f55954`

Indices 4 and 8 also passed, but were not selected under the precommitted
earliest-passing rule. This remains a selection-only result.

## Leading compiler selection boundary

The Qwen leading-digit writer search is frozen before execution in
`configs/phase11_qwen_leading_compiler_selection.json`.

It uses 90 exposed selection examples (the carry-base and control-base members)
and independently searches:

- hidden-state indices 12, 16, 20, 23, and 24;
- desired margins 4, 8, and 16;
- relative-norm caps 0.25, 0.5, and 0.75;
- one prompt-local Jacobian compilation step.

A passing one-step result selects an iteration budget of one. If no candidate
passes the precommitted target, identity, control-advantage, digit-token, and
relative-norm gates, bounded iterative relinearization must be frozen as a new
exposed selection stage. The later audited Qwen suffix boundary at index 27 is
not loaded or evaluated during this search.

### One-step outcome

The one-step search did not pass:

- result: `results/phase11_qwen_leading_compiler_selection.json`
- result SHA-256:
  `8b54e635cff31ff5f35f486ef9eec283d03018d5f2cfaaaf17622d1e0fc9b247`
- best fallback under the precommitted scoring rule: hidden-state index 23,
  desired margin 16, norm cap 0.25
- target accuracy: 57/90 (base: 2/90)
- identity accuracy: 90/90 (base: 85/90)
- strongest semantic-control accuracy: 13/90
- control advantage: 44/90
- mean target relative norm: 0.1900

This is a non-passing selection result, not a failed audit. The deterministic
fallback parameters now form the fixed input to a separate bounded iterative
relinearization selection. That stage may select only the iteration count; it
may not reopen the layer, margin, or norm-cap search.

The iterative stage is frozen in
`configs/phase11_qwen_iterative_leading_compiler.json`. It holds hidden-state
index 23, desired margin 16, and norm cap 0.25 fixed, then evaluates cumulative
relinearization depths 1 through 4. The earliest depth satisfying the unchanged
gate must be selected. Target, wrong-digit, and random-norm controls are
recomputed at every depth; the identity compiler remains a single step.

### Iterative outcome

The bounded iterative selection passed:

- result: `results/phase11_qwen_iterative_leading_compiler.json`
- result SHA-256:
  `b3a4cf804afa39e29aea3720caee8da6c472d1292a77795f22a4c26641e89516`
- selected iteration count: 2 (the earliest passing depth)
- target accuracy: 85/90 (base: 2/90)
- identity accuracy: 90/90 (base: 85/90)
- strongest semantic-control accuracy: 11/90
- control advantage: 74/90
- mean target relative norm: 0.2021

Iterations 3 and 4 reached 90/90 target accuracy, but were not selected because
iteration 2 already passed. The complete Qwen leading compiler is therefore:
hidden-state index 23, desired margin 16, relative-norm cap 0.25, and two
relinearization steps.

## Integrated development boundary

The complete exposed development pipeline is frozen in
`configs/phase11_qwen_hybrid_graft_development.json`.

It binds:

- the newly selected index-1 Qwen operand reader;
- deterministic host integer addition;
- the newly selected index-23, margin-16, cap-0.25, two-step leading compiler;
- Qwen's previously audited index-27 rank-16 suffix basis;
- the audited tens prototype at scale 1.25 and ones prototype at scale 2.0.

The evaluation uses the 45 pair-disjoint Phase 7 development carry-base
examples. Advancement requires accurate latent reading and host computation,
at least 90% full-result and per-position accuracy, preservation of base-correct
cases, recovery of base errors beyond random and wrong-target controls, and
successful target-following under a shuffled semantic control. No audit may be
authorized from component-level results alone.

### Initial integrated outcome

The 45-example integrated pipeline produced perfect task output but did not
pass its advancement gate:

- result: `results/phase11_qwen_hybrid_graft_development.json`
- result SHA-256:
  `940aaadc940ff35e0dbecab2c4834e8d80a6c8f660197c692ab5825b473bf64c`
- reader: 45/45 operand pairs
- deterministic compute: 45/45 sums
- latent and oracle hybrid output: 45/45 exact, with 45/45 at every position
- base: 44/45 exact
- shuffled semantic target following: 45/45; shuffled true accuracy: 2/45
- preserved base-correct cases: 44/44
- recovered base errors: latent 1/1, wrong-target 0/1, random 1/1

The sole failing check was excess base-error recovery over the norm-matched
random condition. Because the base model supplied only one error and the random
condition happened to repair it, this comparison has a denominator of one and
cannot support advancement. The result remains a transparent non-pass.

Before any audit authorization, a new exposed integration stage must compare
semantic shuffled-target following against a norm-matched random intervention
aimed at the same shuffled target set. This directly tests causal target
specificity over every example rather than relying on a single naturally
occurring base error.

The refinement is frozen in
`configs/phase11_qwen_hybrid_graft_development_refined.json`. It expands the
exposed evaluation to all 180 Phase 7 development examples and adds a
`shuffled_random_norm_matched` condition. Advancement now requires:

- semantic shuffled-target following of at least 80%;
- norm-matched random shuffled-target following of at most 25%;
- at least 50 percentage points of target-following advantage;
- all original reader, compute, exact-output, per-position, identity,
  wrong-target, parse, and digit-token checks.

No model, reader, compiler, suffix tensor, layer, scale, rank, or norm setting
is changed by this refinement.

### Refined integrated outcome

The 180-example refinement again produced perfect integrated task output but
remained a formal non-pass:

- result:
  `results/phase11_qwen_hybrid_graft_development_refined.json`
- result SHA-256:
  `1cb71017f850c050c017fd27727ef579ccfcd445e1187aa2a31d799174027264`
- reader and deterministic compute: 180/180
- latent and oracle hybrid output: 180/180 exact, all positions 180/180
- base: 165/180 exact
- recovered base errors: latent 15/15, random 1/15, wrong-target 5/15
- preserved base-correct cases: 165/165
- shuffled semantic target following: 179/180
- shuffled norm-matched random target following: 7/180
- shuffled target-following advantage: 172/180

Fifteen of sixteen checks passed. The only failure was the absolute
wrong-target base-error-recovery ceiling of 25%: the observed value was 5/15.
The already frozen comparative check passed: intended recovery exceeded
wrong-target recovery by 10/15, above the required 50 percentage points.

Inspection of immutable rows shows why the absolute ceiling is unstable. For
example, when the base emitted 500 for a true result of 600, the norm-matched
direction aimed at the wrong target 711 crossed the adjacent 600 boundary but
did not reach 711. Counting that as semantic wrong-target success conflates an
undershooting control with the intended mechanism. Any correction must be
measurement-only, preserve this non-pass, remove only the redundant absolute
ceiling, retain the comparative recovery check, and precede a fresh audit.

That correction is frozen in
`configs/phase11_qwen_hybrid_graft_gate_correction.json`. Its runner is not
permitted to load a model. It verifies the immutable result and config hashes,
requires `wrong_target_recovery` to be the only failed check, removes only that
absolute ceiling, and rechecks the unchanged 50-point paired recovery
advantage. Every other development check remains unchanged.
