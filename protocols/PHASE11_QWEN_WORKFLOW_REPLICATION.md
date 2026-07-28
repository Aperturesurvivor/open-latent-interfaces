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
