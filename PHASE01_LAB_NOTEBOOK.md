# Phase 0.1 Lab Notebook

## 2026-07-27 — Protocol and implementation

Phase 0.1 was created to address two confounds from Phase 0:

1. uneven leading-digit support;
2. lexical template overlap across train, development, and test.

The protocol was written before opening audit activations. The implementation
added:

- exact leading-digit balancing over classes 1–9;
- exact-pair-disjoint train/development/audit splits;
- template families held out by split;
- a development-only layer/strength selector;
- six norm-matched causal directions;
- an internal-layer eligibility cap at 80% depth;
- a separate audit command that verifies the frozen dataset hash.

## Engineering smoke

Artifacts:

- `results/phase01_engineering_smoke.json`
- `configs/phase01_engineering_smoke_frozen.json`

The smoke used two training pairs and one development/audit pair per digit.
It successfully exercised the complete development path but exposed a route
probe threshold problem caused by the intentional 1:3 positive/negative class
balance. The probe ranked examples better than its zero decision threshold
suggested.

The binary ridge probe was changed to select a balanced-accuracy threshold on
training data only. A regression test was added. The smoke configuration was
marked superseded and no audit was run from it.

## Full development

Artifact: `results/phase01_development.json`

Source commit:
`9da83cfec3504610948b8c7af426e023b55842c5`

Resolved model revision:
`7ae557604adf67be50417f59c2c2f167def9a775`

Dataset SHA-256:
`d5c64b07045b32637a4b58bb4314e2f4f8b6c3634fde71b71b9e0ee2edc6fb4d`

Development inspected hidden-state indices 5, 10, 14, 19, and 24 and strengths
0.5, 1, 2, and 4.

### Probe findings

- Leading-digit accuracy remained near chance at every layer (11.1–13.9%).
- Scalar sum R²:
  - index 5: -19.12;
  - index 10: -4.33;
  - index 14: -7.89;
  - index 19: +0.355;
  - index 24: +0.450.
- No layer recovered any exact rounded development sum.
- Route AUC reached 1.0 at index 19 but dropped near chance at indices 10 and
  14, showing strong layer and template sensitivity.

### Causal findings

The final boundary's digit-logit direction reached 36/36 at strengths 1–4.
This is expected output-adjacent control and was not eligible for internal
selection.

At internal layers:

- digit-logit directions at index 19 reached 14/36 at strength 4 from a 2/36
  base, suggesting that output directions can be amplified through remaining
  blocks;
- same-digit donor directions at index 10 reached 8/36 at strength 4;
- the targeted probe direction did not improve top-1 at the selected index.

The predeclared selector chose index 10, strength 4:

- targeted-control margin advantage: +0.040;
- base top-1: 2/36;
- targeted top-1: 2/36;
- mean targeted delta norm: 0.952.

The configuration was committed before audit:
`c79c6de02e97c8181bb48adaa8b7816d6a0429e1`.

## Frozen audit

Artifact: `results/phase01_audit.json`

Frozen configuration:
`configs/phase01_frozen.json`

Frozen configuration content SHA-256 recorded by the evaluator:
`180c672810c6e12b95615daf8c54c9a95f3835722963e854e5d45d9f4976d53d`

Audit used 36 additions and 108 matched negatives in the unseen compact
template family. The base model already produced the correct first digit on
32/36 additions, unlike the much harder development word problems.

### Probe audit

- Route accuracy: 64.6%;
- route balanced accuracy: 76.4%;
- route AUC: 1.0;
- leading-digit accuracy: 13.9%;
- scalar sum R²: -0.314;
- scalar MAE: 274.63;
- exact rounded sums: 0/36.

The route score ordering transferred, but the train/development threshold did
not. This is a calibration failure under template shift.

### Causal audit

- base: 32/36;
- targeted digit probe: 33/36, margin delta -0.132;
- wrong-digit probe: 32/36, margin delta -0.184;
- random: 32/36, margin delta -0.010;
- scalar sum: 32/36, margin delta +0.056;
- same-digit donor: 33/36, margin delta +0.063;
- digit-logit: 32/36, margin delta -0.016.

Primary targeted-control advantage:

`-0.132 - max(-0.184, -0.010) = -0.123`

The primary condition failed. The donor result remains exploratory.

## Decision

Do not iterate on audit-specific thresholds, layers, or strengths. Do not
describe the one additional targeted top-1 as a pass.

The next development dataset must be new. Priority order:

1. integrate released NLA and J-lens outputs through a common artifact schema;
2. construct donor-derived low-rank result subspaces rather than single linear
   class directions;
3. separate route score calibration from route representation;
4. add exact per-digit read contracts before attempting multi-token write;
5. retain output-adjacent logit steering only as an upper-bound baseline.

