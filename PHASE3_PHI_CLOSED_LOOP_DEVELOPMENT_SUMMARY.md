# Phase 3 Phi Closed-Loop Development Summary

## Outcome

The complete donor-free Phi coordinate controller passed every precommitted
closed-loop development gate on its first run:

| Metric | Result | Required | Gate |
| --- | ---: | ---: | --- |
| Exact counterfactual result | 73/90 | at least 45/90 | pass |
| Leading digit | 73/90 | at least 63/90 | pass |
| Tens digit | 89/90 | at least 63/90 | pass |
| Ones digit | 90/90 | at least 63/90 | pass |
| Identity preservation | 89/90 | at least 81/90 | pass |
| Strongest exact control | 1/90 | target advantage at least 23 | pass |
| Parseable output | 90/90 | 90/90 | pass |
| Decimal digit tokens | 270/270 | 270/270 | pass |

Mean intervention norms were 46.30%, 65.88%, and 59.31% of the recipient
residual norm at the three positions, all below the one-residual-norm cap.

## Controls

The donor-free targeted controller produced 73 exact requested results.
Norm-matched controls produced:

- wrong-digit prototypes: 0/90;
- shuffled targets: 0/90;
- random directions inside the selected subspaces: 1/90.

The untouched base model produced 0/90 counterfactual targets and 85/90
original results. The hard-gated identity controller improved preservation to
89/90 while applying mean relative norms of only 1.4%, 1.0%, and 0.0%.

## Controller

- leading: rank 32, hidden index 24, scale 1.0;
- tens: shared suffix rank 32, hidden index 30, scale 1.25;
- ones: the same suffix basis, hidden index 30, scale 1.25;
- norm cap: 1.0;
- hard gate: exact zero delta when the base argmax is already the requested
  digit;
- inference-time donors: none;
- model-weight updates: none;
- neural coefficient predictor: none.

Only 30 fit-derived class-mean coordinate vectors are required beyond the two
selected bases.

## Frozen provenance

- frozen development experiment commit: `b2583a4`
- configuration SHA-256:
  `aef162b2cb42c208771ff073306ccd8846344d83634b0eea9d3ed3a956dc82c9`
- result SHA-256:
  `d093e92925fbe0a850bd37808a3ade4550875210a4fc60ffb874bb55c3a0b28c`
- development target SHA-256:
  `d805f13b9a0bbafb5ac945fd3ad2b41dddef63ab9fe820b076c800d097229c65`
- elapsed evaluation time: 348.53 seconds

## Decision and claim boundary

Freeze the complete controller, controls, count-based gates, and runner for a
single audit evaluation on the untouched Phase 3 audit prompt family.

This establishes development generalization across a new prompt family in a
second model family. It is not yet confirmatory evidence until the one-shot
audit passes, and it remains an answer-channel interface rather than a map of
Phi's internal addition algorithm.
