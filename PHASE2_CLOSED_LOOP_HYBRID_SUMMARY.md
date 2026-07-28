# Phase 2 Closed-Loop Hybrid Summary

## Outcome

**The donor-free hybrid generated 49/90 exact balanced counterfactual results
and passed every frozen gate except ones-digit transfer.**

The leading causal adapter, rank-16 tens prototype implant, and ones causal
adapter were composed autoregressively with deterministic hard gating at every
position. Exact target accuracy reached 54.4%, identity preservation reached
100%, and the exact advantage over every matched control was 48.9 percentage
points.

## Provenance

- Frozen protocol commit: `02dd671`.
- Frozen config SHA-256:
  `754044c8d34c3ce774dfabd1ecbc6635b7e9d1e0b775ce31a6de8ee487adedd0`.
- Result SHA-256:
  `40a5388028fd5fda5f2149b2b914534b20dbfc4dcc5eafdbbd9cf7bedeaebdd7`.
- New weights or prototypes fitted: none.
- Audit examples evaluated: 0/90.

## Development result

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| **Targeted hybrid** | **87/90** | **90/90** | **51/90** | **49/90** | 0/90 |
| Base | 3/90 | 1/90 | 1/90 | 1/90 | 84/90 |
| Identity, hard gated | 0/90 | 0/90 | 0/90 | 0/90 | **90/90** |
| Shuffled target | 21/90 | 14/90 | 16/90 | 1/90 | 2/90 |
| Shuffled state | 18/90 | 40/90 | 18/90 | 5/90 | 29/90 |
| Random direction | 4/90 | 1/90 | 1/90 | 1/90 | 83/90 |

All conditions remained 100% parseable.

The hybrid's mean relative intervention norms were 35.4%, 71.7%, and 35.8%.
The target hard gate fired on 3/90, 3/90, and 4/90 examples because the
unmodified model already produced the requested digit at those actual
closed-loop prefixes.

## Frozen advancement gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Exact target | >=50% | 54.4% | Pass |
| Every position | >=70% | 96.7%, 100%, 56.7% | **Fail** |
| Exact advantage over every control | >=25 points | 48.9 points | Pass |
| Identity preservation | >=90% | 100% | Pass |
| Relative norm at every position | <=100% | 35.4%, 71.7%, 35.8% | Pass |
| Parse rate | 100% | 100% | Pass |

The audit remains sealed because every criterion is conjunctive.

## Identity mechanism

The unmodified base generated the correct original result on 84/90 examples.
The hard-gated identity condition reached 90/90:

- the leading gate emitted zero on 84/90 examples;
- the existing leading adapter corrected the other six;
- tens and ones gates emitted zero on all 90 subsequent prefixes.

Identity preservation is therefore not merely a reduced side effect. It is an
explicit deterministic control path that leaves correct computation untouched
and repairs the observed base errors.

## Interpretation

The tens implant composes cleanly. It retains 90/90 control when driven by the
leading digits actually generated in closed loop, so its teacher-forced result
was not an artifact of idealized prefixes.

Forty-nine of the 51 correct ones digits occur on examples whose first two
digits are also correct. The complete-result ceiling is now almost entirely
the existing ones writer. The leading writer misses three examples; the tens
implant misses none after the generated leading prefixes.

This is the first full-result configuration to pass:

- exact target accuracy;
- causal-control advantage;
- preservation;
- norm;
- parseability.

Only the per-position suffix floor blocks audit.

## Next experiment

Replicate the successful localization-compression-prototype sequence for the
ones position:

1. teacher-force the requested leading and tens prefix;
2. map native ones control across late residual boundaries;
3. find the smallest fit-only donor-delta rank that preserves control;
4. replace donor coefficients with fit-derived ones-digit prototypes and the
   same hard gate;
5. substitute that implant into this frozen hybrid and rerun development.

No audit evaluation is authorized until the ones position reaches 70% and all
other gates remain passing.
