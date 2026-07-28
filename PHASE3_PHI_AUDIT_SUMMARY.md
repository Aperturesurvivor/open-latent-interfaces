# Phase 3 Phi One-Shot Audit Summary

## Outcome

The donor-free native-coordinate controller passed every frozen one-shot audit
gate on `microsoft/Phi-3.5-mini-instruct`:

| Metric | Audit result | Required | Gate |
| --- | ---: | ---: | --- |
| Exact counterfactual result | 70/90 (77.78%) | at least 45/90 | pass |
| Leading digit | 70/90 | at least 63/90 | pass |
| Tens digit | 90/90 | at least 63/90 | pass |
| Ones digit | 90/90 | at least 63/90 | pass |
| Identity preservation | 89/90 (98.89%) | at least 81/90 | pass |
| Strongest exact control | 0/90 | target advantage at least 23 | pass |
| Parseable target output | 90/90 | 90/90 | pass |
| Decimal target tokens | 270/270 | 270/270 | pass |

Mean targeted intervention norms were 45.61%, 69.01%, and 61.05% of the
recipient residual norm. Every matched control produced 0/90 exact target
results.

## Controller

- leading: rank 32, hidden index 24, scale 1.0;
- tens: rank-32 suffix basis, hidden index 30, scale 1.25;
- ones: the same rank-32 suffix basis, hidden index 30, scale 1.25;
- coordinate values: fit-derived digit class means;
- norm cap: one recipient residual norm;
- preservation: exact zero-delta hard gate;
- inference-time donors: none;
- model-weight updates: none;
- neural coefficient predictor: none.

The suffix basis was fitted only from tens-position transports and reused
unchanged at the ones position.

## Controls and baseline

- untouched base: 0/90 target, 85/90 original;
- wrong-digit norm-matched: 0/90 target;
- shuffled-target norm-matched: 0/90 target;
- random-in-subspace norm-matched: 0/90 target;
- hard-gated identity: 89/90 original.

The targeted controller's exact-result advantage over every control was 70/90
(77.78 percentage points).

## Audit integrity

- frozen audit target commit: `9e51288`
- configuration SHA-256:
  `78f3b35e4a8c7d3d2e9ffef76ddab674999194d1bc3bcb8a22ee883632177a7d`
- audit runner SHA-256:
  `78d838a200da53c60213b39a1ce5d771fa2422efe81695d1f581a9db1bba66f3`
- shared evaluation engine SHA-256:
  `b2493842ad2201789e0f14f568b209ff05a93373b097a13a9bbfa2c397d95dff`
- audit target SHA-256:
  `c2a7b3919c551cdcc3e71b69cf83e2514d58643b0d256325128365163c745459`
- result SHA-256:
  `36f02acbe6203df608d88401a819f989786ce658b63e6119e9e1adaac58d61ea`
- elapsed audit time: 454.46 seconds

The runner refused a second invocation because the frozen result path already
existed.

## What this establishes

This is a cross-family replication of the discovery and implementation
workflow:

1. behaviorally qualify a frozen model;
2. freeze a leakage-free corpus and token contract;
3. causally map native write boundaries;
4. measure causal transport rank;
5. replace donor coefficients with fit-derived coordinate prototypes;
6. compose the interface in closed loop;
7. pass a sealed one-shot audit.

Qwen and Phi required different boundaries, ranks, scales, token contracts, and
prototype artifacts. Therefore the evidence supports workflow portability,
not direct vector portability.

## Claim boundary

The controller writes a requested three-digit answer supplied externally. It
does not compute addition, reveal a ground-truth thought transcript, or prove
control of Phi's internal arithmetic algorithm. The demonstrated interface is
a late answer-channel mechanism on two frozen model families and one task
class. Internal operand, carry, and computation-state mapping remains the next
research stage.
