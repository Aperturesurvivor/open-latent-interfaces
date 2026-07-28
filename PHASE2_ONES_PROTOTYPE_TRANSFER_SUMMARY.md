# Phase 2 Cross-Position Ones Prototype Summary

## Outcome

**The unchanged rank-16 tens-derived basis controlled 90/90 development ones
digits with a fit-derived ones dictionary.**

The result passes every frozen teacher-forced diagnostic gate. It uses no
donor execution, model-weight update, new causal basis, or neural coefficient
predictor. Only the ten coordinate prototypes were recomputed from ordinary fit
activations after two correct prefix digits.

## Provenance

- Frozen protocol commit: `554dc51`.
- Frozen config SHA-256:
  `867d6db9b52e44a50ba7ecf4be075674b59b6ca556f96168d3c7bf588a5addde`.
- Result SHA-256:
  `b3ab610fb9114c7b0502cceb267602b461695144e37e9318f28fb0f3d250f931`.
- Ones prototype SHA-256:
  `16064fd1e125e20ef4226b4648cb53e314ab3ee207fd84ee9850662025220db7`.
- Reused basis SHA-256:
  `8c9b0e53a3b8216e724ab37519a51723c0e0697d956ade56dceb362217b3084d`.
- Audit examples evaluated: 0/90.

## Selection

| Scale | Target ones | Identity ones | Target margin | Target norm |
|---:|---:|---:|---:|---:|
| 0.50 | 45/90 | 90/90 | +0.51 | 26.5% |
| 0.75 | 87/90 | 90/90 | +4.08 | 39.8% |
| 1.00 | 90/90 | 90/90 | +6.85 | 53.1% |
| 1.25 | 90/90 | 90/90 | +8.43 | 66.4% |
| 1.50 | 90/90 | 90/90 | +9.23 | 79.5% |
| 1.75 | 90/90 | 90/90 | +9.55 | 90.6% |
| **2.00** | **90/90** | **90/90** | **+9.58** | **96.1%** |

The frozen lexicographic rule selected scale 2.0 by target margin. The per-row
cap prevents the mean norm from exceeding the allowed residual scale.

## Development

| Condition | Target ones | Mean target margin | Relative norm | Digit-token rate |
|---|---:|---:|---:|---:|
| **Prototype writer** | **90/90** | **+8.88** | **90.8%** | **100%** |
| Base | 4/90 | -4.06 | 0% | 94.4% |
| Wrong ones, norm matched | 4/90 | -12.09 | 90.8% | 100% |
| Shuffled target, norm matched | 14/90 | -9.69 | 90.8% | 100% |
| Random subspace, norm matched | 11/90 | -4.83 | 90.8% | 72.2% |
| **Hard-gated identity** | **90/90** | **+12.02** | **0%** | **100%** |

The targeted writer exceeds every matched control by at least 84.4 percentage
points.

## Diagnostic gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Target ones | >=70% | 100% | Pass |
| Advantage over every matched control | >=25 points | 84.4 points | Pass |
| Identity ones | >=90% | 100% | Pass |
| Relative norm | <=100% | 90.8% | Pass |
| Target digit-token rate | 100% | 100% | Pass |

## Interpretation

The causal basis discovered from tens donor deltas is not tens-specific. At the
same late residual boundary, it supports deterministic writes of the next digit
after either one or two answer-prefix tokens.

The reusable interface is therefore:

1. capture the recipient's current coordinates in a small late-layer subspace;
2. look up the requested next-digit native coordinate prototype;
3. replace those coordinates;
4. emit zero when the base model already predicts the requested digit.

Only the class dictionary is position-conditioned. The basis and controller
are shared. This is the clearest evidence so far for a reusable native latent
interface rather than a task-specific steering vector.

## Next experiment

Replace the existing ones causal adapter in the frozen closed-loop hybrid with
this ones prototype dictionary:

- leading: existing hard-gated causal adapter;
- tens: rank-16 digit prototype at scale 1.25;
- ones: the same rank-16 interface with ones prototypes at scale 2.0.

No further component selection is permitted before development. The audit
remains sealed until the full closed-loop gate passes.
