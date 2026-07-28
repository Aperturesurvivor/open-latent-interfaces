# Phase 3 Phi Leading-Prototype Rank Summary

## Outcome

The bounded follow-up selected rank 32 and scale 1.0 for Phi's donor-free
leading-digit controller:

- target leading digit: 75/90 (83.33%);
- identity leading digit: 89/90 (98.89%);
- mean relative intervention norm: 45.51%;
- digit-token rate: 90/90;
- selection gate: pass.

The rank curve supports the hypothesis that donor-free class prototypes require
more coordinates than donor-specific transport:

| Rank | Selected scale | Target | Identity | Digit-token rate | Gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 1.25 | 54/90 | 90/90 | 90/90 | fail |
| 16 | 1.25 | 63/90 | 90/90 | 89/90 | fail |
| 32 | 1.0 | 75/90 | 89/90 | 90/90 | pass |
| 64 | 1.0 | 80/90 | 88/90 | 90/90 | pass |
| 128 | 1.25 | 79/90 | 90/90 | 90/90 | pass |

Rank 32 is the smallest rank satisfying every frozen conjunct.

## Exact-count correction

The initial result stored 63/90 as float32 `0.699999988` and compared it
strictly with `0.7`, incorrectly marking the accuracy conjunct false. The raw
result was preserved. A committed correction recomputed threshold decisions
from integer counts and reused the nested prototype coordinates without
repeating model evaluation.

After correction, rank 16 meets the accuracy conjunct but still fails because
one output is not a decimal digit token. Therefore the numerical bug does not
change the final selected rank or artifact: rank 32 remains correct.

## Frozen provenance

- frozen follow-up commit: `c9fa739`
- raw-result preservation and correction code: `eae7343`
- raw result SHA-256:
  `f470a4fbbb7067fd753acb8add0c87e3f5545fead8d3d3dc22c36324098c9c12`
- corrected result SHA-256:
  `cf3c3be1ba5f11d1b461f91e98fe2f5c951c55f92b3ddc5111da5fe78ac638c9`
- selected prototype SHA-256:
  `3a0623d8f077da61bad8aa5e32ca3e2b9a561f4c0168884e1805aac631aec1b0`
- correction configuration SHA-256:
  `bb01a53ff31046f784d2a61a7ee296ec1baa1991d1bed85ed1a35b80a0a06cde`
- elapsed model-evaluation time: 365.74 seconds
- model evaluations repeated for correction: no

## Decision and claim boundary

Freeze rank 32, hidden index 24, and scale 1.0 for closed-loop development.
Combine it with the already locked rank-32 suffix controller at hidden index
30 and scale 1.25. This remains selection-only evidence; complete generation,
matched controls, development generalization, and audit are untested.
