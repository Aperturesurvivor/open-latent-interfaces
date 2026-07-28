# Phase 5 Qwen Donor-Free Development Summary

## Outcome

Both fixed Qwen donor-free interfaces passed their single untouched
development run without a metric or scale correction.

### Operand writer

At hidden-state index 12 and scale 1.0:

- 44/45 target tens digits;
- 43/45 exact target results;
- 1/45 exact for the wrong-class prototype;
- 0/45 for the isotropic control;
- a 42/45, or 93.33-point, exact-result advantage;
- 45/45 parseable outputs.

### Universal carry writer

At hidden-state index 16 and scale 1.6:

- 34/45 target tens digits;
- 30/45 exact target results;
- 22/45 target tens and 20/45 exact for matched no-carry;
- 0/45 for the isotropic control;
- a 12/45, or 26.67-point, tens-position advantage;
- 45/45 parseable outputs.

## Cross-family result

Qwen independently supports the same interface types as Phi:

1. a source-digit-conditioned early operand edit;
2. a later class-invariant rank-one carry direction.

Their model-specific boundaries and scales differ. No Phi tensor or fitted
state was used.

## Frozen provenance

- frozen experiment commit: `38428f1`
- configuration SHA-256:
  `054e7b16fada68e0d6e6cbedfcc3fd94df9e2401305be962c5a14deffcc10e49`
- result SHA-256:
  `e32603a8e0436a674c5d7a800a10ae92b254f0388b55e8f25b709e7cafba856a`
- elapsed evaluation time: 33.51 seconds
- development quartets: 45, unfiltered

## Decision

Authorize a separately committed one-shot Qwen audit with the exact fixed
artifacts, token contract, layers, scales, controls, and metrics. Audit remains
sealed until that package is immutable.
