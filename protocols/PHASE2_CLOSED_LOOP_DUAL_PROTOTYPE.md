# Phase 2 Protocol: Closed-Loop Dual-Prototype Implant

## Question

Does the shared rank-16 next-digit interface produce complete balanced
counterfactual results when used at both suffix positions in closed loop?

The tens and ones dictionaries each passed 90/90 target and identity
development diagnostics. This experiment substitutes both into the frozen
hybrid without further training or selection.

## Frozen components

| Position | Component | Hidden index | Scale |
|---|---|---:|---:|
| Leading | hard-gated causal adapter | 23 | 1.0 |
| Tens | shared-basis digit prototype | 27 | 1.25 |
| Ones | shared-basis digit prototype | 27 | 2.0 |

The two suffix components share:

- the same rank-16 basis;
- the same late residual boundary;
- the same coordinate-replacement algorithm;
- the same deterministic hard gate.

Only their ten fit-derived prototype vectors and selected scales differ.

Every source result, weight file, basis, prototype artifact, dataset, and target
assignment is verified by SHA-256.

## Development conditions

- untouched base;
- targeted dual-prototype implant;
- original-result identity with hard gating;
- shuffled target, norm matched;
- shuffled state, norm matched;
- random direction, norm matched.

All three digits are generated autoregressively. No idealized target prefix is
injected after generation starts.

## Frozen advancement gate

- exact target result at least 50%;
- every answer position at least 70%;
- exact advantage over every matched control at least 25 points;
- identity preservation at least 90%;
- mean relative norm at most 1.0 at every position;
- parse rate 100%.

If all gates pass, freeze an audit configuration containing these exact
components, hashes, controls, and thresholds before evaluating any audit model
outputs. This development experiment does not itself authorize an unfrozen
audit.
