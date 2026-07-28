# Phase 1H Full-Result Transport Summary

## Outcome

**Raw full-result conditioning failed to improve the bridge and will not
proceed to audit.**

The Phase 1G bridge received only the desired next digit. Phase 1H instead
encoded all three target digits at every answer position and interacted that
29-component typed code with a 16-dimensional recipient-state representation.

The hypothesis was that the complete deterministic result would disambiguate
the native target state. It did not.

## Frozen design

- Protocol commit: `c0b8830`.
- Fit/selection/development: 60/30/45 examples.
- Training transport pairs: 480 per answer position.
- Audit examples opened: 0/45.
- Boundary: hidden state 23 / decoder block 22.
- State rank: 16.
- Feature count: 510.
- Candidate transport ranks: 8, 16, and 32.
- Candidate ridge values: 1, 10, and 100.
- Candidate scales: 0.5, 1, and 2.
- Result SHA-256:
  `d52ff92bb93c0b253f943424b9c3edd7b215235c79145718aa3d77593746344c`.

## Training-only selections

| Position | Transport rank | Ridge | Scale | Selection accuracy | Mean delta / residual norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 10 | 1 | 63.3% | 44.6% |
| 2 | 32 | 1 | 1 | 26.7% | 54.7% |
| 3 | 16 | 1 | 1 | 26.7% | 39.8% |

The full-result bridge was already weaker than Phase 1G on training-only
selection at positions one and three.

## Closed-loop development results

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| Base | 0.0% | 31.1% | 11.1% | 0/45 | 43/45 |
| Random norm-matched | 2.2% | 37.8% | 11.1% | 0/45 | 36/45 |
| Same-result bridge | 4.4% | 28.9% | 15.6% | 0/45 | 3/45 |
| Shuffled result norm-matched | 11.1% | 6.7% | 11.1% | 0/45 | 0/45 |
| Shuffled state norm-matched | 28.9% | 15.6% | 13.3% | 0/45 | 4/45 |
| **Full-result transport** | **42.2%** | **13.3%** | **24.4%** | **1/45** | **0/45** |
| Phase 1G next-digit bridge | 91.1% | 40.0% | 48.9% | 6/45 | 0/45 |

## Teacher-forced local results

| Condition | Digit 1 | Digit 2 | Digit 3 |
|---|---:|---:|---:|
| Base | 0.0% | 17.8% | 13.3% |
| Random norm-matched | 2.2% | 8.9% | 11.1% |
| Shuffled result norm-matched | 11.1% | 6.7% | 8.9% |
| Shuffled state norm-matched | 28.9% | 6.7% | 15.6% |
| **Full-result transport** | **42.2%** | **20.0%** | **22.2%** |

Teacher forcing did not rescue the writer. The low exact rate is therefore not
primarily caused by error propagation from an incorrect earlier prefix; the
local transport itself is weak.

## Interpretation

The external calculator's complete result is useful information, but this
particular representation is not a useful bridge. Position-specific one-hot
features create 510 regression features from 480 training pairs and must
generalize to unseen digit combinations. The result code is semantically
structured but geometrically sparse.

Phase 1H rejects the simple claim that “more target digits as one-hot features”
will close the donor gap. Phase 1G's narrower next-digit interface remains the
strongest compressed writer.

## Claim boundary

- This is a negative development result.
- The single exact output is not evidence of reliable composition.
- Teacher-forced transfer is weak.
- Same-result preservation is poor.
- No audit data was opened.

## Next experiment

Test a nonlinear local transport dictionary:

1. store training recipient states and native transport deltas;
2. restrict retrieval to the requested next digit;
3. retrieve nearest state-matched transports in a frozen PCA space;
4. average and project them into a low-rank output;
5. select neighbor count, rank, and scale on training-only validation;
6. compare correct retrieval with shuffled-state, shuffled-target, random, and
   class-mean controls.

This asks whether recipient conditioning is locally nonlinear. If nearest-state
transport materially exceeds the linear Phase 1G bridge, it can later be
distilled into a compact learned adapter.
