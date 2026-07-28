# Phase 1E Donor-Free Typed-Writer Summary

## Outcome

**A low-rank donor-free writer retained strong leading-digit control but failed
to compose a complete result.**

Three position-specific between-class subspaces were learned from 60 training
examples. Rank and intervention scale were selected independently at each
position using the remaining 30 training examples. Development data was
evaluated only after those choices were fixed.

At inference, the writer received the recipient's native state and a desired
next digit. It did not use a live donor state.

## Frozen design

- Protocol commit: `f4d03a0`.
- Fit/selection/development: 60/30/45 examples.
- Audit examples opened: 0/45.
- Boundary: hidden state 23 / decoder block 22.
- Candidate ranks: 1, 2, 4, and 8.
- Candidate scales: 0.5, 1, 2, 4, 8, 16, and 32.
- Selection objective: next-token target accuracy, with smaller rank favored
  on ties.
- Result SHA-256:
  `e07b60455091d448b1e2b41f21a9d8e48e6819ad7d31a9db06d96bb053b4b629`.

## Training-only selections

| Position | Rank | Scale | Selection accuracy | Mean delta / residual norm |
|---:|---:|---:|---:|---:|
| 1 | 8 | 2 | 76.7% | 72.2% |
| 2 | 1 | 8 | 36.7% | 111.1% |
| 3 | 8 | 2 | 46.7% | 59.0% |

The second-position selection was already weak and required a perturbation
larger than the recipient residual norm. That was a warning sign before
development evaluation.

## Development results

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| Base | 0.0% | 31.1% | 11.1% | 0/45 | 43/45 |
| Random norm-matched | 4.4% | 24.4% | 13.3% | 0/45 | 26/45 |
| Same-digit writer | 17.8% | 24.4% | 15.6% | 1/45 | 9/45 |
| Shuffled target norm-matched | 8.9% | 13.3% | 8.9% | 0/45 | 1/45 |
| **Typed writer** | **80.0%** | **15.6%** | **33.3%** | **2/45** | **0/45** |
| Full native donor upper bound | 93.3% | 93.3% | 97.8% | 38/45 | 0/45 |

The first-position result is a meaningful partial pass: the donor-free rank-8
subspace transferred the intended leading digit on 36/45 examples and clearly
separated from all controls.

The end-to-end writer is a non-pass. It produced only 2/45 complete targets,
and the second-position intervention generalized below the unmodified base
condition. Same-digit preservation was also poor, showing substantial
collateral disruption.

## Interpretation

The result distinguishes two problems that the full native donor state had
hidden:

1. **Digit encoding:** a small between-class subspace is sufficient to write
   the initial target digit with high reliability.
2. **Counterfactual state transport:** after a target prefix diverges from the
   recipient's correct answer, replacing only generic digit coordinates does
   not make the remaining native state coherent with that prefix.

The full donor replacement supplies both the desired digit and a
prompt/prefix-consistent computational state. The prototype writer supplies
only the former. Its second-step failure is therefore evidence that
multi-token composition requires a contextual transport or repair component,
not merely a stronger digit direction.

## Claim boundary

- The leading-digit interface is donor-free but remains development-only.
- Rank 8 is compact relative to the model width, but its intervention is still
  large at roughly 73% of residual norm.
- The writer is not preservation-safe.
- Exact three-digit writing did not pass.
- No frozen audit was opened.

## Next experiment

Learn a low-rank **paired transport writer** from training-only
recipient/donor state differences. Training recipient states will already
contain the counterfactual target prefix, so the learned delta must repair the
recipient-to-target context mismatch as well as encode the next digit.

The paired writer must be evaluated against the existing prototype writer,
random norm-matched, shuffled-target, and same-digit controls. This directly
tests whether contextual state transport is the missing component.
