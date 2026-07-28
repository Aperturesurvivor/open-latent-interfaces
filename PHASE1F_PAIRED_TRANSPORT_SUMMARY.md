# Phase 1F Paired-Transport Writer Summary

## Outcome

**Class-conditioned paired transport improved later-digit writing, but remained
an end-to-end non-pass.**

The writer was trained on residual differences between a recipient prompt and
a matched native donor while both contexts already contained the same target
prefix. This directly tested whether the Phase 1E failure was caused by
recipient/prefix incoherence.

At inference, the writer used only the desired next digit. No donor execution
was required.

## Frozen design

- Protocol commit: `d753cdf`.
- Fit/selection/development: 60/30/45 examples.
- Audit examples opened: 0/45.
- Boundary: hidden state 23 / decoder block 22.
- Candidate ranks: 1, 2, 4, and 8.
- Candidate scales: 0.5, 1, 2, 4, 8, and 16.
- Result SHA-256:
  `7d215ea4fad82e810b4d35a92433967d7e78547f0f2b2c0982c3cbc952a35c6e`.

## Training-only selections

| Position | Rank | Scale | Selection accuracy | Mean delta / residual norm |
|---:|---:|---:|---:|---:|
| 1 | 8 | 2 | 63.3% | 73.5% |
| 2 | 8 | 4 | 50.0% | 168.9% |
| 3 | 8 | 8 | 46.7% | 265.9% |

All three positions selected the maximum candidate rank. Later positions
required increasingly extreme scales.

## Development results

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| Base | 0.0% | 31.1% | 11.1% | 0/45 | 43/45 |
| Random norm-matched | 4.4% | 20.0% | 6.7% | 0/45 | 4/45 |
| Same-digit transport | 13.3% | 22.2% | 15.6% | 0/45 | 6/45 |
| Shuffled target norm-matched | 6.7% | 24.4% | 4.4% | 0/45 | 1/45 |
| **Paired transport** | **68.9%** | **37.8%** | **33.3%** | **5/45** | **0/45** |
| Phase 1E prototype writer | 80.0% | 15.6% | 33.3% | 2/45 | 0/45 |
| Full native donor upper bound | 93.3% | 93.3% | 97.8% | 38/45 | 0/45 |

The paired writer more than doubled second-position target transfer relative to
the prototype writer, from 7/45 to 17/45. Exact target results rose from 2/45
to 5/45. The controls did not reproduce complete targets.

## Interpretation

The improvement supports the Phase 1E diagnosis: native writing after a
counterfactual prefix requires context repair in addition to a generic digit
coordinate. Learning deltas from prefix-aligned recipient/donor pairs captures
some of that repair.

However, a single mean transport for each digit is not enough. The full native
donor remains dramatically stronger, while the compressed writer needs
interventions larger than the entire recipient residual at positions two and
three. This indicates that the missing transport varies with the recipient's
specific computational state.

The current evidence therefore supports a decomposition:

```text
native write
  = target-digit component
  + prefix-coherence repair
  + recipient-conditioned transport
```

Phase 1E recovered much of the first term. Phase 1F recovered part of the
second. The third remains unresolved.

## Claim boundary

- The paired writer is donor-free at inference but development-only.
- Five exact results do not constitute a reliable deterministic graft.
- Later interventions are too large to count as compact or preservation-safe.
- Rank 8 saturated the frozen grid.
- The frozen audit remains unopened.

## Next experiment

Fit an example-conditioned low-rank transport model using multiple donor pairs
per training recipient. The bridge should predict transport coefficients from
the recipient state plus the desired digit, rather than applying a single
class mean.

The next gate should require:

- clear improvement over the 5/45 paired-mean result;
- later-position target transfer above matched controls;
- materially lower intervention norms;
- same-digit preservation;
- no live donor execution;
- no audit access until the architecture and thresholds are frozen.
