# Phase 1I Local-Transport Dictionary Summary

## Outcome

**Nearest-state transport did not beat the linear conditional bridge.**

The experiment stored 480 training recipient states and their native donor
transport deltas at each answer position. At inference it restricted retrieval
to the requested next digit, selected nearest recipient states in a
32-dimensional PCA space, averaged their transport coefficients, and projected
them through a selected low-rank output basis.

No donor execution was required at inference, but the dictionary was intended
as a diagnostic for local nonlinearity rather than a final compact bridge.

## Frozen design

- Protocol commit: `34fc44b`.
- Fit/selection/development: 60/30/45 examples.
- Training transport pairs: 480 per answer position.
- Audit examples opened: 0/45.
- Boundary: hidden state 23 / decoder block 22.
- State rank: 32.
- Candidate neighbors: 1, 3, 5, and 10.
- Candidate transport ranks: 8, 16, 32, and 64.
- Candidate scales: 0.5, 1, and 2.
- Result SHA-256:
  `5a7bf858dca1ecd0c79bc4e128d5ebf9c9f88f272ad26f9235cbac8e23615140`.

## Training-only selections

| Position | Neighbors | Rank | Scale | Selection accuracy | Mean delta / residual norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 64 | 1 | 86.7% | 64.0% |
| 2 | 1 | 16 | 2 | 46.7% | 111.1% |
| 3 | 1 | 16 | 2 | 76.7% | 96.7% |

Every position selected one-nearest-neighbor retrieval. Later positions again
required large scaling.

## Development results

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| Base | 0.0% | 31.1% | 11.1% | 0/45 | 43/45 |
| Random norm-matched | 2.2% | 24.4% | 4.4% | 0/45 | 19/45 |
| Same-digit retrieval | 8.9% | 31.1% | 15.6% | 0/45 | 9/45 |
| Shuffled target norm-matched | 11.1% | 6.7% | 8.9% | 0/45 | 1/45 |
| Shuffled state norm-matched | 31.1% | 31.1% | 37.8% | 1/45 | 2/45 |
| **Local transport** | **73.3%** | **26.7%** | **60.0%** | **4/45** | **2/45** |
| Phase 1G linear conditional bridge | 91.1% | 40.0% | 48.9% | 6/45 | 0/45 |
| Full native donor upper bound | 93.3% | 93.3% | 97.8% | 38/45 | 0/45 |

The dictionary produced four exact targets:

```text
688 → 768
103 → 214
535 → 623
471 → 575
```

## Interpretation

Local retrieval improves third-position transfer relative to the linear bridge,
from 22/45 to 27/45. Correct state alignment also matters: shuffled-state
retrieval falls from 4/45 to 1/45 exact targets.

Those gains do not solve the program's bottleneck. Second-position transfer is
only 12/45, exact writing falls below the linear bridge, and later
interventions are 113% and 104% of recipient residual norm on development.

The evidence rejects a simple nearest-neighbor explanation for the missing
donor structure. A 60-example fit set does not provide enough local coverage
for reliable state-conditioned transport.

## Claim boundary

- Retrieval is a development non-pass.
- It is not compact enough to be a final adapter.
- Later writes exceed residual norm.
- Exact writing does not beat Phase 1G.
- The frozen audit remains unopened.

## Next program decision

Stop iterating increasingly flexible bridges on the same 60-example fit set.
The full native donor result already proves the write boundary; the compressed
experiments now consistently expose data scarcity and poor coverage.

The next phase should:

1. generate a much larger arithmetic training corpus with disjoint operand
   pairs and independently frozen development/audit splits;
2. capture multiple target transports per recipient at the proven boundary;
3. train a genuinely nonlinear, bottlenecked adapter with explicit norm and
   preservation objectives;
4. select architecture and stopping criteria without opening the new audit;
5. compare against Phase 1G, retrieval, full donor, random, and shuffled
   controls;
6. audit only after the adapter, thresholds, and artifact hashes are frozen.
