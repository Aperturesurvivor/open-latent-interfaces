# Phase 1G Conditional-Transport Bridge Summary

## Outcome

**Recipient conditioning was causally useful and sharply reduced intervention
norms, but exact three-digit writing remained a development non-pass.**

The bridge learned from 480 native recipient/donor pairs per answer position:
eight donors for each of 60 training recipients, covering every alternative
leading-result class. A ridge model mapped a 32-dimensional recipient-state
representation plus target-digit interactions into a reduced-rank transport.

At inference, the bridge used the recipient state and desired next digit. No
live donor execution was required.

## Frozen design

- Protocol commit: `9529ef9`.
- Fit/selection/development: 60/30/45 examples.
- Training transport pairs: 480 per answer position.
- Audit examples opened: 0/45.
- Boundary: hidden state 23 / decoder block 22.
- State rank: 32.
- Candidate transport ranks: 4, 8, 16, and 32.
- Candidate ridge values: 1, 10, and 100.
- Candidate scales: 0.5, 1, 2, and 4.
- Result SHA-256:
  `ac9af12cd19b02fc980d7356b90c140446328bd41e6b08538c01bbf39798c55c`.

## Training-only selections

| Position | Transport rank | Ridge | Scale | Selection accuracy | Mean delta / residual norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 1 | 1 | 96.7% | 51.0% |
| 2 | 32 | 1 | 1 | 26.7% | 51.3% |
| 3 | 32 | 1 | 1 | 70.0% | 50.1% |

The second position remained the clear bottleneck before development was
opened.

## Development results

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| Base | 0.0% | 31.1% | 11.1% | 0/45 | 43/45 |
| Random norm-matched | 0.0% | 31.1% | 13.3% | 0/45 | 40/45 |
| Same-digit bridge | 6.7% | 17.8% | 17.8% | 0/45 | 17/45 |
| Shuffled target norm-matched | 6.7% | 11.1% | 8.9% | 0/45 | 2/45 |
| Shuffled state norm-matched | 33.3% | 24.4% | 15.6% | 0/45 | 5/45 |
| **Conditional transport** | **91.1%** | **40.0%** | **48.9%** | **6/45** | **0/45** |
| Phase 1F class-mean transport | 68.9% | 37.8% | 33.3% | 5/45 | 0/45 |
| Full native donor upper bound | 93.3% | 93.3% | 97.8% | 38/45 | 0/45 |

The conditional bridge produced six complete targets:

```text
840 → 928
471 → 575
696 → 768
721 → 840
539 → 623
687 → 768
```

## State-conditioning falsifier

Shuffling recipient states while preserving desired digits and matching every
delta norm reduced:

- leading-digit transfer from 41/45 to 15/45;
- second-digit transfer from 18/45 to 11/45;
- third-digit transfer from 22/45 to 7/45;
- exact targets from 6/45 to 0/45.

Recipient conditioning is therefore not a decorative input. Correct
state/target alignment contributes causally to the write.

## Norm improvement

The class-mean paired writer required development interventions of 74%, 135%,
and 241% of recipient residual norm. Conditional transport used 53%, 54%, and
59%.

This is a substantial improvement in compactness and stability even though
exact-answer reliability increased only from 5/45 to 6/45.

## Interpretation

Phase 1G identifies recipient-conditioned transport as a real component of the
native interface. It nearly restores the full donor's leading-digit effect and
improves third-position transfer without extreme perturbations.

The unresolved second position is informative. At that boundary the bridge
receives only the desired next digit, while the native donor state is
conditioned on the complete target result and donor computation. A next-digit
label may be under-specified for reconstructing the target-consistent state.

## Claim boundary

- Six exact results are not a reliable deterministic graft.
- Same-digit preservation remains only 17/45.
- Transport ranks 32 at positions two and three are compact relative to model
  width but not minimal.
- The result is development-only.
- The frozen audit remains unopened.

## Next experiment

Supply the bridge with the complete deterministic result, not only the next
digit. The external calculator naturally knows that value, and a typed
interface is not required to discard it.

A full-result conditional bridge should:

1. retain recipient-state conditioning;
2. encode all three target digits as typed inputs at every answer position;
3. use training-only rank/ridge/scale selection;
4. compare against shuffled-result, shuffled-state, random, and same-result
   controls;
5. measure teacher-forced local writes alongside closed-loop composition;
6. remain donor-free at inference and keep the audit sealed.
