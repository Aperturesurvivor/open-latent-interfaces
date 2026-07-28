# Phase 2 Causal Adapter Development Summary

## Outcome

**Direct causal training improved exact writing and preservation, but did not
pass the frozen advancement gate. The audit remains sealed.**

The selected three-seed adapter from Phase 2 was fine-tuned through the frozen
language model. At block 22, adapter deltas were injected into the native
residual stream and optimized with:

- target-next-token cross-entropy;
- original-next-token cross-entropy;
- KL divergence from the untouched identity distribution;
- relative intervention-norm regularization.

Every base-model parameter remained frozen.

## Frozen provenance

- Protocol commit: `8d8920e`.
- Dataset SHA-256:
  `c5ffbdec6160efba1f421ed980e40a215a7233f719e37a2a2d1ff864ca03a9f6`.
- Initial adapter result SHA-256:
  `b99fd71ec8ff354aa0ba3811bc8b291e84220925e0f915b5b3f5d6b7f5960e86`.
- Initial weights SHA-256:
  `82fbf640f2d982c488b871256e39f532ae6ad0c154bc594bb512b7f955517aa4`.
- Causal result SHA-256:
  `64614f883f124fdf71c841525e5c3f6e3a21a4b477040c9471659724eb27c3d3`.
- Causal weights SHA-256:
  `a8557a237d46d3602ecc61999582135f91b16e0934c212e98e19f1a9179d83e8`.
- Audit examples opened: 0/90.

A reporting-only metadata defect initially labeled checkpoint numbers as
transport ranks. The result artifact was mechanically corrected to the already
frozen architectures: ranks 32, 64, and 32 with widths 64, 64, and 128. No
metric, output, checkpoint, scale, or weight changed.

## Training convergence

| Position | Target CE epoch 1 | Target CE epoch 3 | Identity CE epoch 3 | Identity KL epoch 3 |
|---:|---:|---:|---:|---:|
| 1 | 0.503 | 0.043 | 0.035 | 0.199 |
| 2 | 1.658 | 1.091 | 0.011 | 0.007 |
| 3 | 1.006 | 0.404 | 0.003 | 0.003 |

The first-position causal objective fit strongly. The second position remained
substantially harder even on the fit template.

## Training-only selection

| Position | Epoch | Width | Rank | Scale | Target accuracy | Identity accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 64 | 32 | 1 | 97.8% | 96.7% |
| 2 | 2 | 64 | 64 | 1 | 13.3% | 70.0% |
| 3 | 1 | 128 | 32 | 1 | 30.0% | 48.9% |

The cross-template selection split exposed severe later-position
generalization failure before development was opened.

## Development results

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result | Mean norms |
|---|---:|---:|---:|---:|---:|---|
| Base | 3.3% | 53.3% | 14.4% | 0/90 | 84/90 | 0%, 0%, 0% |
| Random norm-matched | 3.3% | 53.3% | 15.6% | 0/90 | 83/90 | 36%, 12%, 24% |
| Same-digit adapter | 2.2% | 54.4% | 12.2% | 0/90 | 72/90 | 21%, 12%, 22% |
| Shuffled target norm-matched | 20.0% | 34.4% | 14.4% | 2/90 | 14/90 | 36%, 25%, 34% |
| Shuffled state norm-matched | 23.3% | 46.7% | 16.7% | 0/90 | 48/90 | 36%, 18%, 28% |
| **Causal adapter** | **93.3%** | **40.0%** | **34.4%** | **17/90** | **2/90** | **36%, 23%, 30%** |

All outputs remained parseable.

Relative to donor-imitation training:

- exact targets improved from 12/90 to 17/90;
- leading-digit transfer improved from 68/90 to 84/90;
- identity preservation improved from 52/90 to 72/90;
- later norms fell from 11%/59% to 23%/30%;
- later-position accuracy remained inadequate.

## Frozen advancement gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Exact target | ≥50% | 18.9% | Fail |
| Every position | ≥70% | 93.3%, 40.0%, 34.4% | Fail |
| Exact advantage over every control | ≥25 points | 16.7 points | Fail |
| Identity preservation | ≥90% | 80.0% | Fail |
| Relative norm at every position | ≤100% | 36%, 23%, 30% | Pass |
| Parse rate | 100% | 100% | Pass |

Four required gates still fail. Audit access is prohibited.

## Interpretation

Direct causal optimization is superior to residual imitation for the first
digit and improves end-to-end exactness without increasing intervention norm.
It is the strongest compressed adapter so far.

The sharp fit-to-selection collapse at positions two and three identifies the
next bottleneck as template-conditioned native state geometry. Training uses
one fit template while selection and development use unseen template families.
The adapter learns a causal write within the fit manifold but does not discover
a template-invariant interface for suffix positions.

## Next experiment

Train causally across multiple fit-only paraphrase families for each operand
pair and add an invariance objective:

1. render every fit example through several preregistered templates;
2. require the adapter's transport coefficients to agree for identical
   arithmetic targets across those templates;
3. retain causal token, identity KL, and norm losses;
4. keep selection, development, and audit template families unseen;
5. advance only if later-position and preservation gates pass.
