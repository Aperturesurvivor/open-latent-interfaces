# Phase 2 Scaled Adapter Development Summary

## Outcome

**The scaled nonlinear adapter improved data coverage and intervention norms,
but failed the frozen advancement gate. The audit remains sealed.**

Phase 2 expanded from 60 Phase 1 fit examples to 450 fit examples and from 480
to 1,800 targeted native transports per answer position. A three-seed ensemble
of two-layer GELU adapters predicted reduced-rank transport coefficients from a
64-dimensional recipient state and desired next digit.

The adapter was trained by imitating native donor transports, with 450 explicit
identity examples per position, identity upweighting, and an excess-norm
penalty.

## Frozen provenance

- Dataset commit: `bba26d8`.
- Adapter protocol commit: `afff5eb`.
- Serialization-only repair: `244f955`.
- Dataset SHA-256:
  `c5ffbdec6160efba1f421ed980e40a215a7233f719e37a2a2d1ff864ca03a9f6`.
- Result SHA-256:
  `b99fd71ec8ff354aa0ba3811bc8b291e84220925e0f915b5b3f5d6b7f5960e86`.
- Selected weights SHA-256:
  `82fbf640f2d982c488b871256e39f532ae6ad0c154bc594bb512b7f955517aa4`.
- Audit examples opened: 0/90.

The first execution completed capture, fitting, and selection but stopped
before development when safetensors rejected non-contiguous PCA views. The
repair only packed tensors contiguously; it did not change data, model,
training, selection, controls, or gates.

## Training-only selections

| Position | Hidden width | Transport rank | Scale | Target accuracy | Identity accuracy | Target norm |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 32 | 1.0 | 87.8% | 97.8% | 32.0% |
| 2 | 64 | 64 | 0.5 | 44.4% | 98.9% | 11.7% |
| 3 | 128 | 32 | 2.0 | 41.1% | 62.2% | 55.6% |

Selection exposed the same later-position weakness before development was
opened.

## Development results

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result | Mean norms |
|---|---:|---:|---:|---:|---:|---|
| Base | 3.3% | 53.3% | 14.4% | 0/90 | 84/90 | 0%, 0%, 0% |
| Random norm-matched | 4.4% | 53.3% | 14.4% | 0/90 | 77/90 | 34%, 7%, 47% |
| Same-digit adapter | 1.1% | 53.3% | 11.1% | 0/90 | 52/90 | 19%, 6%, 43% |
| Shuffled target norm-matched | 17.8% | 40.0% | 11.1% | 2/90 | 9/90 | 34%, 11%, 64% |
| Shuffled state norm-matched | 17.8% | 45.6% | 18.9% | 0/90 | 34/90 | 34%, 9%, 54% |
| **Scaled adapter** | **75.6%** | **40.0%** | **35.6%** | **12/90** | **5/90** | **34%, 11%, 59%** |

All 90 adapter outputs were parseable.

## Frozen advancement gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Exact target | ≥50% | 13.3% | Fail |
| Every position | ≥70% | 75.6%, 40.0%, 35.6% | Fail |
| Exact advantage over every control | ≥25 points | 11.1 points | Fail |
| Identity preservation | ≥90% | 57.8% | Fail |
| Relative norm at every position | ≤100% | 34%, 11%, 59% | Pass |
| Parse rate | 100% | 100% | Pass |

Because four required gates fail, opening the 90-example audit would be
methodologically invalid.

## Interpretation

Scaling the corpus solved neither suffix writing nor preservation, but it did
produce a cleaner adapter:

- target-leading control remains strong;
- all mean intervention norms stay below one residual norm;
- correctly aligned state/target inputs outperform shuffled controls;
- three seeded models agree through an averaged bridge;
- selected weights are small enough to distribute directly.

The central limitation is the training objective. Mean-squared imitation of a
full native donor delta spends capacity reconstructing latent variation that
does not control the requested token. Averaging those deltas can also cancel
the small causal component while retaining disruptive context.

## Next experiment

Train the adapter against the frozen model's actual downstream behavior:

1. inject the adapter at block 22 during training;
2. backpropagate target-next-token cross-entropy through frozen downstream
   blocks;
3. add identity KL/preservation loss on native answers;
4. retain explicit relative-norm regularization;
5. train separately for each answer position under correct target prefixes;
6. keep the base model frozen and export only adapter weights;
7. use the same sealed Phase 2 audit only after a new development gate passes.

This replaces donor-state imitation with a causal sufficiency objective.
