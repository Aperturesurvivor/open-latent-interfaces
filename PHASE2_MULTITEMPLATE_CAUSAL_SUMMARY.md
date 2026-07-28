# Phase 2 Multitemplate Causal Adapter Summary

## Outcome

**Fit-only paraphrase training modestly improved suffix transfer and
preservation, but still failed four advancement gates. The audit remains
sealed.**

The causal adapter was trained under four preregistered fit-only prompt
families. Each batch used two views of the same arithmetic target and added
symmetric KL consistency between their adapted next-token distributions.
Selection and development templates remained unseen.

## Provenance

- Protocol commit: `5394eb1`.
- Chat-rendering repair: `98a7ae9`.
- Valid result SHA-256:
  `5fa70bab6fb4ea4566d5954737574d0ce3d21d4b4b3ac533fd82898f3af04b6d`.
- Valid weights SHA-256:
  `86915821394c96d9b2e5720965e3a2bb448f5c223a03c309048939c83aab5fb8`.
- Audit examples opened: 0/90.

The first execution accidentally passed fit paraphrases as raw strings rather
than through the model's chat template. That run is preserved with status
`invalidated_training_prompt_contract` and is excluded from every claim. The
repair changed only prompt rendering; templates, losses, seeds, checkpoints,
selection, development, and gates remained fixed.

## Development comparison

| Metric | Single-template causal | Multitemplate causal |
|---|---:|---:|
| Digit 1 | 84/90 | 83/90 |
| Digit 2 | 36/90 | 38/90 |
| Digit 3 | 31/90 | 34/90 |
| Exact target | 17/90 | 18/90 |
| Identity preservation | 72/90 | 80/90 |
| Mean norms | 36%, 23%, 30% | 36%, 23%, 32% |
| Parse rate | 100% | 100% |

Matched-control exact results were:

- base: 0/90;
- random norm-matched: 0/90;
- shuffled state: 0/90;
- shuffled target: 1/90.

## Frozen gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Exact target | ≥50% | 20.0% | Fail |
| Every position | ≥70% | 92.2%, 42.2%, 37.8% | Fail |
| Exact advantage over every control | ≥25 points | 18.9 points | Fail |
| Identity preservation | ≥90% | 88.9% | Fail |
| Relative norm at every position | ≤100% | 36%, 23%, 32% | Pass |
| Parse rate | 100% | 100% | Pass |

## Interpretation

Template diversity is useful: it improves both suffix positions and recovers
eight additional identity outputs. It is not the main missing mechanism.

All compressed causal adapters still reconstruct deltas inside a frozen PCA
basis learned from the largest-variance native donor differences. Full donor
replacement proves that the causal information exists, but PCA can discard
low-variance token-controlling directions while preserving high-variance
contextual changes.

## Next experiment

Make the transport output basis trainable under causal loss:

1. initialize from the selected donor-PCA basis;
2. optimize the basis jointly with adapter coefficients;
3. penalize loss of row orthogonality;
4. retain hard norm caps, identity CE/KL, and multitemplate consistency;
5. export the learned basis as part of the adapter;
6. keep the audit sealed until every development gate passes.
