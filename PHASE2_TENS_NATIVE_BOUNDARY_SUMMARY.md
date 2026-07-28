# Phase 2 Tens Native-Boundary Summary

## Outcome

**A sharp late-layer native write window controls balanced counterfactual tens
digits. The selected boundary reached 90/90 development targets.**

Selection accuracy jumped from 8/90 at hidden-state index 21 to 86/90 at index
23, then reached 89/90 at indices 27 and 28. The frozen rule selected index 27
(decoder block 26). At that boundary, targeted fit-donor replacement produced
every development tens digit, while the strongest norm-matched control reached
10/90.

## Provenance

- Frozen protocol commit: `4c8afff`.
- Frozen config SHA-256:
  `1aa57de2b1e72b54cd479f1c6d0ac3b05c292367f0e074b358b5b41823e69cdf`.
- Result SHA-256:
  `4033ef607c14b71a4ec9233bff285300c84be4386df61ceb78b3c395ceddcd22`.
- New weights trained: none.
- Donor pool: fit only.
- Audit examples evaluated: 0/90.

## Selection boundary map

| Hidden index | Decoder block | Targeted tens | Strongest matched control | Target margin | Relative norm |
|---:|---:|---:|---:|---:|---:|
| 17 | 16 | 3/90 | 5/90 | -5.37 | 19.5% |
| 19 | 18 | 3/90 | 4/90 | -5.38 | 16.8% |
| 21 | 20 | 8/90 | 8/90 | -5.10 | 27.5% |
| 23 | 22 | 86/90 | 9/90 | +4.67 | 68.0% |
| 25 | 24 | 86/90 | 9/90 | +4.66 | 67.7% |
| **27** | **26** | **89/90** | **9/90** | **+14.15** | **79.4%** |
| 28 | 27 | 89/90 | 9/90 | +7.25 | 94.9% |

The selection rule chose index 27 over index 28 because target accuracy tied,
while index 27 had a larger positive target margin and lower norm.

## Development at hidden index 27

| Condition | Target tens | Mean target margin | Mean target rank | Relative norm |
|---|---:|---:|---:|---:|
| **Targeted native donor** | **90/90** | **+13.15** | **1.00** | **80.9%** |
| Base | 3/90 | -4.63 | 11.66 | 0% |
| Wrong tens, norm matched | 0/90 | -14.51 | 15.39 | 80.9% |
| Shuffled donor, norm matched | 10/90 | -13.74 | 10.13 | 80.9% |
| Random direction, norm matched | 3/90 | -4.67 | 535.24 | 80.9% |

Every targeted output is necessarily a parseable digit because all 90 top-1
tokens equal the requested digit token.

## Diagnostic evidence gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Targeted tens | >=70% | 100.0% | Pass |
| Advantage over every matched control | >=25 points | 88.9 points | Pass |
| Targeted digit-token parseability | 100% | 100% | Pass |

This passes the native diagnostic gate. It does not authorize audit because
full donor replacement is not a compact inference mechanism.

## Interpretation

The causal geometry changes abruptly between the residual boundaries after
blocks 20 and 22. By block 22, a content-matched native state almost completely
determines the next tens digit. The effect strengthens through block 26.

This also rules out the current adapter boundary as the primary failure. The
adapter operates at hidden-state index 23, where native replacement achieved
86/90 selection tens digits. The fixed-weight adapter reached only 31/90 at its
best selection scale. A valid write path is present at the exact same boundary;
the compact writer fails to reconstruct or predict enough of it.

Wrong-tens donors reverse the logit margin and score 0/90 despite identical
norm. Shuffled donor and random controls also fail. The write therefore depends
on aligned native content, not generic late-layer disruption.

## Next experiment

Separate output-rank loss from coefficient-prediction loss:

1. build a tens-specific donor-delta PCA basis from fit examples at index 27;
2. project exact selection donor deltas into frozen ranks
   `16, 32, 64, 128, 256, 512`;
3. compare reconstructed-delta target accuracy with shuffled-coefficient and
   random norm-matched controls;
4. select the smallest rank passing the native diagnostic gate;
5. evaluate that rank once on development.

If rank 64 retains native control, coefficient prediction is the bottleneck.
If it does not, the present output subspace is too small or was learned from
the wrong donor distribution.

The audit remains sealed.
