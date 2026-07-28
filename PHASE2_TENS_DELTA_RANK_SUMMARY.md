# Phase 2 Tens Native-Delta Rank Summary

## Outcome

**A 16-dimensional fit-only transport basis retains nearly all native tens
control at hidden-state index 27. Output rank is not the active bottleneck.**

Rank 8 controlled 50/90 selection tens digits. Rank 16 jumped to 89/90 and was
the smallest rank passing the frozen diagnostic gate. On development, the
rank-16 projected donor delta controlled 89/90 tens digits, versus 10/90 for
the strongest norm-matched control.

## Provenance

- Frozen protocol commit: `b06bf03`.
- Frozen config SHA-256:
  `15d22c4cfc04f23ff1ee123d391b2c3ce37a357a707d621fd14e40fe0ef1129a`.
- Result SHA-256:
  `6bed6d5bbd2ba68901914afec6d4f739f8ad8f5d3301964c83d294c383f74244`.
- Basis SHA-256:
  `8c9b0e53a3b8216e724ab37519a51723c0e0697d956ade56dceb362217b3084d`.
- Fit states SHA-256:
  `f8d3413cefedb6b4d97c065551c86883802a743af01e9f9d8c6d2c4e877a7359`.
- Fit deltas SHA-256:
  `b2bdcc44b7721f4c57bbcae4048e91d7e186c8c1c2d4306f447987272dea7c19`.
- Audit examples evaluated: 0/90.

## Selection rank curve

| Rank | Targeted tens | Strongest matched control | Target margin | Relative norm | Gate |
|---:|---:|---:|---:|---:|---|
| 8 | 50/90 | 8/90 | +1.16 | 60.0% | Fail |
| **16** | **89/90** | **11/90** | **+11.21** | **65.9%** | **Pass** |
| 32 | 90/90 | 9/90 | +12.81 | 70.4% | Pass |
| 64 | 89/90 | 9/90 | +13.59 | 73.5% | Pass |
| 128 | 89/90 | 9/90 | +13.78 | 75.3% | Pass |
| 256 | 89/90 | 9/90 | +13.87 | 76.7% | Pass |
| 450 | 89/90 | 9/90 | +13.98 | 77.5% | Pass |

The non-monotonic one-example difference above rank 16 is consistent with
small added components perturbing an already decisive logit margin. Rank 32
matches all 90 selection targets.

## Development at rank 16

| Condition | Target tens | Mean target margin | Relative norm | Digit-token rate |
|---|---:|---:|---:|---:|
| **Projected targeted delta** | **89/90** | **+10.34** | **68.3%** | **100%** |
| Base | 3/90 | -4.63 | 0% | 100% |
| Wrong tens, norm matched | 0/90 | -12.49 | 68.3% | 100% |
| Shuffled coefficients, norm matched | 8/90 | -11.15 | 68.3% | 100% |
| Random subspace, norm matched | 10/90 | -5.45 | 68.3% | 91.1% |

The development advantage over every matched control is 87.8 percentage
points.

## Diagnostic gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Targeted tens | >=70% | 98.9% | Pass |
| Advantage over every matched control | >=25 points | 87.8 points | Pass |
| Relative norm | <=100% | 68.3% | Pass |
| Targeted digit-token rate | 100% | 100% | Pass |

## Interpretation

The selected boundary's causal tens transport is highly compressible. Sixteen
fit-derived orthogonal directions preserve 89 of the 90 native donor effects.
Increasing rank to 32 recovers all selection examples; increasing it further
does not materially improve control.

This result rules out insufficient output dimensionality at hidden-state index
27. It does not prove that the earlier rank-64 basis at index 23 contained the
same directions: that basis came from a different boundary and donor
distribution. It does show that a new tens-specific writer does not need a
large output space.

The donor-free challenge is now sharply defined. Given a recipient state and a
requested tens digit, predict the 16 coefficients that the projected native
delta currently obtains from a donor.

## Next experiment

Train and compare donor-free coefficient maps at index 27 in the frozen rank-16
basis:

1. target-digit prototype coefficients;
2. ridge regression from recipient state coordinates plus target digit;
3. a small nonlinear MLP only if the linear map fails.

Include zero-delta identity examples and hard zero-gating when the requested
digit already matches the unmodified next digit. Select on the selection split,
then perform one development evaluation with the same matched controls.

The audit remains sealed.
