# Phase 2 Donor-Free Tens Prototype Writer Summary

## Outcome

**A deterministic, donor-free 16-dimensional prototype writer controlled
90/90 development tens digits while preserving 90/90 identity digits.**

The writer uses no donor execution and updates no base-model weights. It
replaces the recipient's coordinates in the frozen causal subspace with a
fit-derived native tens-digit prototype. A hard gate emits exactly zero when
the unmodified model already predicts the requested digit.

## Provenance

- Frozen protocol commit: `4f74973`.
- Frozen config SHA-256:
  `f00d84a07631ad8249c50f5c4e30a2f0825ec201c0e4e501722818b92b366cd4`.
- Result SHA-256:
  `ff882c86a6b96e67bd4b984c7540fa2a0d847b0280edac6035be242e2c7008e1`.
- Prototype artifact SHA-256:
  `42b97d49573333e7817aa3464a4a916c5e26bca3bae675c5b2aaccca71a85f10`.
- Fit-state SHA-256:
  `046739bdd54ae9027ee842b9e7d0dc7ce0bbb94b46ed6c188b2070ece53207f5`.
- Fit-coordinate SHA-256:
  `7f1bf80b6769712fd23ad94cb26cea5af0931e10fb95e812e2cd4a3fda6e0b03`.
- Audit examples evaluated: 0/90.

## Mechanism

For a recipient state `h`, rank-16 orthonormal basis `B`, and requested tens
digit `d`, the writer applies:

`delta = scale * (prototype[d] - h B^T) B`

subject to a per-row residual-norm cap. If the base next-token argmax already
equals `d`, `delta` is set to exactly zero.

The prototype dictionary is the mean native coordinate vector for each digit
across 450 fit examples and four prompt views. No evaluation activation enters
the dictionary.

## Selection

Digit-only and leading-plus-tens-prefix prototypes were compared over five
scales.

| Method | Scale | Target tens | Identity tens | Target margin | Target norm |
|---|---:|---:|---:|---:|---:|
| Digit | 0.50 | 56/90 | 90/90 | +0.60 | 27.4% |
| Digit | 0.75 | 89/90 | 90/90 | +6.50 | 41.1% |
| Digit | 1.00 | 90/90 | 90/90 | +12.24 | 54.9% |
| **Digit** | **1.25** | **90/90** | **90/90** | **+14.94** | **68.6%** |
| Digit | 1.50 | 90/90 | 90/90 | +14.83 | 82.0% |
| Prefix | 1.00 | 90/90 | 90/90 | +11.57 | 56.3% |
| Prefix | 1.25 | 90/90 | 90/90 | +13.86 | 70.4% |

The frozen rule selected the digit prototype at scale 1.25. Leading-digit
conditioning was unnecessary inside this causal subspace.

## Development

| Condition | Target tens | Mean target margin | Relative norm | Digit-token rate |
|---|---:|---:|---:|---:|
| **Prototype writer** | **90/90** | **+14.55** | **72.1%** | **100%** |
| Base | 3/90 | -4.63 | 0% | 100% |
| Wrong tens, norm matched | 3/90 | -14.87 | 72.1% | 100% |
| Shuffled target, norm matched | 13/90 | -12.78 | 72.1% | 100% |
| Random subspace, norm matched | 14/90 | -5.24 | 72.1% | 86.7% |
| **Hard-gated identity** | **90/90** | **+12.57** | **0%** | **100%** |

The writer exceeds every matched control by at least 84.4 percentage points.

## Diagnostic gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Target tens | >=70% | 100% | Pass |
| Advantage over every matched control | >=25 points | 84.4 points | Pass |
| Identity tens | >=90% | 100% | Pass |
| Relative norm | <=100% | 72.1% | Pass |
| Targeted digit-token rate | 100% | 100% | Pass |

Every frozen diagnostic criterion passes.

## Interpretation

The native tens mechanism is not merely decodable; it is directly writable
through a compact, reusable coordinate interface. The successful operation is
class-conditional state replacement in a 16-dimensional late residual
subspace, not a generic steering direction.

The hard identity gate is equally important. It separates deterministic control
logic from representation learning:

- if the requested digit is already present, do nothing exactly;
- otherwise, replace the relevant native coordinates with the requested class
  prototype.

This is a concrete proof of concept for attaching deterministic control to a
pretrained model without retraining or fine-tuning its weights.

## Claim boundary and next experiment

This result is teacher-forced at the tens position. It does not yet demonstrate
three-step closed-loop arithmetic-result composition.

The diagnostic pass authorizes a frozen closed-loop hybrid:

1. use the existing donor-free leading writer at position one;
2. use this prototype implant at position two;
3. use the existing donor-free ones writer at position three;
4. retain hard gating and all matched controls;
5. evaluate complete balanced synthetic results once on development.

The audit remains sealed.
