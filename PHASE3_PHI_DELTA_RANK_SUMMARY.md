# Phase 3 Phi Causal-Rank Summary

## Outcome

Low-rank projections retained full-native donor control at all three Phi answer
positions. The smallest passing ranks were:

| Basis | Fit position | Evaluation position | Selected rank | Target digit | Strongest control | Mean relative norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Leading | leading | leading | 8 | 74/90 | 16/90 | 51.31% |
| Suffix | tens | tens | 32 | 88/90 | 8/90 | 67.42% |
| Suffix | tens | ones | 32 | 90/90 | 12/90 | 51.75% |

The suffix rank was selected conjunctively: it had to pass at both the
tens position used for fitting and the ones position withheld from basis
fitting.

## Rank curves

The leading interface was already compact:

- rank 4: 56/90;
- rank 8: 74/90;
- rank 16: 87/90;
- rank 128: 90/90.

The cross-position suffix test showed a sharper threshold:

- rank 8: 26/90 tens and 9/90 ones;
- rank 16: 65/90 tens and 51/90 ones;
- rank 32: 88/90 tens and 90/90 ones.

All evaluated outputs remained decimal digit tokens. At the selected ranks,
the target advantage over the strongest norm-matched control was 64.44 points
for leading, 88.89 points for tens, and 86.67 points for ones.

## Frozen provenance

- frozen experiment commit: `ef785f0`
- result SHA-256:
  `986c4b11a424ee4ff5233ac30349d1709fc80fca33653d0a5e6367a6a8f72ac9`
- basis artifact SHA-256:
  `33fc5a9dc0e326620769416973a167dc6b370ed18fd1407864f1e539ef449199`
- basis artifact shape: two `428 × 3072` float32 matrices
- basis artifact size: 10,518,704 bytes
- elapsed causal-evaluation time: 762.76 seconds

The large basis matrices remain outside Git; their hash, shapes, fitting
method, input-state hashes, and complete causal rank curves are preserved in
the compact result.

## Decision and claim boundary

Proceed to donor-free coordinate prototypes using rank 8 at hidden index 24
for the leading position and one shared rank-32 suffix basis at hidden index 30
for tens and ones. This stage still uses donor-dependent coefficients and
therefore establishes causal dimensionality, not a deployable interface.
