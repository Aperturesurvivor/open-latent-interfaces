# Phase 3 Phi Prototype-Selection Summary

## Outcome

The first donor-free prototype selection was a partial pass.

| Position | Rank | Selected scale | Target digit | Identity digit | Mean relative norm | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Leading | 8 | 1.25 | 54/90 | 90/90 | 49.30% | fail |
| Tens | 32 | 1.25 | 90/90 | 90/90 | 66.50% | pass |
| Ones | 32 | 1.25 | 90/90 | 90/90 | 57.06% | pass |

The leading gate required at least 63/90 target digits. It was not relaxed.
The suffix prototypes passed at every scale from 1.0 through 2.0, with 100%
target and identity accuracy.

## Interpretation

Rank 8 retained donor-dependent leading transports at 74/90, but replacing
those coefficients with one fit-derived class mean per digit reached only
54/90. Therefore, low causal transport rank does not imply that a single
rank-8 class prototype captures recipient variation.

The result motivates a bounded leading-rank follow-up. It does not motivate
refitting the successful suffix interface: the shared rank-32 basis and scale
1.25 are locked for closed-loop development.

## Engineering note

An initial invocation stopped before writing an artifact or result because the
fitter incorrectly required a leading-digit-0 class. Three-digit answers have
leading classes 1–9. Commit `2499af7` corrected that class-domain invariant
without changing data, scales, gates, bases, or evaluation conditions.

## Frozen provenance

- frozen experiment commit: `7670d8b`
- implementation-invariant fix: `2499af7`
- configuration SHA-256:
  `713649ac12758d92e23305e685ca2ce1649cf11ceada598db786011f73650b5f`
- result SHA-256:
  `891ae9a9228455f76c00ab4220ceb3a8de58e41f3033a1ee336f9f0f3efb4a44`
- prototype artifact SHA-256:
  `cae12daf7b94a9c685e00244b1e039c2d80267ce660a9af9e859518bd983b552`
- prototype artifact size: 3,584 bytes
- elapsed successful-run time: 353.54 seconds

## Claim boundary

This is selection-only evidence. It establishes a donor-free suffix interface
on Phi but not a complete three-position controller. Development and audit
remain untouched.
