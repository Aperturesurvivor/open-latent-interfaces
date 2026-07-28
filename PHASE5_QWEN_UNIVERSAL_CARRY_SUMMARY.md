# Phase 5 Qwen Universal Carry Summary

## Outcome

The first class-invariant Qwen carry-vector grid was a non-pass.

- scale 1.5 reached 20/45 target tens versus 5/45 matched no-carry and 0/45
  random; its 33.33-point advantage passed, but target accuracy missed the
  50% absolute gate by three quartets;
- scale 2.0 reached 29/45 target tens versus 22/45 matched no-carry and 0/45
  random; its absolute accuracy passed, but the 15.56-point advantage failed;
- scale 3.0 saturated both target and matched no-carry behavior;
- scales 0.5 and 1.0 were ineffective.

No predeclared scale passed both accuracy and specificity.

## Interpretation

The fit-weighted universal Qwen direction has a narrow transition between a
specific-but-weak regime and a strong-but-nonspecific regime. This does not
yet establish a universal donor-free Qwen carry coordinate.

A single bounded interpolation follow-up may test scales strictly between 1.5
and 2.0 with the same frozen vector, boundary, token, controls, and gates. If
none pass, the universal-vector hypothesis is rejected and a conditional
donor-free writer is required.

## Frozen provenance

- frozen experiment commit: `2203a95`
- configuration SHA-256:
  `fc15d15b2527813305beb0777c9150b2adca1513c2fb44964a209ee506e5f657`
- result SHA-256:
  `fea0a53210682d40424452733d2523f78baf86a2e350078d39dd74ddb72edd41`
- universal artifact SHA-256:
  `6708309dc0381b71ab08142dd0d4ed5bb3d86cf21b1965b3074e461d1b851e44`
- elapsed evaluation time: 77.27 seconds

## Claim boundary

The Qwen universal vector is a selection non-pass. Development and audit
remain causally unopened.

## Bounded interpolation outcome

The single authorized interpolation follow-up passed at scale 1.6:

- 24/45 target tens digits and 20/45 exact targets;
- 10/45 matched no-carry tens and 8/45 exact;
- 0/45 random;
- a 14/45, or 31.11-point, control advantage.

Scale 1.6 was the smallest passing interpolation. The rebuilt universal
artifact retained the identical SHA-256, proving that only the evaluation
scale changed.

- frozen interpolation commit: `c704437`
- interpolation config SHA-256:
  `e8f8d2e64cca49ba6260d2c2dc087f3d6c4bb6278d6f1b274c543edc0266eba1`
- interpolation result SHA-256:
  `bc5addb816b3bbe9ab5f264cc129ee3e3db4775525dfd4f799b61cc7d2707c6d`
- unchanged universal artifact SHA-256:
  `6708309dc0381b71ab08142dd0d4ed5bb3d86cf21b1965b3074e461d1b851e44`

The first grid remains a non-pass. The bounded interpolation fixes scale 1.6
for untouched development.
