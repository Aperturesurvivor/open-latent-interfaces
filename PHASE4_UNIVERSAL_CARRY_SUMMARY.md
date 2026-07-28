# Phase 4 Universal Carry Summary

## Original outcome

The class-invariant donor-free carry vector produced strong target behavior,
but the original precommitted scale selector returned a non-pass.

At scale 1.5, which the scorer selected for its 39/45 target tens digits and
37/45 exact results, the matched no-carry control also rose to 33/45 tens and
31/45 exact. The required 25-point control advantage therefore failed.

The original result and its `passes: false` status are preserved.

## Selection-rule defect

The same frozen run had already evaluated scale 1.0:

- 32/45 target tens digits;
- 30/45 exact target results;
- 18/45 target tens for the matched no-carry control;
- 1/45 for the isotropic control;
- 45/45 parseable outputs.

Scale 1.0 exceeded the original 50% target threshold and the strongest control
by 14/45, or 31.11 percentage points. It satisfied every numerical gate. The
scorer nevertheless chose scale 1.5 because its declared ordering prioritized
raw target accuracy before whether a scale passed the conjunction.

This is a rule-selection defect, not authorization to relabel the run. A
separate bounded correction is required.

## Frozen provenance

- frozen experiment commit: `e385b2e`
- configuration SHA-256:
  `af3a6f972f3e5483e071e3f8bb94d9f3d54ef73e8840b7b475b76c901df2e188`
- result SHA-256:
  `a4b151f5b1dee6995cb5038c861afbada3f252b9abcd6c46138da4851b378929`
- universal prototype SHA-256:
  `f4f7488e232b45c49567d02e7ea1bfffe85086542e94afc4ccee1e38cbd6bdd3`
- elapsed evaluation time: 89.69 seconds

## Claim boundary

The original universal selection is a non-pass. The artifact is a single
donor-free direction and is therefore already rank one as a writable
coordinate, but development use is unauthorized until the bounded scale
correction is frozen and executed.

## Bounded correction outcome

The no-rerun correction was frozen at commit `3b4cdcb`. It audited only the
three original scales under the original thresholds. Scale 1.0 was the sole
passing scale and is fixed for development; scales 0.5 and 1.5 failed.

- correction result SHA-256:
  `aa26f80978c20a3baca7f64c00df2a37bc8f8288654ca46a2e7d8a0c8acc77ba`
- new model inference: none
- new data, scales, fitted weights, or thresholds: none

This correction does not change the original result's non-pass status. It
authorizes the predeclared scale-1.0 candidate for untouched development under
a separately frozen validation gate.
