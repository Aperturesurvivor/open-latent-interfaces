# Phase 9D Hybrid Graft Gate Correction

## Scope

This correction operates only on the immutable Phase 9D development result.
It performs no model inference, regenerates no tokens, filters no row, and
changes no writer parameter.

## Structural mismatch

The removed check required wrong-target true accuracy at most 0.25. The
control was built by norm-matching wrong-target directions to the correct
target's hard-gated update. On a row where the model already emits the correct
token, the correct update has norm zero. Norm matching therefore gives the
wrong-target control norm zero as well, preserving the base answer.

Consequently, aggregate wrong-target true accuracy is dominated by the base's
38 already-correct rows. It does not answer the intended control question:
whether an equal-magnitude wrong direction repairs the same base failures as
the requested direction.

## Frozen replacement

The correction retains every original check except
`wrong_target_control`. It replaces that check with:

- wrong-target recovery of the seven base errors at most 0.25;
- latent recovery advantage over wrong-target recovery at least 0.50.

All quantities are recomputed from the original paired rows. The correction
runner verifies that the original result failed exactly and only the removed
check before it will emit a result.

## Claim boundary

A corrected pass can authorize a new, independently generated audit protocol.
It cannot convert this exposed development split into audit evidence, and the
original non-pass remains authoritative under its original thresholds.

## Correction outcome

The no-rerun correction passed. The immutable paired measurements were:

- latent base-error recovery: `7/7` (`1.0`);
- wrong-target base-error recovery: `0/7` (`0.0`);
- latent recovery advantage: `1.0`;
- wrong-target preservation of base-correct rows: `38/38` (`1.0`).

Every retained original check and both replacement checks passed. The
correction result is
`results/phase9d_phi_hybrid_graft_gate_correction.json`, SHA-256
`07038f7f834774df0f505e8a9b85859fb1c752d01fc5e8c6251e698f581e757b`.

