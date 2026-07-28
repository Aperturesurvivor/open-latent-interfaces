# Phase 4 Carry Sequence-Boundary Summary

## Outcome

Full-prompt residual transport exposed a strong causal arithmetic-update route,
but it did not pass the precommitted carry-specific gate.

The complete carry-pair sequence delta reached 39/45 target tens digits and
39/45 exact target results at hidden-state indices 1, 5, and 9. Its selected
boundary was index 5, where the matched no-carry `+1` sequence delta reached
38/45. The full condition therefore passed its absolute accuracy threshold but
failed the required control advantage.

The carry difference-in-differences condition peaked at index 13:

- 29/45 target tens digits;
- 28/45 exact target results;
- 32/45 target tens digits for the matched no-carry `+1` control;
- 1/45 for the isotropic norm-matched random control.

It passed its absolute accuracy threshold but did not outperform the matched
control, so it cannot support a carry-specific coordinate claim.

## Interpretation

Early residual sequences contain a highly effective generic operand-update
route. Transplanting the representation of the same `+1` change from a
no-carry problem into a carry-base recipient usually produces the correct
carry result. The recipient's downstream computation therefore appears able
to determine that the transplanted increment crosses a decimal boundary.

This is stronger than ordinary output steering: the random control remained at
1/45 while matched arithmetic updates reached up to 39/45. It is nevertheless
an operand-update interface, not an isolated carry interface.

The effect persists through index 17 and collapses sharply by index 21. This
places the useful generic update upstream of that transition, while the
carry-specific difference-in-differences result remains unresolved.

## Frozen provenance

- frozen experiment commit: `5b73f5f`
- original behavior gate passed: no
- eligible complete-correct fit quartets: 127
- selection quartets: 45, unfiltered
- configuration SHA-256:
  `d85053e7453a62a80000237f9fbf7418b9fb326e7c141729c7071fb86ea43d22`
- result SHA-256:
  `656a4df1e9f20bd0ba7639d819f1113b3b87c84dc75887832dcec6c7beea603e`
- elapsed evaluation time: 438.62 seconds

## Decision and claim boundary

Reject a carry-specific sequence coordinate under this test. Retain the
generic operand-update route as a causal finding and localize it to the
changed operand token. Then test whether carry-specific effects emerge from
interactions between that token and the recipient's second-operand tokens.
Audit and development causal evaluation remain unopened.
