# Phase 5 Qwen Donor-Free Arithmetic Audit

## Outcome

The fixed Qwen operand interface passed the single authorized audit. The
universal carry interface failed its control-advantage gate, so the combined
Qwen arithmetic-coordinate package is an audit non-pass.

### Operand interface: pass

At hidden-state index 12 and scale 1.0:

- 45/45 target tens digits;
- 43/45 exact target results;
- 0/45 exact for the wrong-class operand prototype;
- 0/45 for the isotropic control;
- a 43/45, or 95.56-point, exact-result advantage;
- 45/45 parseable outputs.

### Universal carry interface: audit non-pass

At hidden-state index 16 and scale 1.6:

- 31/45 target tens digits;
- 28/45 exact target results;
- 21/45 target tens and 16/45 exact for matched no-carry;
- 0/45 for the isotropic control;
- a 10/45, or 22.22-point, tens-position advantage;
- 45/45 parseable outputs.

Carry target accuracy exceeded the 50% threshold, but specificity missed the
fixed 25-point advantage by two quartets. The carry and combined gates fail.

## Cross-family interpretation

The early donor-free operand-edit interface now has sealed-audit support in
both Phi and Qwen, with different model-specific boundaries and scales.

Qwen also reproduces carry-context localization and strong donor-free target
behavior, but its universal vector does not maintain enough separation from
matched no-carry on sealed audit. Workflow portability is partially
established; a cross-family universal carry-writer claim is not.

The exposed audit may not be used to tune or rerun this writer. Any further
Qwen carry work requires a fresh corpus and a conditional writer hypothesis.

## Frozen provenance

- frozen audit target commit: `42bd76e`
- audit configuration SHA-256:
  `9d9b1e38e4b34dd4c5bff39d7bbcc041ed11e329a860d69f1a4f4222d9e08c6f`
- audit result SHA-256:
  `a80d84b8e128be2e17723750010612b0126db36c872fe8bdd33c47f19ff15fa4`
- engine SHA-256:
  `a3b90db856fb6bfea5337872dee42d5a00bd8613c73323d4ad6026fdbb9bd2bf`
- elapsed evaluation time: 35.51 seconds
- repeat invocation: refused before model loading

## Claim boundary

Qwen operand editing is an audited donor-free coordinate. Qwen universal carry
is an audit non-pass. The result does not justify publishing the universal
Qwen carry vector as an audited interface or changing the audit threshold
afterward.
