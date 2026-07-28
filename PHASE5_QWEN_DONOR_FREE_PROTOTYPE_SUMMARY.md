# Phase 5 Qwen Donor-Free Prototype Summary

## Outcome

The Qwen donor-free operand writer passed. The digit-conditioned carry writer
failed class specificity, independently reproducing the structural pattern
found in Phi.

### Operand writer: pass

At hidden-state index 12 and the smallest passing scale, 1.0:

- 40/45 target tens digits;
- 40/45 exact target results;
- 7/45 target tens but 1/45 exact for the wrong-class prototype;
- 0/45 for the isotropic control;
- 45/45 parseable outputs.

### Carry writer: class-specific non-pass

At hidden-state index 16:

- scale 2.0 reached 28/45 target tens and 26/45 exact;
- matched no-carry reached 16/45 tens and 11/45 exact;
- random remained at 0/45;
- a wrong source-digit carry class reached 30/45 tens and 27/45 exact.

At scale 3.0 the wrong-class carry vector became stronger than the matched
class. No scale passed the complete class-specific gate.

## Interpretation

Qwen fit states independently show that source-digit identity matters for the
early operand edit but not for the later carry transport. The carry writer is
therefore better modeled as a universal direction than four distinct class
vectors. This conclusion did not use Phi activations or artifacts.

## Frozen provenance

- frozen experiment commit: `ee5ac80`
- eligible fit quartets: 158
- fit counts by source digit: 47, 36, 42, and 33
- configuration SHA-256:
  `2799b6e2d33f265ae4f72ab46de22d30ff5eb1c7c4ec982f9fc2ee41b998d4b1`
- result SHA-256:
  `c554f6879d408034b51c29e39a20daaac51ea9da8bf69201fe41b28149dead60`
- prototype artifact SHA-256:
  `a0b5771fff316292f386631bad7bee4566760c0add73635919f734e287e6b224`
- elapsed evaluation time: 216.52 seconds

## Decision

Fix the donor-free Qwen operand writer at index 12 and scale 1.0. Test the
fit-count-weighted universal Qwen carry vector at index 16 against a universal
matched no-carry vector and isotropic control. Development and audit remain
causally unopened.
