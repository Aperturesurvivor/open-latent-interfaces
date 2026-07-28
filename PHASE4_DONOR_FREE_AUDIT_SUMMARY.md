# Phase 4 Donor-Free Arithmetic-Coordinate Audit

## Outcome

Both fixed donor-free interfaces passed the single authorized audit run on 45
sealed, unfiltered carry quartets.

### Operand-edit interface

At hidden-state index 1 and fixed scale 1.5:

- 38/45 target tens digits;
- 37/45 exact target results;
- 0/45 exact for the wrong-class operand prototype;
- 2/45 exact for the isotropic norm-matched control;
- a 35/45, or 77.78-point, exact-result advantage over the strongest control;
- 45/45 parseable outputs.

### Universal carry interface

At hidden-state index 13 and fixed scale 1.0:

- 26/45 target tens digits;
- 25/45 exact target results;
- 11/45 target tens digits for the matched no-carry vector;
- 2/45 for the isotropic norm-matched control;
- a 15/45, or 33.33-point, tens-position advantage over the strongest control;
- 45/45 parseable outputs.

The carry writer is one class-invariant direction. Its mean intervention norm
was 0.18% of the full prompt-state Frobenius norm. The operand writer's mean
relative norm was 14.56%.

## Provenance and one-shot enforcement

- frozen audit target commit: `b25351b`
- audit configuration SHA-256:
  `c7856a84020b2390e70874364aa3847abc0930d77b032aba4f4c0f4da757c8c6`
- audit result SHA-256:
  `c66f97b794e2be62fc5b3cbe46a2ca6e08ddc4593ca80aebfe709456d8d9efc9`
- engine SHA-256:
  `55657da21c72fa65516b9d3c8d4e86451d4c6068c7f15a7ceac732d67739e30d`
- elapsed evaluation time: 66.47 seconds
- repeat invocation: refused because the frozen output already existed
- original Phase 4 behavior gate passed: no
- corrected development package passed: yes

## What this establishes

For frozen Phi-3.5 Mini under this arithmetic distribution:

1. an early token-local coordinate can edit a represented operand without a
   donor activation or model-weight update;
2. a later, rank-one, class-invariant coordinate at the contextualized second
   operand digit causally controls carry-linked tens behavior;
3. both effects generalize through untouched development and a sealed
   one-shot audit against matched controls.

## Claim boundary

These are causal writable coordinates, not identified individual neurons or
natural-language thought transcripts. The audit covers the frozen model,
prompt/token contract, four source-digit classes, and matched three-digit
addition distribution. Broader arithmetic operations, tokenizers, templates,
and model families require independent mapping and audit.
