# Phase 4 Donor-Free Development Summary

## Original outcome

The fixed rank-one universal carry coordinate passed untouched development.
The fixed donor-free operand writer narrowly failed its precommitted tens-only
control-advantage gate, so the original conjunctive development result is a
non-pass.

### Universal carry: pass

At hidden-state index 13 and fixed scale 1.0:

- 29/45 target tens digits;
- 29/45 exact target results;
- 16/45 for the matched no-carry vector;
- 4/45 for the isotropic control;
- a 13/45, or 28.89-point, advantage over the strongest control;
- 45/45 parseable outputs.

This independently validates a single class-invariant, donor-free carry
direction on untouched development.

### Operand writer: original non-pass

At hidden-state index 1 and fixed scale 1.5:

- 42/45 target tens digits;
- 41/45 exact target results;
- 31/45 target tens digits but 0/45 exact for the wrong-class prototype;
- 4/45 for the isotropic control;
- 45/45 parseable outputs.

The tens-only advantage is 11/45, or 24.44 points, one quartet below the
precommitted 25-point threshold. The original operand and conjunctive statuses
remain non-passes.

## Metric defect

The wrong-class operand control is designed to change the source digit to an
incorrect target. It can preserve the correct hundreds and tens characters
while making the ones character—and therefore the full result—wrong. That is
exactly what happened: 31 wrong-class outputs matched the target tens digit,
but none matched the target result.

Tens accuracy is the correct primary metric for the carry interface, whose
causal target is the tens transition. It is not a sufficient discrimination
metric for a full operand edit. A bounded no-rerun correction may audit the
already written operand outputs using exact-result accuracy under the same
70% absolute and 25-point advantage thresholds. The original development
result cannot be relabeled.

## Frozen provenance

- frozen experiment commit: `812a980`
- development quartets: 45, unfiltered
- configuration SHA-256:
  `ebfcb37694f854ce62efa7df6d1b0ae96413e5d87017c5780c3416e4f36ab426`
- result SHA-256:
  `ec524cd83a7823ddcadf8640f204b6ea1f448f4764c5ff671ccc8efd93b0c94d`
- elapsed evaluation time: 62.42 seconds

## Claim boundary

The universal carry coordinate has a one-shot untouched-development pass. The
operand writer and combined development package retain their original
non-pass status pending a separately frozen semantic metric correction. Audit
remains sealed.

## Bounded correction outcome

The metric correction was frozen at commit `91c1153` and performed no model
inference. Under exact-result accuracy, the operand target reached 41/45
against 4/45 for the strongest exact control, an 82.22-point advantage. The
unchanged carry gate also passed, so the corrected development package passes.

- correction result SHA-256:
  `8fd2b91d89fcb266ba20e7edd50a1e9c671be26c0b3ae88564c9ca1947e12555`
- original development result status: unchanged non-pass
- corrected operand gate: pass
- unchanged carry gate: pass
- new model inference, data, weights, scales, or thresholds: none

This correction authorizes a separately committed one-shot audit package using
exact-result discrimination for the operand interface and tens discrimination
for the carry interface.
