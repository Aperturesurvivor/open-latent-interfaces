# Phase 9E Phi Hybrid-Graft Audit Summary

## Headline

The complete read → deterministic compute → hidden-state write pipeline passed
its sole authorized audit on 90 pair- and template-disjoint additions.

The frozen Phi-3.5 Mini model answered `80/90` correctly without intervention.
The latent graft answered `90/90`, recovering all ten base errors while
preserving all 80 base-correct rows. Norm-matched random updates recovered
none of the ten errors.

## Component chain

1. An external semantic locator identified the two operand token spans.
2. A nearest-centroid reader decoded every operand pair from hidden-state
   index 1 without reading token IDs or prompt text.
3. Host integer addition computed every target exactly.
4. A three-step prompt-local causal compiler wrote the leading digit at
   hidden-state index 24.
5. Fixed rank-32 native coordinates wrote tens and ones at hidden-state index
   30.
6. Greedy closed-loop decoding produced all 90 complete answers exactly.

No model weight was trained or changed.

## Controls

- Base: `80/90` true answers.
- Random norm-matched residual updates: `80/90`; `0/10` base errors recovered.
- Shuffled requested targets: `82/90` requested results written exactly and
  `0/90` true answers, demonstrating target-specific control rather than a
  generic arithmetic improvement.
- Oracle versus latent: both `90/90`, so reader errors introduced no gap.

Every answer-position accuracy was `90/90`. This matters because the audit
contained each tens and ones digit exactly nine times, unlike the earlier
carry-base development slice.

## Intervention size

Mean relative update norms were:

- leading: `0.00123`;
- tens: `0.00840`;
- ones: `0.00413`.

These means are descriptive. The frozen mechanism also enforced per-row caps
of 0.75 for the cumulative leading update and 1.0 for each suffix update.

## What is established

For the named Phi model, revision, prompt/output contract, and external
operand locator, a repeatable workflow can:

- read decimal operands from an early residual stream;
- pass their decoded values through a deterministic external arithmetic
  mechanism;
- compile the resulting requested answer back into bounded intermediate
  residual updates;
- generate the exact requested answer on a fresh, balanced corpus.

This is a working deterministic reasoning implant in the operational sense:
the frozen model's intermediate states are used as the input and output
interface around a deterministic mechanism.

## What is not established

- The model did not naturally invoke the external adder.
- The leading compiler is target-conditioned and differentiates the frozen
  output suffix; it is not a discovered universal “math neuron.”
- The experiment does not decode a natural-language chain of thought.
- The external locator supplies operand spans.
- The exact interface has not yet been audited on another model, operation,
  answer length, tokenizer contract, or free-form prompt distribution.
- Workflow portability is now plausible; vector portability is not claimed.

The immutable result SHA-256 is
`d8a2ef793653814d04147050e0bdfaa7bccf7fefaeef066294861c50179ec8cf`.
