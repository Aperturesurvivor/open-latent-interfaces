# Phase 4 Carry Token-Region Summary

## Outcome

The frozen token-region selection experiment passed both the generic
operand-edit gate and the carry-context gate.

### Generic operand edit

Patching only the changed first-operand digit token produced:

- 39/45 target tens digits and 39/45 exact target results at hidden-state
  index 1;
- 0/45 target tens digits for the isotropic region- and norm-matched control;
- a mean intervention norm equal to 11.40% of the full prompt-state norm.

The same one-token edit retained 38/45 exact at index 5 and 39/45 at index 9.
This is a compact causal interface for changing the represented operand by
one while allowing the frozen downstream model to recompute the answer.

### Carry-context localization

Patching only the second operand's carry-relevant ones digit at index 13
produced:

- 32/45 target tens digits;
- 31/45 exact target results;
- 17/45 target tens digits and 16/45 exact for the matched no-carry `+1`
  regional delta;
- 1/45 for the isotropic region- and norm-matched control;
- a mean intervention norm equal to 0.30% of the full prompt-state norm.

The carry-context condition exceeds its matched arithmetic control by 15/45
target tens digits and passed the precommitted absolute and advantage gates.

### Downstream tail

Patching every token after the carry-context digit remained at the 1/45
baseline at all tested boundaries. The selected causal effect is therefore
localized to the contextualized second-operand digit rather than copied into
the later instruction or assistant-prefill tail under this intervention.

## Interpretation

The results separate two interfaces:

1. an early generic operand-edit coordinate at the exact changed input digit;
2. a later carry-specific interaction at the point where the model has seen
   both operands' ones digits.

The second location is structurally appropriate for a causal decoder: the
second-operand digit can attend to the earlier changed digit, whereas the
changed digit cannot yet see the second operand. Its advantage over both a
matched no-carry update and random control is causal evidence for
carry-specific computation.

The intervention is still donor-dependent. It identifies a token, boundary,
and causal state difference, not an individual neuron, natural-language
thought, or donor-free carry writer.

## Frozen provenance

- frozen experiment commit: `adf4a91`
- original behavior gate passed: no
- eligible complete-correct fit quartets: 127
- selection quartets: 45, unfiltered
- configuration SHA-256:
  `9ea31f582352d3bef84976aef055ee8d958143dca72c07a5039e07de065b076b`
- result SHA-256:
  `9f85011d1f5c575927f837d4e2317127a1176b40b3707e0ed5b604015a7078ef`
- elapsed evaluation time: 350.20 seconds

## Decision and next gate

Fix hidden-state index 1 as the generic changed-operand interface and index 13
as the carry-context boundary. Use behavior-correct fit quartets only to
estimate compact donor-free bases at those exact token positions. Select rank
and writer form on the same unfiltered selection split, then validate once on
untouched development. Audit remains sealed.
