# Phase 5 Qwen Token-Region Summary

## Outcome

Qwen independently reproduced the generic operand-edit interface but did not
pass the carry-context gate at unit scale.

### Generic operand edit: pass

At hidden-state index 12, patching only the changed first-operand digit token
reached:

- 43/45 target tens digits;
- 43/45 exact target results;
- 0/45 for the isotropic region- and norm-matched control;
- a mean intervention norm equal to 0.26% of the full prompt-state norm.

The same one-token effect reached 42/45 exact at indices 1, 4, and 8. It
declined at indices 16 and 20 and vanished by index 24.

### Carry context: unit-scale non-pass

At its selected hidden-state index 16, the carry-context digit reached:

- 11/45 target tens digits;
- 9/45 exact target results;
- 2/45 target tens and 1/45 exact for the matched no-carry regional delta;
- 0/45 for the isotropic control.

The condition localized a specific effect but failed the precommitted 50%
absolute accuracy and 25-point control-advantage thresholds. The downstream
tail remained at baseline throughout the stack.

## Cross-model interpretation

The workflow rediscovered the same causal ordering with different boundaries:
an early, highly effective operand edit followed by a later carry-context
effect. Qwen's unit-scale carry effect is weaker than Phi's and is not a pass.
No Phi vector, layer, or scale was used.

The selected index 16 may receive a bounded scale-only follow-up. The boundary,
token region, target delta, matched controls, and all thresholds must remain
fixed.

## Frozen provenance

- frozen experiment commit: `88ecdf8`
- configuration SHA-256:
  `223fcbaacada3db5d15ad50d8f69552de9160cf53b56613a3e31d91336cd6939`
- result SHA-256:
  `9ce897d04bed4f7fc5eb5a6633e40358a91c62a53e842fa811b2b5926aa9f029`
- elapsed evaluation time: 370.56 seconds
- selection quartets: 45, unfiltered
- audit and development causal evaluation: unopened
