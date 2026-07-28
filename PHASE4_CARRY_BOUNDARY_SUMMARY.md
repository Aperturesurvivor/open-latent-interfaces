# Phase 4 Carry Prompt-Boundary Summary

## Outcome

The single last-prompt-token carry intervention was a clear non-pass.

Across hidden indices 1, 5, 9, 13, 17, 21, 24, 27, and 30:

- full carry-pair transport reached at most 2/45 target tens digits;
- the carry difference-in-differences residual reached at most 2/45;
- the matched no-carry +1 control also reached 2/45;
- shuffled and random controls reached at most 2/45;
- no target/control advantage appeared.

Both the full-delta and carry-specific selection rules chose index 21 as their
least-bad condition, but neither passed. Exact full-result recovery also peaked
at 2/45.

## Interpretation

Replacing the residual state only at the final `Answer=` prompt token is not a
causally sufficient intervention for carry recomputation. Even a full
donor-recipient delta from the corresponding incremented problem failed.

This differs from late answer-position control. At the prompt boundary, earlier
operand-token representations and their downstream cached key/value states
remain recipient-specific. A last-token patch cannot be interpreted as a full
prompt-state transplant.

The result motivates sequence-wide and token-local causal tracing. It does not
show that carry is absent or non-causal.

## Frozen provenance

- frozen experiment commit: `c93c40a`
- original behavior gate passed: no
- eligible complete-correct fit quartets: 127
- selection quartets: 45, unfiltered
- configuration SHA-256:
  `849c88cdf096a0b8f99de514a4ec425d3cde4266000c40f29e4d431454ca7d1c`
- result SHA-256:
  `8457eb4afe5c487db23ee04306b0d57bdab18a74346bdf71ea726fae87bc1d21`
- elapsed evaluation time: 520.20 seconds

## Decision and claim boundary

Reject the last-prompt-token bottleneck. Preserve the matched quartet design
and move to sequence-wide residual replacement, followed by operand-token
localization. Audit and development causal evaluation remain unopened.
