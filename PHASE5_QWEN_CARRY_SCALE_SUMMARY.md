# Phase 5 Qwen Carry-Context Scale Summary

## Outcome

The fixed Qwen carry-context token at hidden-state index 16 passed the bounded
scale-only follow-up.

Scale 2.0 was the smallest passing scale:

- 26/45 target tens digits;
- 24/45 exact target results;
- 7/45 target tens and 6/45 exact for the matched no-carry regional delta;
- 0/45 for the isotropic control;
- a 19/45, or 42.22-point, advantage over the strongest control;
- a mean intervention norm equal to 0.27% of the full prompt-state norm.

Scale 3.0 also passed but was rejected by the precommitted smallest-passing
rule. Scales 1.0 and 1.5 failed.

## Frozen provenance

- frozen experiment commit: `4cb4f13`
- configuration SHA-256:
  `1178e2701162ec4c436ac13eea22942bb39f17de236fab457c588fad294e396f`
- result SHA-256:
  `e756f1773d4368dab14729b92c6318d053662ac5cad8b15b13bffdf675790cd5`
- elapsed evaluation time: 70.83 seconds

## Decision

Fix index 12 for Qwen operand editing and index 16, scale 2.0 for Qwen carry
context. Fit Qwen-only donor-free transports from the 158 behavior-correct fit
quartets. The passing per-example delta remains a donor-dependent upper bound.
Development and audit causal evaluation remain unopened.
