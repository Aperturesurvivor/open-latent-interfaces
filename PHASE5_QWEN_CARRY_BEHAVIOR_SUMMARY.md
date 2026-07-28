# Phase 5 Qwen Carry-Behavior Summary

## Outcome

Qwen2.5-1.5B passed every frozen non-audit behavior gate on the matched carry
quartets.

- fit: 688/720 exact rows and 158/180 complete-correct quartets;
- selection: 171/180 exact rows and 39/45 complete-correct quartets;
- development: 172/180 exact rows and 41/45 complete-correct quartets;
- parse rate: 100% in every split;
- digit-token contract: all ten decimal digits are distinct single tokens.

The behavior-correct fit pool contains 158 quartets. Source-digit class counts
are 47, 36, 42, and 33 for digits 1 through 4.

## Frozen provenance

- frozen experiment commit: `0f65219`
- model revision:
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- dataset SHA-256:
  `daa08c5e572676046bc3f8e27abcaab249a59dffc5965ba6729f491b098d4574`
- configuration SHA-256:
  `38b8df36ad8a7c017c89b5b556dfdc92f9aed535949c9d8e9903a31f9c6c62e8`
- result SHA-256:
  `37c82c255a80d44b8f272a8f6211fae9c37279df9838f649f4416b6fbe8f8d74`
- eligible fit quartet SHA-256:
  `2b61a7dd95c97202e9bf26d1fb8a16b33e117d0e5b8c8334127a97f49d160f1d`
- elapsed evaluation time: 118.18 seconds

## Decision

Advance to Qwen-specific causal boundary and token-region mapping. No Phi
layer, vector, scale, or fitted artifact may be reused. Audit remains sealed.
