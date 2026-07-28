# Phase 4 Carry-Quartet Behavior Summary

## Outcome

The original conjunctive behavior gate did not pass:

| Split | Correct rows | Complete quartets | Row gate | Quartet gate | Overall |
| --- | ---: | ---: | --- | --- | --- |
| Fit | 641/720 (89.03%) | 127/180 (70.56%) | fail | fail | fail |
| Selection | 169/180 (93.89%) | 36/45 (80.00%) | pass | pass | pass |
| Development | 172/180 (95.56%) | 37/45 (82.22%) | pass | pass | pass |

All 1,080 responses were parseable and followed the frozen three-digit token
contract. No audit row was evaluated.

The largest fit weakness was the carry-increment condition: 139/180 exact,
versus 169/180 carry bases, 167/180 control bases, and 166/180 control
increments.

## Decision

Do not regenerate the corpus or search for a favorable seed. Preserve the
original non-pass.

For fit-only representation estimation, a declared eligibility filter may use
the 127 quartets for which all four untouched answers were correct. Selection
and development remain unfiltered. Any causal result must report both all-row
metrics and behavior-correct strata; the audit remains sealed.

This amendment changes the role of fit data, not the frozen dataset, selection
examples, development examples, targets, or future controls.

## Frozen provenance

- frozen corpus and gate commit: `8f57d92`
- dataset SHA-256:
  `daa08c5e572676046bc3f8e27abcaab249a59dffc5965ba6729f491b098d4574`
- configuration SHA-256:
  `bc0740b9b106f6360573bf4c2828cdae1c2418a80d61c54cffea3d84f553f075`
- result SHA-256:
  `6e74201cce34f283b044c9245b82455e3082873bb7a3f9e6cb72151c90d713a5`
- elapsed evaluation time: 226.03 seconds

## Claim boundary

This is a behavioral screening result. It provides no evidence for a latent
carry representation. The original gate failure must remain visible in every
downstream Phase 4 summary.
