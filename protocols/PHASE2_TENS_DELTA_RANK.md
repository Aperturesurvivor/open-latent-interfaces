# Phase 2 Protocol: Tens Native-Delta Rank

## Question

How much of the native tens transport subspace must be retained to preserve
causal control?

Native replacement at hidden-state index 27 reached 90/90 development tens
digits. The compact adapter uses a rank-64 output. This experiment determines
whether rank 64 itself is too small or whether the adapter fails mainly when
predicting coefficients.

## Frozen basis fit

- Boundary: hidden-state index 27 / decoder block 26.
- Fit recipients: 450 Phase 2 fit examples.
- Targets: balanced all-digits-changed synthetic results.
- Donors: deterministic fit-pool examples matching each target's
  leading-plus-tens prefix.
- Context: recipient and donor prompt plus target leading digit.
- Delta: native donor residual minus recipient residual.
- Basis: uncentered SVD of the 450 fit deltas.

The maximum fitted rank is 450. The basis, fit states, and fit deltas receive
content hashes.

## Selection rank sweep

Project exact selection donor deltas into ranks:

`8, 16, 32, 64, 128, 256, 450`

For every rank, evaluate:

- no intervention;
- projected targeted donor delta;
- projected wrong-tens donor delta, norm matched;
- shuffled projected coefficients, norm matched;
- random direction inside the same subspace, norm matched.

A rank passes if:

- targeted tens accuracy is at least 70%;
- target accuracy exceeds every matched control by at least 25 points;
- mean relative norm is at most 1.0;
- every targeted top-1 token is a digit.

Select the smallest passing rank. If none passes, select by target accuracy,
control advantage, target margin, then lower rank.

## Development

Evaluate only the selected rank once on development with the same conditions.
The full native result is the fixed upper bound.

## Decision rule

- If rank 64 passes, output dimensionality is sufficient and the next effort
  should improve coefficient prediction.
- If a larger rank is required, train a tens-specific writer at the smallest
  passing rank before attempting a general multi-position adapter.
- If rank 450 fails, the fit PCA distribution does not span transferable native
  control and direct residual optimization becomes the next oracle.

Projected donor deltas still require donor-derived coefficients. This is a
representation diagnostic, not a donor-free implant or an audit.
