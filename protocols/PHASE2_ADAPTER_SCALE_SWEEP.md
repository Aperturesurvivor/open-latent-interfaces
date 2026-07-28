# Phase 2 Protocol: Fixed-Weight Adapter Scale Sweep

## Question

Is the balanced-counterfactual adapter's tens failure primarily caused by
insufficient intervention amplitude?

The existing weights remain fixed. This experiment selects only one scalar per
answer position and therefore cannot add representational capacity.

## Frozen inputs

- Source result and weights: balanced-counterfactual development checkpoint,
  verified by SHA-256.
- Model, revision, hidden-state index 23, and decoder block 22: unchanged.
- Dataset and balanced all-digits-changed target hashes: unchanged.
- Base-model and adapter parameters: frozen.
- Audit authorization: false.

## Selection

The scale grid is:

`0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0`

For each answer position, the selection split measures teacher-forced target
token accuracy and same-digit identity accuracy. Candidates whose mean target
or identity relative norm exceeds 1.0 are ineligible.

Among eligible candidates, choose lexicographically by:

1. maximum of the smaller target/identity accuracy;
2. target accuracy;
3. identity accuracy;
4. lower target relative norm;
5. lower identity relative norm.

No development metric participates in scale selection.

## Development evaluation

After all scales are fixed, run the same closed-loop conditions:

- untouched base;
- targeted adapter;
- same-digit identity;
- shuffled target, norm matched;
- shuffled state, norm matched;
- random direction, norm matched.

The existing advancement gate is unchanged:

- exact target result at least 50%;
- every answer position at least 70%;
- exact advantage over every matched control at least 25 points;
- identity preservation at least 90%;
- mean relative norm at most 1.0 at every position;
- parse rate 100%.

All gates must pass before any audit run.

## Decision rule

- If higher selected tens amplitude materially improves target accuracy while
  remaining within preservation and norm gates, amplitude was a limiting
  factor and the scaled checkpoint becomes the next baseline.
- If tens accuracy remains low near the norm ceiling, the compressed direction
  or write boundary—not amplitude alone—is limiting. The next experiment then
  maps layer and rank for the tens position only.

This is iterative development diagnosis, not an audit.
