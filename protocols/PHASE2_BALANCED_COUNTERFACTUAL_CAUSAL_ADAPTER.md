# Phase 2 Protocol: Balanced Counterfactual Causal Adapter

## Question

Can the learned-basis causal adapter control all three answer digits when every
training and evaluation label is genuinely counterfactual?

Earlier matched-donor labels changed the leading digit while deliberately
minimizing suffix distance. The untouched model consequently matched 53.3% of
development second-digit targets. This experiment removes that label shortcut.

## Frozen target assignment

For each split and answer position independently:

1. create an exactly balanced pool of legal target digits;
2. order examples by original digit and stable example ID;
3. select the first cyclic pool rotation with no original-target equality;
4. reassemble the three independently assigned digits.

Every target is a three-digit synthetic integer. Every target digit differs
from the correct result digit. Fit, selection, and development target hashes
are frozen in the configuration. Target construction is deterministic and
order-invariant.

Native donor activations are not used as target labels. They remain only the
prior causal evidence locating hidden-state index 23 / decoder block 22.

## Frozen adapter and training

- Model and revision: inherited unchanged from the learned-basis result.
- Base-model parameters: frozen.
- Initial adapter: the published learned-basis development checkpoint, verified
  by result and weight hashes.
- Three position-specific online transport ensembles.
- Trainable transport basis and adapter coefficients.
- Four rendered fit prompt families.
- Three causal epochs with learning rate `1e-4`.
- Target-token cross entropy plus identity cross entropy/KL, view consistency,
  norm, and basis-orthogonality losses.
- Scales selected from `{0.5, 1.0}` on selection only.
- Seeds and every loss weight are frozen in the configuration.

## Development evaluation

The development split is evaluated once after all three position checkpoints
and scales are selected. Conditions are:

- untouched base;
- targeted adapter;
- same-digit identity;
- shuffled target, norm matched;
- shuffled state, norm matched;
- random direction, norm matched.

The advancement gate is unchanged:

- exact target result at least 50%;
- every answer position at least 70%;
- exact advantage over every matched control at least 25 points;
- identity preservation at least 90%;
- mean relative norm at most 1.0 at every position;
- parse rate 100%.

All criteria must pass. Otherwise the 90-example audit remains sealed.

## Claim boundary

This is an iterative development experiment motivated by the prior development
diagnosis. It tests whether suffix-label bias caused the observed ceiling; it
is not an audit or an unbiased estimate of final generalization.
