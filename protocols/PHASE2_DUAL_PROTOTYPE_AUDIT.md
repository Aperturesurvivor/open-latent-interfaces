# Phase 2 Protocol: One-Shot Dual-Prototype Audit

## Authorization

The closed-loop dual-prototype implant passed every frozen development gate at
commit `96fe68f`. This protocol authorizes exactly one audit run after the audit
runner and configuration are committed.

## Immutable inputs

- Model and revision: inherited from the hashed source result.
- Dataset: frozen Phase 2 corpus.
- Evaluation split: audit, 90 examples.
- Target transform: balanced all-digits-changed.
- Leading component: hard-gated causal adapter, hidden index 23, scale 1.0.
- Tens component: shared rank-16 prototype interface, hidden index 27,
  scale 1.25.
- Ones component: shared rank-16 prototype interface, hidden index 27,
  scale 2.0.
- Norm cap: 1.0.
- Conditions and random-control seed: identical to passing development.

The audit configuration contains SHA-256 values for:

- passing development result and configuration;
- dataset configuration;
- adapter result and weights;
- causal basis;
- tens result and prototypes;
- ones result and prototypes;
- deterministic audit targets.

The runner refuses:

- audit without explicit authorization;
- more than one authorized run;
- a non-audit evaluation split;
- an output path different from the frozen path;
- overwrite of an existing audit result;
- any source or target hash mismatch;
- audit if the hashed development result does not pass every conjunctive gate.

## Frozen audit gate

The audit uses the unchanged thresholds:

- exact target result at least 50%;
- every answer position at least 70%;
- exact advantage over every matched control at least 25 points;
- identity preservation at least 90%;
- mean relative norm at most 1.0 at every position;
- parse rate 100%.

## Interpretation rule

- Passing every criterion supports a held-out demonstration of the complete
  donor-free implant under a fourth prompt family.
- Any failed criterion is reported as an audit failure. No post-audit
  hyperparameter, component, threshold, target, or control adjustment may be
  described as part of this audit.

The complete raw result and its hash must be published regardless of outcome.
