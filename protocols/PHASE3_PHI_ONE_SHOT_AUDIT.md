# Phase 3 Protocol: Phi One-Shot Audit

## Authorization

The complete donor-free Phi controller passed every frozen closed-loop
development gate at commit `d04dacc`. This protocol authorizes exactly one
evaluation of the untouched 90-example Phase 3 audit split.

## Immutable controller

- model: `microsoft/Phi-3.5-mini-instruct`
- revision: `2fe192450127e6a83f7441aef6e3ca586c338b77`
- assistant prefill: `Answer=`
- leading: rank 32, hidden index 24, scale 1.0
- tens: shared suffix rank 32, hidden index 30, scale 1.25
- ones: shared suffix rank 32, hidden index 30, scale 1.25
- norm cap: one recipient residual norm
- hard gate: exact zero delta when base argmax is the requested digit

The bases and 30 prototype vectors are fit-derived and frozen. No donor state,
model-weight update, or neural coefficient predictor is used at inference.

## Immutable evaluation

- split: audit only
- examples: 90
- prompt family: held-out shelf word problem
- targets: balanced all-digits-changed transform
- conditions: base, donor-free targeted, identity hard-gated, wrong-digit
  norm-matched, shuffled-target norm-matched, and random-in-subspace
  norm-matched
- generation: three greedy digit steps in closed loop

The audit runner verifies its own hash, its shared evaluation engine hash,
every controller artifact and source-result hash, the passing development
result and config, and the audit target hash. It refuses a different output
path or overwrite of an existing result.

## Conjunctive gate

- at least 45/90 exact target results;
- at least 63/90 target digits at every position;
- at least 23/90 exact-result advantage over every matched control;
- at least 81/90 exact original results in the identity condition;
- mean intervention norm no greater than one residual norm at every position;
- 90/90 parseable target outputs;
- 270/270 target output tokens in the decimal digit vocabulary.

Any failed conjunct is an audit failure. The complete result is published
regardless of outcome.

## Claim boundary

A pass supports reproducibility of the native-coordinate discovery and
implementation workflow in a second open model family. It does not establish
direct vector portability between models, internal arithmetic-algorithm
control, or applicability to arbitrary models and tasks.
