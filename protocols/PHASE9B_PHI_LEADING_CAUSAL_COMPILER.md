# Phase 9B: Phi Leading-Token Causal Compiler

## Purpose

Test whether the leading-token bottleneck can be crossed without another
global prototype and without changing any model weight. The candidate is a
prompt-local compiler at hidden-state index 24. It differentiates the
requested leading-digit margin through the frozen remaining decoder blocks
and applies the minimum-L2 first-order residual update.

This is an output-side causal interface experiment. It is not evidence that a
digit is represented by the computed gradient, nor that the gradient exposes
the model's reasoning.

## Frozen data boundary

Selection reuses only the already exposed Phase 8 selection split:

- dataset configuration SHA-256:
  `2cefb24b966ff9423dbb04be6b97f7633da09b0d64b894302ca8acf856c21aa2`;
- 180 selection example IDs SHA-256:
  `cbead14468b1088afabccded879e82a6b6681d6e0762304b6356fa3b9e7aa461`;
- balanced counterfactual target SHA-256:
  `afdb5503d936dcc4df8bdb3c6fabdf6e5a2ac26b16c9c8e0e99a61d6e6e03b91`.

No Phase 8 audit row may be read, scored, filtered, or used for parameter
choice. Passing this selection can authorize development only. A generalizing
claim requires a new pair-disjoint corpus and one-shot audit.

## Frozen mechanism

- model parameters are frozen before differentiation;
- hidden-state boundary: 24;
- candidates: the tokenizer's ten verified single-token decimal digits;
- competitor: the highest-logit non-target decimal digit per prompt;
- update: requested margin shortfall times the local margin gradient, divided
  by its squared norm;
- hard gate: exact zero whenever the unmodified model's full-vocabulary
  argmax is already the requested digit;
- per-row norm cap: 1.0 times the recipient residual norm;
- desired-margin grid: `1, 2, 4, 8, 12, 16, 24, 32`.

The wrong-digit control compiles a deterministic different leading digit and
is norm-matched to the target update. The random control is independently
seeded and norm-matched per row.

## Frozen selection rule

A desired margin passes only if all of the following hold:

- counterfactual target accuracy at least 0.90;
- identity accuracy at least 0.90;
- target advantage over the stronger norm-matched control at least 0.50;
- target digit-token rate exactly 1.0;
- mean target relative norm at most 0.75.

Among passing margins, select the lowest mean relative norm and then the
smaller margin. If none pass, record the best diagnostic row and close the
hypothesis.

## Reproducibility boundary

The runner, compiler module, source data, behavior result, selection IDs,
targets, thresholds, random seed, and numerical grid are hash-locked in
`configs/phase9b_phi_leading_causal_compiler.json`. The runner refuses to
overwrite an existing result.

## Frozen selection outcome

The one-shot compiler did not pass. The selected diagnostic margin was 32:

- counterfactual target accuracy: `101/180` (`0.5611`);
- identity accuracy: `180/180` (`1.0`);
- wrong-digit norm-matched control: `9/180` (`0.05`);
- random norm-matched control: `1/180` (`0.0056`);
- target advantage over the stronger control: `0.5111`;
- mean target relative norm: `0.1393`;
- target digit-token rate: `1.0`.

The target accuracy threshold of 0.90 was not met. The complete write-once
result is `results/phase9b_phi_leading_causal_compiler_selection.json`, with
SHA-256
`b95083fc2f3f98be473a8c78759b2f2a35c54df6d8a49e02a33d3e444cd895b5`.

The monotonic improvement across the frozen margin grid, strong control
separation, low relative norm, and perfect identity preservation support only
a narrower conclusion: the prompt-local derivative is causally useful, but a
single first-order linearization does not cross the nonlinear leading-token
boundary reliably. This result authorizes an exposed iterative-relinearization
follow-up; it does not authorize an audit.
