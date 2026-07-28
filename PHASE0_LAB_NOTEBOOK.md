# Phase 0 Lab Notebook

## 2026-07-27 — Project initialization

### Research question

Can a frozen pretrained model's existing mathematical representations be read
and written precisely enough to serve as the interface for deterministic
computation?

The initial engineering target was deliberately narrower:

1. capture residual states from a frozen Hugging Face model;
2. distinguish addition requests from matched contrasts;
3. decode result information across held-out operand pairs;
4. move the first result digit through a probe-defined residual intervention;
5. retain every failure and distinguish pilot selection from confirmation.

### Environment

- Hardware: Apple M4, 16 GB unified memory
- Runtime: Python 3.12, PyTorch 2.13, MPS
- Target: `Qwen/Qwen2.5-0.5B-Instruct`
- Resolved model revision:
  `7ae557604adf67be50417f59c2c2f167def9a775`
- All target-model parameters frozen

The local machine cannot reproduce Anthropic's published NLA training recipe,
which used multi-H100 systems. The project therefore starts with lightweight
latent cartography and intervention while treating released NLAs as optional
future backends.

## Smoke v0 — range extrapolation and tokenizer failure

Artifact:
`results/phase0_qwen05b_smoke_v0_range_extrapolation.json`

The first generator assigned non-overlapping numeric ranges to train,
development, and test. Routing was perfectly separable in the small sample,
but scalar-sum regression failed badly:

- hidden-state index 12 test R²: -162.83;
- hidden-state index 24 test R²: -198.08;
- rounded exact recovery: 0 at both boundaries.

The causal stage did not run because Qwen tokenizes multi-digit numbers as
individual digit tokens, not one result token.

This was a protocol-design failure, not evidence that result information is
absent. The artifact is retained. The corrected pilot uses exact-pair-disjoint
splits over shared numeric support and names the causal endpoint as the first
result digit.

## Smoke v1 — scalar probe write diagnostic

Artifact:
`results/phase0_qwen05b_smoke_v1_scalar_interpolation.json`

With pair-disjoint splits over shared support:

- routing remained highly separable;
- hidden-state index 12 development scalar-sum R² reached 0.33, but test R² was
  -0.29;
- a minimum-norm shift along the scalar probe did not beat the shuffled-result
  control;
- top-1 first-digit accuracy stayed 4/6 in all conditions.

The scalar decoder was a poor write interface for a categorical first-token
endpoint. The next implementation added a ten-class leading-digit probe and a
minimum-norm pairwise-margin intervention.

## Smoke v2 — categorical leading-digit bridge

Artifact: `results/phase0_qwen05b_smoke.json`

This engineering smoke used 24 training pairs, eight development pairs, and
eight test pairs at hidden-state indices 12 and 24.

At the development-selected index 24:

- frozen base first-digit top-1: 4/8;
- targeted categorical intervention at strength 2: 6/8;
- shuffled-result intervention: 3/8;
- targeted correct-token margin change: +1.234 logits;
- targeted-minus-shuffled margin change: +1.307 logits.

The result justified a larger pilot, but the sample was too small and strength
was selected on the same test examples. No claim was made.

## Full local pilot

Artifact: `results/phase0_qwen05b_pilot.json`

Code commit recorded in the artifact:
`463f6f3d983ad0a3db10f2d0064659b6457f684c`

Dataset:

- 96 training operand pairs;
- 32 development pairs;
- 32 test pairs;
- four matched semantic conditions per pair;
- 640 prompts total;
- exact operand pairs disjoint across splits.

Observed mapping:

- The route probe reached 100% test accuracy and AUC at all five inspected
  boundaries. This is not a robust-routing result because the prompt families
  contain easy lexical distinctions and were not family-held-out.
- Scalar sum information was linearly decodable but imprecise. The best test
  R² was 0.915 at hidden-state index 5 with MAE 49.14 and 0/32 rounded exact.
- Leading-digit decoding peaked at 17/32 test examples at index 24. The layer
  was selected by development accuracy.

Causal first-digit pilot at index 24:

| Condition | Top-1 | Mean target rank | Margin change |
|---|---:|---:|---:|
| Frozen base | 15/32 | 3.16 | — |
| Targeted, strength 2 | 16/32 | 2.72 | +0.045 |
| Shuffled result, strength 2 | 14/32 | 3.09 | +0.005 |

The targeted-minus-shuffled margin difference was +0.040 logits. This is a
weak positive pilot signal. It is much smaller than the eight-example smoke,
changes only one top-1 decision over base, operates at the final block output,
and uses test-set strength selection. It does not establish a native
mathematical write channel or a deterministic graft.

## Current interpretation

Phase 0 passed as infrastructure and failed as a scientific demonstration of a
reliable latent graft.

What is supported:

- the repository can run a frozen-model activation study end to end on local
  Apple Silicon;
- addition intent is easily decodable in the present dataset;
- approximate scalar result information exists across multiple layers;
- a probe-defined digit intervention produces a small targeted advantage over a
  shuffled-label control in this pilot.

What is not supported:

- exact operand or result decoding;
- causal use of the scalar-sum direction;
- an internal rather than output-adjacent write interface;
- template-general routing;
- multi-token arithmetic correction;
- NLA faithfulness or portability;
- an end-to-end deterministic graft.

## Next engineering gate

Before a frozen audit:

1. balance leading-digit classes and increase development support;
2. add template-family-held-out route and value evaluation;
3. run causal write sweeps at every candidate layer, not only the
   development-selected decoder layer;
4. add donor activation patching and logit/Jacobian directions as baselines;
5. compare scalar, per-digit, and low-rank nonlinear read/write bridges;
6. select all hyperparameters on development only;
7. freeze a new prompt- and pair-disjoint audit;
8. integrate a released NLA/J-lens backend after the causal harness is stable.

