# Model Onboarding

## Purpose

Model onboarding answers a narrow question before expensive latent-interface
discovery begins: can the frozen model and tokenizer support the observation,
token contract, residual intervention, and local-gradient operations required
by this repository?

It does not test whether a useful reader or writer exists.

## Specification

An `oli.model-onboarding/v1` file pins:

- model repository and exact 40-character revision;
- expected Hugging Face architecture metadata;
- one non-selection arithmetic prompt used only for compatibility;
- the `Answer=` prefix and fixed-width decimal token contract;
- seven mandatory instrumentation gates;
- optional hash-bound historical evidence.

Audited reference specifications are provided for Phi and Qwen:

- `configs/model_onboarding_phi35_mini_audited.json`
- `configs/model_onboarding_qwen25_15b_audited.json`

## Metadata preflight

The metadata mode loads only configuration and tokenizer files:

```bash
oli-model-onboarding \
  configs/model_onboarding_qwen25_15b_audited.json \
  --output /tmp/qwen-onboarding-metadata.json \
  --metadata-only
```

It checks pinned architecture metadata, chat-prefill rendering, fast-tokenizer
offsets, operand locations, distinct contextual decimal tokens, fixed-width
composition, and deterministic candidate reader boundaries.

## Live preflight

The live mode additionally loads and freezes the model:

```bash
oli-model-onboarding \
  configs/model_onboarding_qwen25_15b_audited.json \
  --output results/model_onboarding_qwen25_15b_live.json \
  --device mps \
  --dtype float16
```

It verifies the decoder-block path, block count, residual width, hidden-state
indexing convention, next-token vocabulary, frozen parameters, and a finite
nonzero decoder-block-to-digit-margin gradient.

The output path is write-once. A failed preflight is preserved with its error
type and message.

## Interpretation

A pass means the generic instrumentation can begin model-specific discovery.
It does not authorize reuse of another model's layers or tensors and is not a
latent-interface result. Reader selection, writer discovery, integrated
development, and a fresh one-shot audit remain separate advancement stages.

See
`protocols/PHASE13_MODEL_ONBOARDING_AND_THIRD_FAMILY.md`
for the prospective third-family rules.
