# Phase 13: Model Onboarding and Third-Family Replication

## Purpose

Convert the successful Phi and Qwen arithmetic-graft replications into a
prospective model-onboarding contract, then exercise that contract on a third
decoder architecture without transferring any fitted tensor or
model-specific boundary.

Phase 13 separates three claims that must not be conflated:

1. **instrumentation compatibility** — the frozen model can be observed and
   intervened on by the workflow;
2. **interface discovery** — model-specific readers and writers pass exposed
   selection and development gates;
3. **audited graft** — one newly sealed dataset passes the full causal gate.

A compatibility pass supports only the first claim.

## Universal compatibility gates

`oli.model-onboarding/v1` requires:

- a fast tokenizer with exact offset mappings;
- a chat template that supports a continued assistant prefix;
- ten distinct contextual one-token decimal digits after `Answer=`;
- exact composition of fixed-width results from those contextual digit tokens;
- a discoverable repeated decoder-block stack;
- one embedding state plus one hidden state per decoder block;
- stable residual width across those hidden states;
- a finite nonzero gradient from a decoder-block output to a requested
  next-token digit margin;
- a fully frozen parameter set during gradient probing.

The preflight records observed architecture, width, layer count, vocabulary,
decoder path, token IDs, operand positions, candidate reader boundaries, and
the gradient-probe boundary and norm.

## Audited regression references

The compatibility contract must first pass on the exact revisions already
covered by successful one-shot graft audits:

- `microsoft/Phi-3.5-mini-instruct`
  (`Phi3ForCausalLM`, 32 blocks, width 3072);
- `Qwen/Qwen2.5-1.5B-Instruct`
  (`Qwen2ForCausalLM`, 28 blocks, width 1536).

These models are regression references, not third-family candidate-selection
data. Their linked manifests and audit hashes establish that a compatibility
pass was followed by successful discovery in those historical cases; they do
not make compatibility sufficient.

## Live regression outcome

Both audited references passed all 15 observed metadata and live-model checks
under the frozen onboarding implementation:

### Phi-3.5-mini

- result: `results/model_onboarding_phi35_mini_live.json`
- result SHA-256:
  `b29354c83e5023e380ab3a52f33060b440de5631b125d7f70d8b66e9519da476`
- decoder path: `model.layers`
- decoder blocks / hidden states: 32 / 33
- residual width: 3072
- contextual digit IDs: 29900, 29896, 29906, 29941, 29946, 29945,
  29953, 29955, 29947, 29929
- mid-stack gradient probe: hidden-state index 16, norm 5.5180

### Qwen2.5-1.5B

- result: `results/model_onboarding_qwen25_15b_live.json`
- result SHA-256:
  `08823a15819b4c8df227ce2dd2bf326d5e7e69eb0cd20e7c45ea5a4a79d17c46`
- decoder path: `model.layers`
- decoder blocks / hidden states: 28 / 29
- residual width: 1536
- contextual digit IDs: 15 through 24
- mid-stack gradient probe: hidden-state index 14, norm 3.4550

Every parameter was frozen, both operand locators resolved each decimal
character to one token, and fixed-width results composed exactly. This
regression validates the onboarding instrumentation against two models with
independently passing downstream audits.

## Third-family candidate rule

Before task-specific discovery, the candidate must be frozen by exact
repository revision and satisfy all of the following:

- publicly downloadable, ungated weights and tokenizer;
- an explicit open-source license compatible with redistribution of project
  metadata and independently derived interface tensors;
- no more than four billion parameters for feasible local selection and audit;
- a Hugging Face `model_type` distinct from `phi3` and `qwen2`;
- an instruction-tuned causal decoder with a supported chat template;
- all universal compatibility gates pass without modifying model code.

Candidate ordering and the evidence used to select it must be committed before
any task-specific activation capture. A failed compatibility preflight closes
that candidate without parameter or prompt tuning.

## Candidate selection and OLMo closure

The ordered candidate selection was frozen in
`configs/phase13_third_family_candidate_selection_frozen.json`:

1. `allenai/OLMo-2-0425-1B-Instruct`, revision
   `48d788eca847d4d7548f375ad03d3c9312f6139e`;
2. `HuggingFaceTB/SmolLM2-1.7B-Instruct`, revision
   `31b70e2e869a7173562077fd711b654946d38674`.

OLMo ranked first because its `olmo2` architecture provides greater diversity,
its 1.485-billion-parameter size is smaller, and its Apache-2.0 release
includes extensive public training artifacts.

OLMo closed at metadata preflight:

- result: `results/model_onboarding_olmo2_1b_candidate_metadata.json`
- result SHA-256:
  `1f394aa3cf2a22e85c610184c268133dd673b033f2314ace05c7a9f4151a0227`
- no model weights were loaded;
- individual digits were stable one-token continuations;
- tested three-digit strings were merged into single tokens instead of
  composing from those digit tokens (`100` → 1041, `237` → 14590,
  `580` → 18216, `999` → 5500).

This is a real incompatibility with the frozen three-step decimal writer
contract. The contract is not relaxed after observing the result. OLMo is
closed for this replication, while tokenization-general grafts remain a
separate future workflow extension. The predeclared SmolLM2 fallback may
proceed under a separately frozen activation record.

## SmolLM2 compatibility outcome

The fallback activation was frozen in
`configs/phase13_smollm2_fallback_activation_frozen.json` before the live
preflight. SmolLM2 passed all 15 checks on its one authorized live run:

- result: `results/model_onboarding_smollm2_17b_candidate_live.json`
- result SHA-256:
  `ba62b638b81b2f70c73597d20496af1aee9836630e05a58574731cfa23151df1`
- model: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- revision: `31b70e2e869a7173562077fd711b654946d38674`
- architecture: `LlamaForCausalLM`, `model_type=llama`
- decoder path: `model.layers`
- decoder blocks / hidden states: 24 / 25
- residual width: 2048
- contextual digit IDs: 32 through 41
- candidate reader hidden-state indices: 1, 4, 7, 10, 13, 16, 19, 22
- gradient probe: hidden-state index 12, norm 0.6535
- every model parameter frozen

This pass authorizes fresh SmolLM2-specific discovery. It is not evidence that
the operand reader, leading compiler, suffix writer, or integrated graft will
pass.

## Fresh SmolLM2 discovery corpus

`configs/phase13_smollm2_discovery_dataset_frozen.json` defines 270 examples:

- 90 fit, 90 selection, and 90 development examples;
- zero operand-pair overlap with every prior source through the Phase 12
  Qwen audit;
- zero pair overlap between the three Phase 13 splits;
- three new prompt templates per split, with no prior template reused;
- balanced leading, tens, ones, and carry labels within every split.

The complete dataset SHA-256 is
`cb99c87d4c70caa5b738b534c928e40526b19acbded3c7508d909b49d42c6b35`.
The dataset is audit-sealed. Fit and selection may be exposed for discovery;
development may be used only after component selection; none of these examples
may become the later one-shot audit.

## Frozen capability baseline

`configs/phase13_smollm2_capability.json` freezes the first task-specific
SmolLM2 model run before observing its output. The runner:

- uses only the 90 exposed fit examples;
- greedily requests exactly three continuation tokens;
- records exact-result, position, and contextual digit-token rates;
- verifies the model revision, onboarding result, dataset, rendered prompts,
  token contract, digit-token map, runner, and direct code dependencies by
  SHA-256;
- keeps every model parameter frozen and refuses to overwrite an existing
  result.

This measurement has no advancement threshold. A strong or weak frozen-model
baseline does not select a latent boundary and does not count as reader,
writer, development, or audit evidence.

## Model-specific discovery boundary

The following workflow components transfer:

- semantic operand-span contract;
- split and hash discipline;
- reader family and selection thresholds;
- local-margin compiler algorithm;
- native-coordinate discovery families;
- paired wrong-target and norm-matched random controls;
- development-before-audit and one-run audit rules;
- typed manifests, validators, and release packaging.

The following must be rediscovered:

- reader hidden-state boundary and centroids;
- candidate token IDs as observed from the frozen tokenizer;
- leading compiler boundary, margin, norm cap, and convergence depth;
- suffix boundaries, bases, ranks, prototypes, scales, and norm caps;
- batch sizes required by the model;
- all fitted artifacts.

No Phi or Qwen reader, basis, prototype, delta, layer, scale, margin, or
iteration choice may be loaded as a third-family discovery input.

## Advancement ladder

1. Pass the model-onboarding compatibility preflight.
2. Generate pair- and template-disjoint fit, selection, and development data.
3. Measure the frozen model's arithmetic capability and output contract.
4. Select a token-local operand reader with a rotated-label control.
5. Discover the leading and suffix writers under target, identity,
   wrong-target, and norm-matched random controls.
6. Pass a complete exposed read → host addition → hybrid write development
   gate.
7. Only then generate and freeze one fresh audit dataset and authorize exactly
   one run.
8. Package only a passing audit; preserve any nonpass unchanged.

## Claim boundary

Three successful model families would satisfy the research program's current
workflow-portability milestone. It would still not prove that every
transformer contains the interface, that fitted vectors transfer, that the
model autonomously invokes the deterministic mechanism, or that the workflow
decodes natural-language chain of thought.
