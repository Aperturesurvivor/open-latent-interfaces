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
