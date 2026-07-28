# Open Latent Interfaces Research Program

## Thesis

Pretrained language models may already contain native, task-specific
representational interfaces that their own circuits use to exchange semantic
state. If those interfaces can be discovered and causally validated, frozen,
auditable computation may be attached to them without retraining the base model
to adopt an artificial vector protocol.

The long-run artifact is a repeatable open workflow for:

1. reading latent state;
2. mapping task-specific channels across layers and positions;
3. establishing causal necessity and sufficiency;
4. compiling typed values from native latent state;
5. executing a deterministic mechanism;
6. writing the exact result back in the model's native representational format;
7. measuring preservation outside the target task.

## Terminology

- **Latent interface:** a causally validated representational surface through
  which model components exchange task-relevant state.
- **Latent cartography:** mapping where, when, and in what geometry a task state
  appears.
- **Read bridge:** a frozen-at-inference decoder from model state to a typed
  deterministic contract.
- **Write bridge:** a frozen-at-inference encoder from a deterministic result
  to a model-native intervention.
- **Deterministic latent graft:** a read bridge, deterministic mechanism, and
  write bridge attached during inference while every base-model weight remains
  unchanged.

These terms are project vocabulary, not historical-novelty claims.

## Program ladder

### Phase 0 — instrumentation and native math-channel pilot

Build model-agnostic activation capture, split-safe datasets, linear probes,
and causal residual interventions. Test a probe-defined oracle-result shift on
frozen Qwen2.5-0.5B.

### Phase 1 — triangulated mathematical cartography

Run NLA explanations, Jacobian-lens readouts, contrastive directions, probes,
and activation patching on the same prompts. A claimed channel must survive
independent methods and counterfactual contrasts.

### Phase 2 — native typed read interface

Decode route, operand roles, digits, operation, and execution timing from
existing activations. Freeze a prompt-family-disjoint audit. Compare linear,
nonlinear, NLA-derived, and J-space-derived readers without changing the base.

### Phase 3 — native result write interface

Encode exact results through activation-reconstructor differences, J-lens
directions, learned low-rank bridges, and native donor-state patches. Require
causal downstream use, multi-token exactness, and unrelated-task preservation.

### Phase 4 — closed deterministic graft

Connect the typed reader to a zero-parameter exact arithmetic mechanism and the
validated writer. The base model remains frozen. Compare against untouched
base, parser/tool, matched learned bridge, random/norm-matched intervention,
and oracle read/write conditions.

### Phase 5 — model portability

Repeat the frozen protocol across at least three unrelated open model families.
Separate a universal workflow from model-specific trained artifacts.

### Phase 6 — open NLA factory

Package activation extraction, proxy-data generation, AV/AR warm start, RL,
faithfulness checks, checkpoint metadata, and inference. Reduce compute through
small verbalizers/reconstructors and staged token selection. Reproduce at least
one released NLA before claiming recipe portability.

## Central falsifiers

The program should narrow or stop if:

- task information is decodable but causal interventions do not alter the
  relevant computation;
- exact operands are not available in a stable representational format before
  output generation;
- write interventions work only by directly biasing final logits;
- grafts require large, prompt-specific, off-manifold perturbations;
- unrelated capabilities degrade materially;
- readers or writers fail across paraphrases, layers, seeds, or model families;
- NLA text cannot be independently corroborated.

## Relationship to deterministic-neuron implants

The existing implant program trains an explicit semantic interface around a
frozen deterministic mechanism. This program asks whether the base model's
existing representations can supply that interface. The programs share the
same failure decomposition—routing, typing, deterministic execution, result
encoding, and preservation—but modify different variables.

