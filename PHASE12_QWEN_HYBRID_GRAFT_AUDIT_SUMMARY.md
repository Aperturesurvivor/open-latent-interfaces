# Phase 12 Qwen Hybrid-Graft Audit

## Result

The first and only authorized Phase 12 audit passed all 19 frozen checks on
`Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

The audited path was:

1. decode both operands from hidden-state index 1 with a frozen nearest-centroid
   digit reader;
2. add the decoded integers with ordinary host-language integer addition;
3. compile the leading answer digit into hidden-state index 23 with the
   prospectively selected three-step local margin compiler;
4. write the tens and ones digits through rank-16 native coordinates at
   hidden-state index 27.

No model parameter was trained or changed.

## Prospective separation

The 90 audit examples used three prompt templates and operand pairs absent from
all earlier discovery, selection, development, and audit data through Phase 12.
The corpus balanced leading, tens, ones, and carry labels. The dataset,
rendered prompts, operand token positions, token contract, reader, compiler,
suffix writer, controls, thresholds, source hashes, and one-run output path
were committed before evaluation at `6caf062`.

The audit result was committed separately at `bad2f02`.

## Metrics

| Measure | Result |
|---|---:|
| Operand pairs decoded | 90/90 |
| Operand digits decoded | 498/498 |
| Deterministic sums | 90/90 |
| Base-model exact answers | 59/90 |
| Latent hybrid exact answers | 90/90 |
| Oracle hybrid exact answers | 90/90 |
| Base errors repaired | 31/31 |
| Base-correct answers preserved | 59/59 |
| Random-control base errors repaired | 1/31 |
| Wrong-target-control base errors repaired | 2/31 |
| Shuffled requested targets followed | 85/90 |
| Shuffled random targets followed | 1/90 |
| Shuffled semantic advantage over random | 93.3 points |
| Shuffled true answers retained | 0/90 |

The shuffled semantic condition reached 86/90 leading digits, 90/90 tens
digits, and 89/90 ones digits. Every output parsed, every intervention output
token was a decimal digit, and all intervention norms remained within the
frozen caps.

## Interpretation

The true-task condition alone would not distinguish causal writing from a
model that already knows many additions. The decisive control is the rotated
shuffled target: the semantic hybrid mechanism made the model emit an
arbitrary requested three-digit answer on 85/90 prompts, while an
equal-norm random intervention did so on only 1/90. That supports a
target-specific causal write mechanism at this boundary.

Together with the earlier passing Phi audit, this is a cross-family
replication of the latent-read → deterministic-compute → latent-write
workflow. It is not evidence that tensors transfer between models: Qwen
required a separately fitted reader, separately selected compiler depth,
different write boundaries, and different suffix ranks and scales.

## Claim boundary

The result applies to one frozen Qwen revision, two-operand three-digit integer
addition, the `Answer=` response contract, and an externally supplied semantic
operand locator. The deterministic adder is explicitly invoked outside the
model. The leading writer is a target-conditioned output-side causal compiler,
not a universal “math neuron.” The audit does not decode chain of thought,
establish autonomous tool calling, or prove a general deterministic reasoning
implant.

## Reproducible artifacts

- Audit protocol:
  `protocols/PHASE12_QWEN_COMPILER_HARDENING.md`
- Frozen audit config:
  `configs/phase12_qwen_hybrid_graft_audit.json`
- One-shot result:
  `results/phase12_qwen_hybrid_graft_audit.json`
- Result SHA-256:
  `9318e4e564a4e8f3cf37e00f0292d4f2c3ad11ec08e0446023cf483aec197ffc`
- Typed manifest:
  `manifests/qwen25-15b-hybrid-arithmetic-graft-v1.json`
- Manifest SHA-256:
  `80d410649a05837de06d4b49ce85d2ac1f39ae08662f092fa0918b16342f2a5f`
- Interface documentation:
  `docs/HYBRID_GRAFT_INTERFACE.md`
- Release:
  `phase12-qwen-hybrid-graft-audit-v1`
