# Phase 13 SmolLM2 Compiler-Graft Audit

## Result

The first and only authorized Phase 13 audit passed all 17 frozen checks on
`HuggingFaceTB/SmolLM2-1.7B-Instruct` revision
`31b70e2e869a7173562077fd711b654946d38674`.

The audited path was:

1. decode both operands at hidden-state index 1 with a frozen nearest-centroid
   digit reader;
2. add the decoded integers with ordinary host-language integer addition;
3. sequentially compile the three answer digits at hidden-state index 24 with
   one bounded prompt-local residual step per position.

No model parameter was trained or changed.

## Prospective separation

The 90 audit examples used three new prompt templates and operand pairs absent
from every prior dataset through all 270 Phase 13 fit, selection, and
development examples. The corpus balanced leading, tens, ones, and ones-carry
labels. The dataset, prompts, operand positions, token contract, reader,
compiler settings, controls, thresholds, source hashes, and one-run output path
were committed before evaluation at `55c816a`.

The first result was preserved unchanged in the separate commit `1e1290c`.

## Metrics

| Measure | Result |
|---|---:|
| Operand pairs decoded | 90/90 |
| Deterministic sums | 90/90 |
| Base-model exact answers | 33/90 |
| Latent compiler-graft exact answers | 90/90 |
| Oracle compiler-graft exact answers | 90/90 |
| Base errors repaired | 57/57 |
| Base-correct answers preserved | 33/33 |
| Random-control base errors repaired | 1/57 |
| Wrong-target-control base errors repaired | 2/57 |
| Shuffled requested targets followed | 83/90 |
| Shuffled random targets followed | 0/90 |
| Shuffled target-following advantage | 92.2 points |
| Shuffled true answers retained | 0/90 |

Every latent output parsed, every latent output token was a decimal digit, and
all three latent positions reached 90/90 requested-digit accuracy. Mean
relative intervention norms were 4.87%, 1.23%, and 0.30% at positions zero,
one, and two.

## Model-specific writer outcome

The transferable workflow did not imply an identical mechanism. A frozen
rank/scale/norm search for compact native suffix-coordinate prototypes failed
prospective selection and was preserved as a nonpass. A separately frozen
prompt-local compiler fallback then passed for both suffix positions. SmolLM2
therefore uses the same high-level read → deterministic compute → write
contract as Phi and Qwen, but all three output positions use prompt-local
compilation.

This mechanism heterogeneity is evidence for an adaptive discovery workflow,
not for portable tensors or universal latent coordinates.

## Interpretation

The true-task condition alone would not distinguish causal writing from
ordinary model competence. The rotated shuffled-target control did: the
compiler made the model emit an arbitrary requested three-digit answer on
83/90 prompts, while an equal-norm random intervention did so on 0/90. Random
and wrong-target controls repaired only 1 and 2 of the 57 base errors,
respectively, compared with 57/57 for the semantic graft.

Together with the earlier passing Phi and Qwen audits, this establishes the
current three-family workflow-portability milestone. It does not establish
that one fitted interface, vector, boundary, or writer transfers between model
families.

## Claim boundary

The result applies to one frozen SmolLM2 revision, two-operand three-digit
integer addition, the `Answer=` response contract, and an externally supplied
semantic operand locator. The deterministic adder is invoked outside the
model. The writers are target-conditioned output-side causal compilers, not
identified “math neurons.” The audit does not decode chain of thought,
establish autonomous invocation, cover arbitrary operations, or prove an
any-model deterministic reasoning implant.

## Reproducible artifacts

- Audit protocol:
  `protocols/PHASE13_MODEL_ONBOARDING_AND_THIRD_FAMILY.md`
- Frozen audit config:
  `configs/phase13_smollm2_compiler_graft_audit.json`
- One-shot result:
  `results/phase13_smollm2_compiler_graft_audit.json`
- Result SHA-256:
  `b630522727aff690bd40a66633798da308f1ba30edc19c3912873ea2324881fb`
- Typed compiler-graft manifest:
  `manifests/smollm2-17b-compiler-arithmetic-graft-v1.json`
- Manifest SHA-256:
  `d21c07479368b3d908d46a55ae9916990cc2550e9f2f7b6f29ecca54286d18a8`
- Typed operand-reader manifest:
  `manifests/smollm2-17b-operand-reader-v1.json`
- Release tag:
  `phase13-smollm2-compiler-graft-audit-v1`
