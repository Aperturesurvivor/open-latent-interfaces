# Phase 1B J-lens Development Summary

## Outcome

**Engineering pass; internal-interface non-pass.**

A 12-prompt, all-layer Jacobian lens was fit for the behaviorally competent
Qwen2.5-1.5B target. It produces finite matrices and valid provenance artifacts.
On the selected three-digit development regime, however, result-token
information becomes reliable only in the final three decoder blocks. J-lens
does not reveal a materially earlier answer representation than vanilla logit
lens.

## Timing result

Thirty-six development prompts were evaluated at three teacher-forced
boundaries:

1. before the first answer digit;
2. after the correct first digit;
3. after the correct first two digits.

For each layer, the predicted digit at each boundary was combined into a
teacher-forced three-digit score:

| Decoder block | J-lens full result | Logit-lens full result |
|---:|---:|---:|
| 0–22 | 0% | 0% |
| 23 | 13.9% | 13.9% |
| 24 | 19.4% | 16.7% |
| 25 | 91.7% | 91.7% |
| 26 | 97.2% | 94.4% |
| Actual final model | 97.2% | — |

At the first, pre-answer boundary, J-lens digit accuracy rises from 16.7% at
block 19 to 50.0% at block 23, 66.7% at block 24, and 100% at blocks 25–26.

Blocks 25–26 are 89–93% through a 28-block model. They violate the frozen rule
that no candidate deeper than 80% may be the sole interface. The small J-lens
advantage at block 26 is output-adjacent and does not establish a distinct
global-workspace readout.

## Fit integrity

```text
model revision:
  989aa7980e4cf806f80c7fef2b1adb7bc71aa306
J-lens revision:
  581d398613e5602a5af361e1c34d3a92ea82ba8e
fit prompts:
  12 generic WikiText passages, committed before fitting
elapsed:
  1241.59 seconds on MPS
lens SHA-256:
  94c6d1c326393370adc5de538cdaaef2d3c3a0d83bb6b9c2eba0102afcf550b9
```

One frozen passage was a Jacobian-norm outlier (`5.070 / sqrt(d)`) and moved the
running mean by 0.871. It was retained. The final prompt moved the mean by
0.158. The lens is development-only; an audit fit needs a larger corpus or a
preregistered robust estimator.

## Interpretation

The result rules out an easy version of the hypothesis: a generic-corpus
average Jacobian does not translate a stable, early exact-answer direction into
vocabulary space on this task.

It does not rule out:

- earlier operand/carry state in a non-vocabulary geometry;
- nonlinear or distributed exact-value state;
- task-specific features that average-Jacobian transport suppresses;
- a causal state that probes can decode but that is not yet disposed toward
  direct verbalization.

The next discriminator is layer-wide exact digit/operand/carry probing followed
by matched donor patches. If those also remain output-adjacent, this model/task
is not a viable native-interface candidate.

## Claim boundary

Teacher-forced full-result accuracy combines three different forward passes.
Digits 2 and 3 are measured after their preceding answer digits are present in
the context. This is a timing map, not evidence that the full result can be
decoded simultaneously before generation.

Eighteen real J-lens records were written through the common artifact schema.
All retain `hypothesis` status.
