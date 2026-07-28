# Phase 9C: Phi Iterative Leading-Token Compiler

## Purpose

Test the specific failure mode exposed by Phase 9B: a useful local derivative
whose one-shot linearization becomes inaccurate before crossing every
leading-token decision boundary.

The follow-up relinearizes the same frozen downstream model after each bounded
update. Rows receive an exact-zero additional update as soon as the requested
token is the full-vocabulary argmax. No model weight is trained or changed.

## Frozen boundary

This is another selection-only experiment on the same already exposed 180
Phase 8 selection examples. It cannot strengthen an audit claim. Phase 8 audit
rows remain forbidden.

The candidate is frozen as follows:

- residual boundary: hidden-state index 24;
- requested decimal-digit margin per step: 8.0;
- maximum relinearizations: 4;
- cumulative per-row norm cap: 0.75 times the original recipient-state norm;
- iteration candidates: 1, 2, 3, and 4;
- hard gate: exact-zero additional update for a row already at requested
  full-vocabulary top-1;
- identity condition: one compiled step, retained unchanged afterward;
- wrong-digit control: independently relinearized toward a deterministic
  different leading digit, then norm-matched to the target cumulative update;
- random control: independently seeded and norm-matched at each iteration.

Margin 8.0 is fixed from the middle of the monotonic Phase 9B grid. Iteration
count is the only selected numerical parameter. The compiler projects the
complete cumulative update, not each step independently, onto the norm ball.

## Selection rule

An iteration passes only if:

- target accuracy is at least 0.90;
- identity accuracy is at least 0.90;
- target advantage over the stronger control is at least 0.50;
- target digit-token rate is exactly 1.0;
- mean target relative norm is at most 0.75.

The smallest passing iteration is selected. If none pass, the best diagnostic
iteration is recorded and the hypothesis is closed.

## Claim boundary

A pass establishes an exposed development candidate only. It would show that
a deterministic optimizer can compile a requested output token into a frozen
intermediate residual state. It would not establish a stable semantic neuron,
a transcript of reasoning, or transfer to another model. Any integrated or
generalization claim requires a newly generated, pair-disjoint, one-shot
audit.

## Frozen selection outcome

The iterative compiler passed at the smallest eligible depth, three:

| Iterations | Target | Identity | Wrong control | Random control | Mean relative norm | Pass |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 77/180 | 180/180 | 6/180 | 1/180 | 0.0772 | no |
| 2 | 161/180 | 180/180 | 13/180 | 1/180 | 0.0971 | no |
| 3 | 180/180 | 180/180 | 7/180 | 1/180 | 0.0973 | yes |
| 4 | 180/180 | 180/180 | 7/180 | 1/180 | 0.0973 | yes |

The identical third- and fourth-step target results and norms confirm that the
hard gate applied no further update after success. At the selected depth, the
target advantage over the stronger control is `0.9611`.

The complete write-once result is
`results/phase9c_phi_iterative_leading_compiler_selection.json`, with SHA-256
`6776b9e315e6ceb56e3e61019774b9520e8ee2f7f85aa895907d117342b26447`.

This passes only the exposed selection gate. The selected implementation is
fixed at hidden index 24, margin 8.0, three relinearizations, and cumulative
norm cap 0.75 for any subsequent development or new audit.
