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

