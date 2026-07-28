# Phase 6: Fresh Qwen Conditional Carry Writer

## Purpose

Test whether Qwen's token-local carry transport is predictable from the
recipient's native hidden state when a population-average carry vector is not
specific enough. This is a new hypothesis on a new corpus, not a correction of
the exposed Phase 5 audit.

## Independence boundary

- The frozen Qwen revision and the previously localized hidden-state boundary
  (16, at the second operand's ones token) are retained as prior findings.
- Every canonical operand pair in Phase 4/5, including the exposed audit, is
  excluded before any Phase 6 split is assigned.
- Phase 6 fit, selection, development, and audit pairs are mutually disjoint.
- Phase 5 audit rows are not used for fitting, selection, threshold changes, or
  model choice.
- The Phase 6 audit remains sealed until a complete development package passes.

## Precommitted writer family

For each behavior-eligible fit quartet:

1. capture the carry-base recipient state at hidden-state index 16 and the
   second operand's ones-token position;
2. define the native carry target as carry-increment minus carry-base at that
   exact site;
3. reduce recipient states with fit-only PCA;
4. reduce native target deltas with fit-only SVD;
5. fit ridge regression from standardized recipient coordinates plus source
   ones-digit interactions to target-delta coordinates;
6. reconstruct one donor-free, token-local residual delta at inference.

A matched no-carry bridge is fitted with the same architecture from control
base/increment pairs. It is a causal control, not part of the target writer.

## Frozen selection grid

- recipient-state ranks: 8, 16, 32
- transport ranks: 4, 8, 16, 32
- ridge penalties: 1, 10, 100
- output scales: 0.75, 1.0, 1.25, 1.5

The smallest-rank, smallest-norm candidate is preferred among candidates with
equal target accuracy and control advantage.

## Controls

- matched no-carry conditional bridge, norm-matched per example;
- source-digit-rotated carry prediction, norm-matched per example;
- shuffled-recipient carry prediction, norm-matched per example;
- deterministic random direction, norm-matched per example.

## Selection and development gates

On 45 untouched quartets:

- parse rate is 1.0;
- target tens accuracy is at least 0.50;
- target exceeds the strongest control by at least 0.25.

Development is one shot after selection. A failed development package closes
this writer family on this corpus. Audit authorization requires a passing
development package and freezes every model, rank, ridge, scale, token
position, hash, control seed, metric, and threshold.

## Claim boundary

A passing audit would establish a reproducible donor-free conditional
coordinate for this matched carry transition in this frozen Qwen revision. It
would not establish an individual carry neuron, a transcript of hidden
reasoning, universality across prompts or models, or general arithmetic
correctness.
