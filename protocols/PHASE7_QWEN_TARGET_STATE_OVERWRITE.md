# Phase 7: Qwen Target-State Overwrite

## Purpose

Test whether the carry-linked site supports a typed target-state coordinate
even though portable additive carry deltas failed. A deterministic mechanism
supplies the intended result tens digit. The writer overwrites that digit's
coordinates in the recipient state rather than attempting to replay an
increment transport.

## Fresh-data boundary

- Model: `Qwen/Qwen2.5-1.5B-Instruct` at revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Every canonical operand pair in Phase 4/5 and Phase 6 is excluded before
  splitting.
- Fit, selection, development, and audit pairs are mutually disjoint.
- Audit remains sealed until selection and one-shot development both pass.

## Writer

At hidden-state index 16 and the second operand's ones-token position:

1. capture carry-increment target states from behavior-exact fit quartets;
2. label each state by the target result's tens digit;
3. estimate one target-state centroid per digit;
4. obtain the between-centroid basis with fit-only SVD;
5. at inference, replace the recipient's coordinates in the retained basis
   with the deterministic target digit's centroid coordinates;
6. leave the orthogonal residual state unchanged.

This is an affine, recipient-dependent state overwrite:

`delta = ((centroid[target] - recipient) @ basis.T) @ basis * scale`.

It requires no live donor activation and changes no model weight.

## Frozen model and selection rule

- coordinate rank: the smallest fit-only between-centroid rank explaining at
  least 95% of between-centroid squared singular-value energy, capped at 9
- scales: 0.5, 1.0, 1.5, 2.0

Selection does not choose rank. It chooses the lowest passing scale. If none
pass, the writer family closes without development.

## Controls

- identity digit: request the unincremented result's tens digit;
- wrong digit: rotate the target tens digit to the next fitted class;
- deterministic random direction, norm-matched per example.

## Gates

On each untouched 45-quartet split:

- parse rate: 1.0;
- target tens accuracy: at least 0.50;
- target advantage over the strongest control: at least 0.25.

Development runs once with the selected rank and scale. A passing development
package may authorize exactly one audit after runner, config, artifact,
dataset, token-contract, and evidence hashes are frozen.

## Claim boundary

A passing audit would establish a deterministic-target-conditioned native
state interface at one Qwen site for this matched addition distribution. It
would not establish autonomous carry detection, a natural-language thought
transcript, a single neuron, unrestricted arithmetic, or cross-model transfer.
