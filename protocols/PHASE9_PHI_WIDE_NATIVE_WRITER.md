# Phase 9: Phi Wide-Distribution Native Writer

## Purpose

Improve the limiting component from the Phase 8 integrated audit without
using any Phase 8 audit row for fitting or parameter selection.

The previously audited Phi causal bases and boundaries remain fixed:

- leading digit: hidden-state index 24, rank 32;
- tens and ones digits: hidden-state index 30, rank 32;
- norm cap: 1.0;
- hard gate: exact zero when base argmax already equals the requested digit.

Only the ten native coordinate prototypes per answer position may be refitted.
The fit pool combines behavior-exact examples from the original non-audit
Phase 3 fit distribution and the wider non-audit Phase 8 fit distribution.
Pooling is required because the matched carry corpus intentionally contains
only a subset of natural answer ones digits.

## Fit eligibility

Phi behavior is measured separately on both frozen source corpora. Prototype
fitting uses only examples whose complete natural answer is exact. Every
position must retain all required digit classes with at least four examples
per class. Source dataset, behavior result, and eligible-ID hashes are frozen
independently.

## Selection

The frozen scales are `0.75, 1.0, 1.25, 1.5, 2.0`. Selection uses
counterfactual three-digit targets on the Phase 8 selection split. Each answer
position must satisfy:

- target digit accuracy at least 0.90;
- identity digit accuracy at least 0.90;
- target advantage over wrong-digit and random norm-matched controls at least
  0.50;
- digit-token rate 1.0.

The lowest-norm passing scale is selected independently by position. A failed
position closes this refit.

## Integrated development

If selection passes, the Phase 8 development examples may be reused as
development-only diagnostics because their prior outputs are already exposed.
They cannot provide a new audit claim.

A second-generation integrated audit requires an entirely new pair-disjoint
corpus and a new one-shot audit configuration. The closed Phase 8 audit may
not be rerun, filtered, or used to choose prototypes, scales, or thresholds.

## Claim boundary

Passing selection or exposed development would establish only a wider
development candidate. A new held-out audit is required before claiming that
the refitted writer or the rebuilt latent graft generalizes.

## Frozen selection outcome

The selection run closed the all-position refit as a non-pass:

- leading digit, hidden index 24, rank 32, scale 1.0: target accuracy
  `87/180` (`0.4833`), identity accuracy `172/180` (`0.9556`), and control
  advantage `0.2667`;
- tens digit, hidden index 30, rank 32, scale 1.0: target and identity
  accuracy `180/180`, with control advantage `0.9056`;
- ones digit, hidden index 30, rank 32, scale 1.0: target and identity
  accuracy `180/180`, with control advantage `0.9278`.

The complete result is
`results/phase9_phi_wide_writer_selection.json`. Its SHA-256 is
`c6a57224e2824845e2c7b60d449b3d0b58df9306aad69c15ee2fefee719acfb6`.
The fitted prototype artifact SHA-256 is
`62d7302c7bcfe7ebe529f4f20ab91f43b61ebc8bda8bf2df86ccb053868122f0`.

This result localizes the remaining writer bottleneck to the leading token.
The two suffix coordinates are retained as development candidates. The failed
leading prototype is not eligible for a new audit.
