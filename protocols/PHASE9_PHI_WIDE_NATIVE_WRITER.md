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
