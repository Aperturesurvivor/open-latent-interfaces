# Phase 12: Qwen Compiler Convergence Hardening

## Motivation

The immutable Phase 11 audit passed 18 of 19 checks. Its only failure was
arbitrary shuffled-target following: 70/90 versus a frozen 81/90 requirement.
Failure localization showed:

- leading position: 72/90
- tens position: 90/90
- ones position: 88/90
- 18/20 failed full outputs differed only at the leading position

The true arithmetic correction path remained perfect. Phase 12 therefore
targets prompt-local leading-compiler convergence, not the reader,
deterministic computation, or suffix coordinates.

## Prohibited shortcuts

Phase 12 may not:

- rerun or rewrite the Phase 11 audit;
- change its threshold or claim that it passed;
- train on Phase 11 audit pairs or prompts;
- transfer any Phi parameter or tensor;
- change the Qwen reader or audited suffix writer during compiler selection;
- simply substitute a previously observed iteration without prospective
  selection on new data.

## New exposed corpus

`configs/phase12_qwen_compiler_robustness_dataset_frozen.json` defines 180 new
examples:

- 90 selection and 90 development examples;
- zero pair overlap with all prior data through the Phase 11 audit;
- no pair overlap between selection and development;
- three new template families per split, with no template reused from the Phi
  or Qwen audits;
- balanced leading, tens, ones, and carry labels in each split.

## Compiler selection

The Phase 11 boundary, desired margin, and norm cap remain fixed:

- hidden-state index 23
- desired margin 16
- relative-norm cap 0.25

`configs/phase12_qwen_compiler_robustness_selection.json` prospectively
evaluates cumulative relinearization depths one through four on balanced
counterfactual leading targets.

The earliest passing depth must satisfy:

- at least 95% overall target accuracy;
- at least 90% target accuracy in every template family;
- at least 95% identity accuracy;
- at least 50 percentage points over the strongest wrong/random control;
- mean relative norm no greater than 0.25;
- 100% decimal-digit token rate.

No integration or audit claim is open at this stage.

## Selection outcome

The robustness selection passed:

- result: `results/phase12_qwen_compiler_robustness_selection.json`
- result SHA-256:
  `0726da399a0b17ba50daed6c2d80b196d2a40882a445b9f6fdca287bb1e9b7f0`
- selected convergence depth: 3
- target accuracy: 88/90
- per-template target accuracy: 30/30, 29/30, and 29/30
- identity accuracy: 90/90
- wrong-target control: 8/90
- random control: 1/90
- mean relative norm: 0.2048

Depth two reached 83/90 overall but only 26/30 in its weakest template family,
so it failed. Depth four reached 89/90, but depth three was selected under the
frozen earliest-passing rule.

## Integrated development boundary

`configs/phase12_qwen_hybrid_graft_development.json` freezes the selected
three-step compiler with the unchanged reader and audited suffix writer on the
separate 90-example Phase 12 development split.

The gate requires:

- at least 95% exact and per-position target accuracy;
- at least 98% reader and deterministic-compute accuracy;
- at least 98% base-correct preservation and 80% base-error recovery;
- at least 50-point recovery advantages over random and wrong-target controls;
- at least 90% shuffled semantic target following;
- at most 15% shuffled random target following;
- at least 70 points of semantic target-following advantage;
- complete parse and decimal-token rates.

No Phase 12 audit may be generated unless this separate integration gate
passes.
