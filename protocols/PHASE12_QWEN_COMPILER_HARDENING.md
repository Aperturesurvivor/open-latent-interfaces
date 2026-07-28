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
