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

## Integrated development outcome

The separate Phase 12 development gate passed:

- result: `results/phase12_qwen_hybrid_graft_development.json`
- result SHA-256:
  `55b20af26f71368e33ec97b21688088c9fe28050e4e871f9a731de254ee40c16`
- reader and deterministic compute: 90/90
- latent and oracle true-task output: 90/90
- base: 82/90
- recovered base errors: latent 8/8, random 0/8, wrong-target 1/8
- preserved base-correct outputs: 82/82
- shuffled semantic target following: 86/90
- shuffled random target following: 0/90
- shuffled target-following advantage: 86/90
- shuffled position accuracy: 88/90 leading, 90/90 tens, 88/90 ones

All frozen checks passed. Phase 12 may therefore construct one new audit using
pairs and templates disjoint from Phase 11, Phase 12 selection, and Phase 12
development. The development result itself is not audit evidence.

## Frozen one-shot audit

The authorized audit is now sealed before model evaluation:

- dataset config:
  `configs/phase12_qwen_hybrid_graft_audit_dataset_frozen.json`
- audit config: `configs/phase12_qwen_hybrid_graft_audit.json`
- dataset SHA-256:
  `4f2e4181d368e45bd4ef7846569eb7c06de47c03730e388693149264d48f0bb5`
- 90 new operand pairs, with zero overlap through Phase 12 development
- three new prompt templates, balanced at 30 examples each
- balanced leading, tens, ones, and carry labels
- fixed reader at hidden-state index 1
- fixed three-step leading compiler at hidden-state index 23
- fixed audited suffix writer at hidden-state index 27
- exactly one authorized run and a non-overwritable result path

The audit retains the development thresholds, including at least 95% exact
true-task output, at least 90% shuffled semantic target following, no more than
15% shuffled random target following, and at least 70 percentage points of
semantic target-following advantage. A preflight verified every frozen hash,
token position, token contract, artifact, and development linkage without
loading the model or evaluating an audit example.

## One-shot audit outcome

The single authorized audit passed every frozen check:

- result: `results/phase12_qwen_hybrid_graft_audit.json`
- result SHA-256:
  `9318e4e564a4e8f3cf37e00f0292d4f2c3ad11ec08e0446023cf483aec197ffc`
- reader: 90/90 operand pairs and 498/498 operand digits
- deterministic addition: 90/90
- latent and oracle hybrid output: 90/90 exact
- base model: 59/90 exact
- recovered base errors: latent 31/31, random 1/31, wrong-target 2/31
- preserved base-correct outputs: 59/59
- shuffled semantic target following: 85/90
- shuffled position accuracy: 86/90 leading, 90/90 tens, 89/90 ones
- shuffled random target following: 1/90
- shuffled semantic advantage over random: 84/90, or 93.3 percentage points
- shuffled true-task output: 0/90

All 19 checks passed on the first and only run. This supports a
pair- and template-disjoint Qwen replication of the complete latent-read,
deterministic-addition, hybrid-write workflow at the stated three-digit
arithmetic boundary. It does not establish a universal model-independent
interface or a general deterministic reasoning implant.
