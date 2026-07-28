# Phase 9E: Fresh Phi Hybrid-Graft Audit Dataset

## Purpose

Create a claim-bearing corpus that is disjoint from every earlier fit,
selection, development, and audit source and that removes the narrow
carry-base answer-suffix distribution used in Phase 8 development.

The corpus is frozen before audit behavior is measured. Its dataset config
sets `audit_authorized` to false.

## Construction

The deterministic generator creates 90 additions with results from 100
through 999. It excludes the canonical unordered operand pairs from the
complete generated Phase 3, Phase 4, Phase 6, and Phase 7 datasets. Those
sources recursively include the earlier capability and Phase 1–2 pair
universes.

The frozen result has:

- ten examples for each leading digit 1–9;
- nine examples for each answer tens digit 0–9;
- nine examples for each answer ones digit 0–9;
- 45 ones-carry and 45 no-ones-carry examples;
- 43 tens-carry and 47 no-tens-carry examples;
- 30 examples in each of three new prompt-template families;
- 90 unique ordered prompts and 90 unique canonical operand pairs.

The exact dataset SHA-256 is
`8f598a3a1891aae5bd3f4c12cb8ad4b3c08347fffc4dfc3969986f6a6ae54c0e`.
The ordered example-ID SHA-256 is
`fb1d6fa21e4ea762a6f6d67589eda88f6493d59e3204a25de7dbe6e4bd268e5c`.
The ordered canonical-pair SHA-256 is
`f83211e26f61544b66952eeb75524666aab30e08c89aa6cd44b6303f4238d365`.

## Leakage boundary

Result values and individual operand values may recur; arithmetic
generalization does not require globally novel integers. The leakage unit is
the unordered operand pair and its rendered prompt. Both are unique, and no
audit pair appears in the named prior datasets.

No model output has been read to choose, filter, replace, or stratify these
examples. Base-model difficulty is therefore unknown at freeze time.

## Authorization boundary

This corpus alone does not authorize an audit. The complete runner, token
contract, source artifacts, development evidence, controls, thresholds,
output path, and one-run limit must be hash-locked in a separate audit config
before `audit_authorized` may be true there.

