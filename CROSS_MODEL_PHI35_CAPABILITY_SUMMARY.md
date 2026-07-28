# Phi-3.5 Cross-Model Capability Screen

## Outcome

`microsoft/Phi-3.5-mini-instruct` passed the precommitted behavioral screen for
cross-family replication. On the 180 native-chat development conditions it
answered 179 exactly (99.44%) and produced a parseable integer on all 180.

Every regime passed the frozen gate:

| Regime | Exact | Worst template cell | Gate |
| --- | ---: | ---: | --- |
| Single digit, no carry | 36/36 | 12/12 | pass |
| Single digit, carry | 36/36 | 12/12 | pass |
| Two digit, no carry | 36/36 | 12/12 | pass |
| Two digit, carry | 36/36 | 12/12 | pass |
| Mixed three digit | 35/36 | 11/12 | pass |

The sole error was `209 + 381`: the symbolic condition returned `600` rather
than `590`. No row was removed.

## Frozen provenance

- model: `microsoft/Phi-3.5-mini-instruct`
- revision: `2fe192450127e6a83f7441aef6e3ca586c338b77`
- license recorded by the upstream model card: MIT
- frozen protocol commit: `ea1be74`
- configuration SHA-256:
  `bf3677cf4fc842bdb25ea85218316cb4a29dfd133ca9f30fd5944b337ff82375`
- dataset SHA-256:
  `bed9d447a6aa59c5c825e537b69d5cbadd9c8a0e8c049acad8badd1338af56af`
- result SHA-256:
  `348504758f3757939bbeaaad6c47a73e2bf5cc4d6990e27ffd858b28740dfc80`
- runtime: Python 3.12.12, PyTorch 2.13.0, MPS, float16

The result is preserved in
`results/cross_model_phi35_capability_development.json`.

## Decision

Proceed with Phi-3.5 as the first cross-family replication target. The
model-specific interface experiment must use newly generated fit, selection,
development, and sealed-audit operand pairs that exclude all prior Qwen
corpora. The mixed three-digit regime is behaviorally eligible.

## Claim boundary

This screen establishes behavioral competence only. It does not show that Phi
uses the same latent representation, boundary, subspace rank, or controller as
Qwen. It is not evidence of cross-model portability, and no Phi confirmatory
audit has been opened.
