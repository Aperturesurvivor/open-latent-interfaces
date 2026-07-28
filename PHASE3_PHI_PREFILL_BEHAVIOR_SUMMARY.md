# Phase 3 Phi Prefill Behavior Summary

## Outcome

The frozen `Answer=` assistant-prefix contract passed on every development
split of the fresh Phase 3 corpus:

| Split | Exact | Parseable | Gate |
| --- | ---: | ---: | --- |
| Fit | 428/450 (95.11%) | 450/450 | pass |
| Selection | 82/90 (91.11%) | 90/90 | pass |
| Development | 85/90 (94.44%) | 90/90 | pass |
| Combined | 595/630 (94.44%) | 630/630 | pass |

The gate required at least 90% exact accuracy separately on every evaluated
split. No example was removed. The 35 incorrect rows and their generated token
IDs remain in the raw result.

## Token contract

With the assistant response prefilled through `Answer=`, every result is
exactly three generated tokens. All 630 rows produced three-token,
three-character responses. The contextual digit-token map is:

| Digit | Token ID |
| ---: | ---: |
| 0 | 29900 |
| 1 | 29896 |
| 2 | 29906 |
| 3 | 29941 |
| 4 | 29946 |
| 5 | 29945 |
| 6 | 29953 |
| 7 | 29955 |
| 8 | 29947 |
| 9 | 29929 |

## Provenance

- frozen protocol and dataset commit: `906fba2`
- dataset SHA-256:
  `5a8783160f8add9bf551a8de3207b1b5a5c9ac763638250d00cd6de1807bf941`
- result SHA-256:
  `b8c71ac2ae6bb03b9fb707e3434695cea5f98cecdfb584f25a697e0dfdcbda35`
- model revision:
  `2fe192450127e6a83f7441aef6e3ca586c338b77`
- runtime: Python 3.12.12, PyTorch 2.13.0, MPS, float16
- generation time: 133.01 seconds

## Decision and claim boundary

Proceed to causal boundary rediscovery without evaluating the audit split.
This result verifies arithmetic behavior and digit-token alignment under the
frozen output contract. It does not establish any latent representation,
causal interface, or cross-model portability.
