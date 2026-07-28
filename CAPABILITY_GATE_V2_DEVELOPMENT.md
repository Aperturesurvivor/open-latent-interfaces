# Capability Gate v2 Development

## Outcome

All five regimes pass the precommitted primary-chat thresholds on
Qwen2.5-1.5B-Instruct. Per the "hardest passing regime" rule,
`three_digit_mixed` is selected for the sealed audit.

| Regime | Primary chat aggregate | Worst chat template | Pass |
|---|---:|---:|---|
| Single-digit, no carry | 100% | 100% | yes |
| Single-digit, with carry | 100% | 100% | yes |
| Two-digit, no carry | 94.4% | 83.3% | yes |
| Two-digit, with carry | 94.4% | 83.3% | yes |
| Three-digit, mixed | 97.2% | 91.7% | yes |

Across both primary chat and diagnostic raw presentation, exact accuracy was
97.2% on 360 conditions and every continuation contained a parseable integer.

## Selection

The frozen preference chooses the hardest passing regime, so only the 24
primary-chat and 24 diagnostic-raw audit conditions for mixed three-digit
addition may be opened. The other regimes' audit examples remain unused.

Development artifacts:

```text
config SHA-256:
  3a1ba3a398fb89d7474a947a635e86c716e1ada2b03d414a266502e647a05fea
result SHA-256:
  139ca508a64b126027d8f4beafd358af8c472fee2a797fabf5b9325320512c9c
```

## Interpretation

The move from 0.5B/v1 to 1.5B/v2 is not evidence that the larger model has a
particular latent mechanism. It establishes a stable behavioral envelope and
an input contract appropriate to the instruction checkpoint. The three-digit
audit must pass before this regime enters Phase 1 causal cartography.
