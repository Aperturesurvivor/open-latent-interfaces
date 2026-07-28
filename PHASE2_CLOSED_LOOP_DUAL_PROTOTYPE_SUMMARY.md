# Phase 2 Closed-Loop Dual-Prototype Summary

## Outcome

**The donor-free dual-prototype implant passed every frozen development gate,
generating 87/90 exact balanced counterfactual results.**

The leading causal adapter was composed with a shared rank-16 next-digit
interface at both suffix positions. Tens and ones control each reached 90/90
in closed loop. Identity preservation reached 90/90, and the strongest exact
matched control reached 10/90.

## Provenance

- Frozen protocol commit: `f1da7bc`.
- Frozen config SHA-256:
  `4100439a712f037af7ee2f0294294651c195feb5e66b46e10514e0d90e3d6908`.
- Result SHA-256:
  `2058ac08e242be4b0b3f7832400b25c29e6eb4f74eb4cfc171fdb3b10d949cb7`.
- New weights, bases, or prototypes fitted: none.
- Audit examples evaluated: 0/90.

## Development result

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| **Targeted implant** | **87/90** | **90/90** | **90/90** | **87/90** | 0/90 |
| Base | 3/90 | 1/90 | 1/90 | 1/90 | 84/90 |
| Identity, hard gated | 0/90 | 0/90 | 0/90 | 0/90 | **90/90** |
| Shuffled target | 21/90 | 14/90 | 12/90 | 1/90 | 0/90 |
| Shuffled state | 18/90 | 40/90 | 70/90 | 10/90 | 10/90 |
| Random direction | 3/90 | 1/90 | 1/90 | 1/90 | 83/90 |

All conditions were 100% parseable.

The targeted implant's mean relative norms were 35.4%, 71.7%, and 90.6%.
Its hard gate fired on 3/90, 3/90, and 4/90 examples where the unmodified model
already produced the requested next digit.

## Frozen advancement gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Exact target | >=50% | 96.7% | Pass |
| Every position | >=70% | 96.7%, 100%, 100% | Pass |
| Exact advantage over every control | >=25 points | 85.6 points | Pass |
| Identity preservation | >=90% | 100% | Pass |
| Relative norm at every position | <=100% | 35.4%, 71.7%, 90.6% | Pass |
| Parse rate | 100% | 100% | Pass |

Every conjunctive development criterion passes.

## Interpretation

The three remaining target failures are exactly the three leading-digit
failures. Once a leading digit is generated, both suffix implants write their
requested digits on every example, including prefixes that differ from the
correct arithmetic answer.

The suffix mechanism is one reusable native latent interface:

- one rank-16 basis;
- one hidden-state boundary;
- one coordinate-replacement rule;
- one hard identity gate;
- position-specific ten-vector dictionaries.

It is donor-free at inference and changes no base-model parameter. This is
substantially more specific than activation steering: it is a typed,
deterministic next-digit write interface with explicit no-op semantics.

The shuffled-state control's high ones accuracy does not undermine exact causal
specificity. It reaches 70/90 ones digits only after producing incorrect or
control-dependent prefixes and reaches 10/90 complete targets. The targeted
implant's exact advantage remains 85.6 points.

## Audit status

The development pass authorizes freezing a one-shot audit package. It does not
authorize changing any component, scale, threshold, control, target mapping, or
metric after audit output is observed.

Before audit, commit:

1. the exact evaluation script;
2. all source and artifact hashes;
3. the deterministic audit target hash;
4. all six conditions and random seeds;
5. the unchanged advancement thresholds;
6. a configuration explicitly authorizing exactly one audit run.

The audit remains unopened until that commit exists.
