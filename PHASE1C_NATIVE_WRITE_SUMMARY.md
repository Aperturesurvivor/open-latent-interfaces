# Phase 1C Native Donor-Write Summary

## Outcome

**Causal leading-digit write found; it motivated a passing stepwise
full-result development experiment.**

Replacing the final-prompt residual state with a matched native donor state
localized a sharp causal transition at HF hidden state 23 / decoder block 22,
about 79% through Qwen2.5-1.5B.

This is the first Phase 1 result that satisfies the protocol's internal-depth
boundary and beats all matched controls.

## All-layer result

The target donor always had the cyclic next leading result digit, while suffix
distance and carry mismatches were minimized.

| HF hidden state | Target leading digit | Full donor result | Original result | Mean delta / residual norm |
|---:|---:|---:|---:|---:|
| 14 | 0% | 0% | 95.6% | 3.3% |
| 18 | 0% | 0% | 95.6% | 9.4% |
| 21 | 6.7% | 0% | 51.1% | 32.1% |
| 23 | 93.3% | 4.4% | 2.2% | 62.8% |
| 24–27 | 93.3% | 2.2–4.4% | 2.2–4.4% | 66.5–71.2% |
| 28 | 91.1% | 2.2% | 20.0% | 73.7% |

At hidden state 23:

- targeted donor target-digit accuracy: 42/45;
- random norm-matched: 1/45;
- shuffled-donor norm-matched: 2/45;
- same-leading donor: 0/45 target transfer and 44/45 original preservation.

The target-digit margin changes from -14.85 at base to +12.82 under the
targeted donor. The patching mechanism is therefore functional and
semantically selective at this boundary.

## Hybrid-output follow-up

The all-layer run established leading-digit transfer but did not record which
suffix survived. A separately frozen follow-up selected hidden state 23 from
the committed result hash.

Under the targeted donor:

```text
target leading digit in generated answer: 38/45 (84.4%)
recipient original suffix preserved:       42/45 (93.3%)
donor suffix transferred:                   2/45 (4.4%)
target-leading + recipient-suffix hybrid:   38/45 (84.4%)
full donor result:                           2/45 (4.4%)
```

The model therefore emits a counterfactual hybrid: the native patch replaces
the currently prepared leading result digit, while downstream autoregressive
computation retains the recipient's remaining two digits.

## Interpretation

The combined Phase 1 evidence supports a sequential native interface:

1. the model prepares the next answer digit late but before output;
2. a native residual state at block 22 causally controls that digit;
3. later digits are not simultaneously carried by the patched state;
4. after the first digit is emitted, downstream computation continues from the
   recipient prompt and prefix.

This is stronger than probe decodability and stronger than a final-logit
direction. The causal effect exists at the latest boundary permitted by the
frozen internal-depth gate and survives direct generation.

## Limitations

- The patch is a full residual replacement, not a minimal typed direction.
- Its mean norm is 62.8% of the recipient residual norm.
- A donor state transfers many latent variables even though the observed
  output effect is digit-selective.
- The result is development-only.
- Full multi-token target writing has not passed.
- No NLA-compatible 1.5B pair exists yet.

## Follow-up

Apply matched native donor states at the same boundary at each autoregressive
digit step. The target is a complete three-digit counterfactual result:

```text
deterministic target digits
  → native donor state for digit 1
  → emit digit 1
  → native donor state for digit 2
  → emit digit 2
  → native donor state for digit 3
  → emit digit 3
```

That frozen follow-up passed on 38/45 development examples, versus at most 1/45
for any control. See the
[Phase 1D stepwise native-write summary](PHASE1D_STEPWISE_NATIVE_WRITE_SUMMARY.md).
The next experiment is to compress the donor replacements into a low-rank
typed write bridge and compare against random, shuffled, same-digit, and
full-residual controls.
