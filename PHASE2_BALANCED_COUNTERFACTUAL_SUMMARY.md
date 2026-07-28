# Phase 2 Balanced-Counterfactual Summary

## Outcome

**The unbiased target construction exposed a position-specific bottleneck.**

The adapter controlled the leading digit on 87/90 development examples and the
ones digit on 56/90, but controlled the tens digit on only 20/90. It generated
the complete synthetic target on 13/90 examples. Preservation, intervention
norm, and parseability gates passed; exactness, suffix, and control-advantage
gates failed. The audit remains sealed.

## Provenance

- Frozen protocol commit: `2392f97`.
- Frozen config SHA-256:
  `494e58973e71fe38092bbcece056ffabc85022925b7e0ed3c4038358ad9d2e2b`.
- Result SHA-256:
  `01900b9391e6f03358922d2aacadd06695c228af2eb4eb5f3e7357f3afd605bf`.
- Weights SHA-256:
  `be23dac0cb1d30a1929fc9ee24a839c3c530565f5e7efd3b1505261563c5d7c2`.
- Audit examples evaluated: 0/90.

## Target integrity

Every synthetic target digit differs from the correct result digit. Target
classes are exactly balanced at each answer position within every split.

| Split | Examples | Unchanged digits by position | Target SHA-256 |
|---|---:|---:|---|
| Fit | 450 | 0, 0, 0 | `34b4eeeb...ce58c` |
| Selection | 90 | 0, 0, 0 | `e634189d...44098` |
| Development | 90 | 0, 0, 0 | `9113b3aa...c803` |

The untouched model still matched 3, 1, and 1 target digits by chance because
six of its development answers were arithmetically incorrect. It matched one
complete synthetic target.

## Development result

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| **Targeted adapter** | **87/90** | **20/90** | **56/90** | **13/90** | 0/90 |
| Base | 3/90 | 1/90 | 1/90 | 1/90 | 84/90 |
| Same digit | 1/90 | 0/90 | 1/90 | 0/90 | 82/90 |
| Shuffled target | 19/90 | 5/90 | 9/90 | 0/90 | 13/90 |
| Shuffled state | 18/90 | 5/90 | 17/90 | 2/90 | 38/90 |
| Random direction | 2/90 | 0/90 | 0/90 | 0/90 | 83/90 |

All conditions were 100% parseable. The targeted adapter's mean relative
intervention norms were 36.3%, 24.3%, and 37.0%.

## Frozen gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Exact target | >=50% | 14.4% | Fail |
| Every position | >=70% | 96.7%, 22.2%, 62.2% | Fail |
| Exact advantage over every control | >=25 points | 12.2 points | Fail |
| Identity preservation | >=90% | 91.1% | Pass |
| Relative norm at every position | <=100% | 36.3%, 24.3%, 37.0% | Pass |
| Parse rate | 100% | 100% | Pass |

## Tens-position diagnosis

The tens result is not a checkpoint-selection artifact. Selection accuracy was
15/90 at epoch 1, 16/90 at epoch 2, and 13/90 at epoch 3; the selected
checkpoint reached 20/90 on development.

On development, 51/90 targeted outputs retained the original tens digit.
Desired digit zero was the sole easy class at 9/9; every other desired tens
class reached at most 2/9. The learned tens basis also moved furthest from its
initialization:

- 14.0% relative Frobenius change;
- mean principal cosine 0.994;
- minimum principal cosine 0.764.

The prior full native-donor intervention reached 42/45 tens digits at this
boundary with 69.7% mean relative norm. The present compressed tens writer
used 24.3%. This makes intervention strength the next minimal hypothesis to
test before changing rank, architecture, or layer.

## Interpretation

The earlier donor-matched benchmark overstated suffix performance because many
suffix labels were unchanged. Under genuinely counterfactual labels, causal
leading-digit control remains strong and target-specific, while ones control
remains substantial but below gate.

The result does not show that the residual boundary lacks tens information.
Native replacement already established that it can write the tens digit. It
shows that the current compact state-and-target writer does not yet overcome
the model's original tens computation at its selected amplitude.

## Next experiment

Run a preregistered scale sweep on the already-trained adapter:

1. choose each position's scale using selection only;
2. include amplitudes capable of approaching the native donor's relative norm;
3. keep the current weights, targets, controls, and gate fixed;
4. evaluate development once after scale selection;
5. proceed to a layer/rank study only if added amplitude fails.

No audit evaluation is authorized.
