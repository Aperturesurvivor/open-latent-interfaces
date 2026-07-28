# Phase 1B Probe and Carry-Causality Summary

## Outcome

**Exact pre-output read non-pass; linear carry write non-pass.**

Fresh, canonical-pair-disjoint data were frozen before probing:

```text
90 train examples — direct template
45 development examples — symbolic template
45 unopened audit examples — word-problem template
dataset SHA-256:
  33bde4a417cbde662295c3c69265b17dc8160aa110ae8090f5e74819f1e6c7b4
```

All capability-gate pairs were excluded.

## Pre-output probe map

At the final user token, separate ridge probes attempted to decode all operand
digits, result digits, and carry bits from every hidden-state boundary.

- result hundreds digit: 97.8% at the final hidden state; first exceeded 80%
  at hidden state 25;
- result tens digit: best 17.8%;
- result ones digit: best 33.3%;
- exact three-digit result: best 6.7%;
- exact operand A: best 15.6%;
- exact operand B: best 33.3%.

The model is behaviorally correct in this regime, but a simultaneous exact
result is not linearly readable before generation.

## Teacher-forced timing probes

The same train/development split was rendered at three boundaries: before the
first digit, after the correct first digit, and after the correct first two
digits.

| HF hidden state | First digit | Second digit | Third digit | Combined teacher-forced result |
|---:|---:|---:|---:|---:|
| 23 | 46.7% | 17.8% | 22.2% | 0% |
| 24 | 73.3% | 60.0% | 97.8% | 40.0% |
| 25 | 86.7% | 80.0% | 100% | 71.1% |
| 26 | 93.3% | 95.6% | 100% | 88.9% |
| 27 | 95.6% | 97.8% | 100% | 93.3% |
| 28 | 97.8% | 100% | 100% | 97.8% |

This independently reproduces the J-lens timing transition. Reliable digit
state appears after 80% depth and each suffix probe benefits from the correct
prefix already being in context.

## Internal carry candidate

Tens-carry was the strongest typed variable inside the depth gate:

```text
hidden state 21 / decoder block 20 (~71% depth)
accuracy: 82.2%
balanced accuracy: 82.5%
AUC: 87.7%
shuffled-label balanced accuracy: 47.9%
```

This candidate was committed before intervention.

## Causal intervention

On 42 eligible development prompts, the binary probe direction was shifted
toward the opposite carry label. The corresponding mechanistic prediction was
an answer changed by exactly ±100.

At strength 1:

- desired probe label after intervention: 100%;
- mean delta norm: 1.134;
- mean delta / residual norm: 1.57%;
- counterfactual hundreds-digit accuracy: 2/42 (4.8%);
- original hundreds-digit accuracy: 40/42 (95.2%).

Counterfactual accuracy stayed 2/42 for targeted, shuffled-label, and
norm-matched random directions at every tested strength. At strength 4, the
targeted delta reached 6.28% of residual norm and still did not change that
count.

The direction changes what the probe reads without changing what the model
uses. It is a decodable correlate, not a sufficient carry-write interface.

## Current model of the computation

The combined evidence favors a late, autoregressive construction:

1. internal state contains some operand/carry correlates;
2. the first output digit becomes explicit in the final blocks;
3. later digits become explicit after earlier digits enter context;
4. neither average-Jacobian transport nor linear probes recover a stable,
   simultaneous pre-output full result;
5. flipping the strongest internal carry probe does not implement a
   counterfactual computation.

This does not rule out a nonlinear or distributed internal carry mechanism.
The next causal test is a native donor-state patch across the internal layer
grid. A full residual donor patch can distinguish “wrong linear direction”
from “no causally useful state at this boundary.”

## Claim boundary

All results are development-only. The 45-example audit remains unopened.
Nothing here supports a deterministic-graft or NLA-faithfulness claim.
