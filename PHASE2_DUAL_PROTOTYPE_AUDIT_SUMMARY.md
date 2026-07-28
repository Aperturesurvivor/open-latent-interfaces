# Phase 2 Dual-Prototype Audit Summary

## Outcome

**The frozen donor-free implant passed its one-shot 90-example audit with
88/90 exact counterfactual results.**

Leading, tens, and ones control reached 88/90, 90/90, and 90/90. Identity
preservation reached 90/90. The strongest exact norm-matched control reached
12/90. Every frozen audit criterion passed.

## One-shot provenance

- Audit-package commit: `115b3a9`.
- Audit config SHA-256:
  `4fbb7aad82ce52be5f83a72d69dcd1e03be6760f453cc72525e20cf92753a9a6`.
- Audit result SHA-256:
  `3412f243b9ce0f8951a06dc5ac5f834858394c51264df898b93c67301e09c647`.
- Audit target SHA-256:
  `ff602ee80c84f1f175e8f9e2ed30ac90eb894d63859d9e7f0cea6380ecb73a00`.
- Passing development result SHA-256:
  `2058ac08e242be4b0b3f7832400b25c29e6eb4f74eb4cfc171fdb3b10d949cb7`.
- Audit runs performed: 1.
- Audit examples evaluated: 90/90.

The audit runner verified every source hash, the passing development gate, the
audit target hash, explicit one-run authorization, and the frozen output path
before model loading. The output did not exist before execution and cannot be
overwritten by the runner.

## Held-out audit result

The audit uses a fourth prompt family:

> A warehouse has A units in one section and B in another. How many units are
> there altogether? Reply with only the integer.

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| **Targeted implant** | **88/90** | **90/90** | **90/90** | **88/90** | 0/90 |
| Base | 3/90 | 0/90 | 0/90 | 0/90 | 82/90 |
| Identity, hard gated | 0/90 | 0/90 | 0/90 | 0/90 | **90/90** |
| Shuffled target | 19/90 | 10/90 | 12/90 | 0/90 | 0/90 |
| Shuffled state | 36/90 | 39/90 | 47/90 | 12/90 | 16/90 |
| Random direction | 5/90 | 1/90 | 1/90 | 1/90 | 79/90 |

All conditions were 100% parseable.

The targeted implant's mean relative norms were 36.0%, 70.1%, and 95.6%.
Its hard gate emitted zero on 3/90, 3/90, and 2/90 targeted positions where
the unmodified model already predicted the requested digit.

## Frozen audit gate

| Requirement | Threshold | Audit | Result |
|---|---:|---:|---|
| Exact target | >=50% | 97.8% | Pass |
| Every position | >=70% | 97.8%, 100%, 100% | Pass |
| Exact advantage over every control | >=25 points | 84.4 points | Pass |
| Identity preservation | >=90% | 100% | Pass |
| Relative norm at every position | <=100% | 36.0%, 70.1%, 95.6% | Pass |
| Parse rate | 100% | 100% | Pass |

The mechanically tested conjunctive verifier returns `true`.

## Development-to-audit comparison

| Metric | Development | Audit |
|---|---:|---:|
| Exact target | 87/90 | 88/90 |
| Leading digit | 87/90 | 88/90 |
| Tens digit | 90/90 | 90/90 |
| Ones digit | 90/90 | 90/90 |
| Identity | 90/90 | 90/90 |
| Strongest exact control | 10/90 | 12/90 |
| Parse rate | 100% | 100% |

The held-out result reproduces the development behavior without degradation.
Both audit failures are leading-writer failures; the shared suffix interface
remains perfect.

## What has been demonstrated

For `Qwen/Qwen2.5-1.5B-Instruct` on three-digit addition outputs:

1. a late residual write window was causally localized;
2. native next-digit transport was compressed to a 16-dimensional subspace;
3. a ten-vector native-coordinate dictionary replaced donor coefficients;
4. the same basis and controller transferred across two autoregressive
   positions;
5. deterministic hard gating provided exact no-op behavior when the requested
   digit was already present;
6. the components composed in closed-loop generation;
7. the complete implant generalized to a sealed prompt family and operand
   split;
8. matched content, state, and random controls failed by large margins.

No base-model weight was retrained or fine-tuned. Audit inference required no
donor execution.

## Claim boundary

This is a strong proof of concept, not yet a universal NLA workflow.

It does not establish:

- transfer to another model family or size;
- control of arbitrary semantic or reasoning features;
- autonomous discovery of the best boundary and rank;
- robustness outside the frozen three-digit output grammar;
- safety under adversarial prompts or distribution shifts;
- that internal mathematical reasoning was changed rather than its typed
  answer representation.

The deterministic target digits are supplied externally. The mechanism writes
the requested output representation; it does not itself compute the sum.

## Next research phase

Turn the successful experiment into a reusable open-source workflow:

1. extract dataset, capture, basis fitting, prototype fitting, gating, controls,
   and gate verification into stable library APIs;
2. expose a declarative experiment specification rather than phase-specific
   scripts;
3. reproduce the interface on a second open-weight model;
4. test other typed latent variables and longer output grammars;
5. separate reasoning-state interventions from answer-channel interventions;
6. publish signed manifests containing all hashes and one-shot evaluation
   provenance.

The audit result is final. Any later experiments are new studies, not repairs
to this audit.
