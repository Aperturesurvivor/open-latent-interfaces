# Phase 1D Stepwise Native-Write Summary

## Outcome

**Closed-loop full-result native writing passed on the development split.**

At each of three answer positions, the experiment replaced the recipient's
residual state at HF hidden state 23 / decoder block 22 with the state from a
matched donor prompt carrying a counterfactual three-digit result. The model
then greedily emitted one token, and the process repeated using the generated
prefix.

The targeted condition generated the complete donor result on 38/45 examples
(84.4%). Base, same-leading norm-matched, and random norm-matched controls
generated 0/45 target results; a shuffled-donor norm-matched control generated
1/45.

## Frozen design

- Model: Qwen2.5-1.5B-Instruct at the previously frozen revision.
- Dataset: 45 development examples; all 45 audit examples remained unopened.
- Boundary: hidden state 23 / decoder block 22, selected by the committed
  all-layer donor-patch result.
- Target: the complete result associated with the preselected matched donor.
- Decoding: greedy and closed-loop for exactly three single-digit tokens.
- Controls: base, same-leading donor, shuffled donor, and random direction.
- All non-base controls were norm-matched per example and answer position.

The protocol was committed as `dd5e3b9` before execution. The result artifact
has SHA-256
`f9a232bd60d1435c9ae386e472334e33643528fddf2e6c26fb0719eefe982346`.

## Results

| Condition | Digit 1 | Digit 2 | Digit 3 | Exact target | Original result |
|---|---:|---:|---:|---:|---:|
| Base | 0.0% | 31.1% | 11.1% | 0/45 | 43/45 |
| Random norm-matched | 4.4% | 31.1% | 11.1% | 0/45 | 32/45 |
| Same-leading norm-matched | 0.0% | 26.7% | 15.6% | 0/45 | 2/45 |
| Shuffled donor norm-matched | 4.4% | 15.6% | 6.7% | 1/45 | 1/45 |
| **Targeted donor** | **93.3%** | **93.3%** | **97.8%** | **38/45** | **0/45** |

The mean intervention-to-recipient residual norm in the targeted condition was
62.8%, 69.7%, and 59.3% across the three positions. The model produced a
parseable three-digit integer in every condition and every example.

The seven targeted failures were local digit errors rather than malformed
outputs:

```text
928 → 908
961 → 901
214 → 114
906 → 806
109 → 100
961 → 901
214 → 114
```

## Interpretation

This result causally demonstrates that the model exposes a reusable native
write boundary for sequential answer digits. Replacing the state at that
boundary can control the next digit, and recomputing the replacement after each
generated prefix can compose those one-token effects into a complete
counterfactual answer.

The control separation matters. Equivalent-norm perturbations, same-leading
donors, and donor states assigned to the wrong recipient do not reproduce the
effect. The write depends on the content and alignment of the native donor
state, not merely on intervention magnitude or generic disruption.

## Claim boundary

This is not yet a deterministic reasoning implant:

- each write uses a full donor residual state from another model execution;
- the interventions are large and carry uncontrolled latent information;
- the deterministic mechanism does not yet compute and encode its own answer;
- no compact or cross-prompt write bridge has passed;
- the result is development-only;
- the frozen audit split remains unopened.

The demonstrated object is therefore a **causal native sequential write path**,
not yet a portable Natural Language Adapter or audited deterministic graft.

## Next experiment

Compress the full donor replacement into a typed, low-rank digit write:

1. estimate answer-position-conditioned digit subspaces from native donor
   deltas;
2. select rank and scale on development data only;
3. freeze the writer before audit;
4. compare targeted writes against random-subspace, shuffled-label,
   same-digit, and full-residual controls;
5. measure target-digit transfer, preservation, collateral divergence, and
   exact three-digit composition.

Passing that test would replace donor-state copying with a compact interface
that can accept digits from an external deterministic calculator.
