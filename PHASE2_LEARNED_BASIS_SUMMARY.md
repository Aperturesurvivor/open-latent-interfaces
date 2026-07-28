# Phase 2 Learned-Basis Causal Adapter Summary

## Outcome

**Joint causal basis learning produced the strongest compressed writer so far
and passed preservation, norm, and parseability gates. Exactness and suffix
gates still failed, so the audit remains sealed.**

The multitemplate causal adapter's transport basis was promoted from a frozen
donor-PCA buffer to a trainable shared parameter. Adapter coefficients and
basis directions were optimized jointly under causal token, identity CE/KL,
view-consistency, norm, and orthogonality losses.

## Provenance

- Protocol commit: `017e3ae`.
- Result SHA-256:
  `92c4c8494ea1d66a7ed7d3885207436b011024fabef504c4cc7833905d523293`.
- Weights SHA-256:
  `d0fa911d2d95e4b5efc299f830a4ecc966bd94d5dec9368ec60042c9c508adb0`.
- Audit examples opened: 0/90.

## Development comparison

| Metric | Fixed basis | Learned basis |
|---|---:|---:|
| Digit 1 | 83/90 | 84/90 |
| Digit 2 | 38/90 | 44/90 |
| Digit 3 | 34/90 | 39/90 |
| Exact target | 18/90 | 21/90 |
| Identity preservation | 80/90 | 81/90 |
| Mean norms | 36%, 23%, 32% | 36%, 23%, 34% |
| Parse rate | 100% | 100% |

Exact matched controls reached at most 1/90.

The selected bases moved modestly from initialization. Frobenius changes were
0.49, 0.69, and 0.57 across the three positions. Small rotations therefore
produced measurable suffix improvements without sacrificing preservation.

## Frozen gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Exact target | ≥50% | 23.3% | Fail |
| Every position | ≥70% | 93.3%, 48.9%, 43.3% | Fail |
| Exact advantage over every control | ≥25 points | 22.2 points | Fail |
| Identity preservation | ≥90% | 90.0% | Pass |
| Relative norm at every position | ≤100% | 36%, 23%, 34% | Pass |
| Parse rate | 100% | 100% | Pass |

## Interpretation

The donor-PCA basis was a real constraint. Causal rotation improves both suffix
positions, exact composition, and preservation simultaneously.

The remaining training target is also biased. Counterfactual donors were
selected to change the leading digit while minimizing suffix distance. On
development, the untouched model already matches 53.3% of target second
digits. Causal suffix training therefore contains too few genuinely
counterfactual digit changes.

## Next experiment

Remove donor-selection bias from causal training:

1. deterministically map every correct result to a counterfactual three-digit
   target in which all three digits change;
2. balance target digit classes across fit examples;
3. train the learned-basis multitemplate adapter directly against those digits;
4. use the same transform on selection and development;
5. retain the current preservation, control, norm, and audit gates.

Native donors remain evidence for the write boundary, but causal adapter
training no longer needs donor results as labels.
