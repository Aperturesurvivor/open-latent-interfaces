# Phase 0.1 Protocol: Balanced, Template-Held-Out Cartography

Status: **development protocol frozen before audit**

## Purpose

Phase 0 found approximate result information and a weak late-layer causal
signal, but its route task had lexical overlap across splits and its
leading-digit support was uneven. Phase 0.1 removes those confounds and
separates development from a one-shot audit.

This remains a first-result-digit diagnostic. It is not an end-to-end
deterministic graft.

## Frozen target

- Default model: `Qwen/Qwen2.5-0.5B-Instruct`
- Every target-model parameter remains frozen.
- Audit reloads the exact resolved model revision written by development.
- Hidden-state index `k` means the output of decoder block `k - 1`.

## Dataset

Every split is exactly balanced over leading result digits 1–9.

- train: 12 operand pairs per digit;
- development: 4 pairs per digit;
- audit: 4 pairs per digit;
- every pair receives one addition and three matched contrasts:
  multiplication, quoted addition, and unrelated numerical facts;
- exact ordered operand pairs are disjoint across all splits;
- train uses only `direct` and `calculate` templates;
- development uses only `word_problem`;
- audit uses only the unseen `compact` template.

The generator and split hashes are frozen in the configuration artifact.

## Development

Inspect five evenly spaced residual boundaries. At each boundary:

1. fit route, scalar-sum, and ten-class leading-digit ridge probes on train;
2. evaluate all probes on development;
3. compute the externally exact sum and leading digit;
4. compare six norm-matched intervention directions:
   - desired leading-digit probe;
   - deterministic wrong-digit probe control;
   - random-direction control;
   - scalar-sum probe;
   - same-digit donor activation;
   - digit unembedding/logit direction;
5. sweep strengths 0.5, 1, 2, and 4.

The primary selection score is:

`targeted margin delta − max(wrong-digit margin delta, random margin delta)`

Tie breakers are top-1 gain over base, then smaller intervention norm.

To keep the selected interface internal rather than output-adjacent, only
boundaries at or before 80% of model depth are eligible. The final boundary is
still reported as a diagnostic.

Development writes a frozen JSON configuration. It does not load or evaluate
audit activations.

## Audit

The separate audit command:

1. verifies the complete dataset hash against the frozen configuration;
2. reloads the exact model revision;
3. refits probes on train plus development, as predeclared;
4. captures only the selected boundary;
5. evaluates route, scalar sum, leading digit, and all six causal conditions on
   the untouched compact-template audit;
6. performs no audit-driven layer, strength, direction, or bridge changes.

## Interpretation

- Route metrics remain probe decodability, not proof that the model uses the
  same route internally.
- Scalar R² does not imply exact typed values.
- A targeted advantage over wrong/random controls is causal evidence for a
  direction, not proof of a complete native interface.
- Same-digit donor and digit-logit directions are alternative write baselines,
  not negative controls.
- First-digit improvement is not multi-token exact arithmetic.
- The externally exact sum is an oracle result; operand readout is bypassed.
- NLA and J-lens explanations are not yet part of this phase.

## Reproduction

Development:

```bash
uv run oli-phase01 develop
```

Commit `configs/phase01_frozen.json` before audit.

Audit:

```bash
uv run oli-phase01 audit --config configs/phase01_frozen.json
```

Retain all artifacts, including failures.

