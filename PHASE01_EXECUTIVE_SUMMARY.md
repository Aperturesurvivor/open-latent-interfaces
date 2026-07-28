# Phase 0.1 Executive Summary

Phase 0.1 completed a separated development and frozen audit on
Qwen2.5-0.5B. It corrected the original pilot's digit imbalance and template
leakage, compared six norm-matched write directions across five layers, and
then opened a previously unused compact-template audit only after committing
the selected configuration.

The primary probe-defined internal write condition **failed**.

## Development

The dataset contained 720 prompts:

- 12 train, 4 development, and 4 audit operand pairs for each leading result
  digit 1–9;
- one addition and three matched contrasts per pair;
- exact operand pairs disjoint across splits;
- train templates: direct/calculation;
- development template: word problem;
- audit template: compact sum request.

Development showed that the optimistic Phase 0 signal did not survive the
word-problem template shift:

- leading-digit probe accuracy was 11–14% across layers, approximately chance;
- scalar-sum decoding was negative R² at most layers and reached only 0.355 at
  hidden-state index 19, with no exact rounded sums;
- route AUC reached 1.0 at index 19 but was poor at several earlier layers;
- the final residual boundary supported strong digit-logit steering, as
  expected for an output-adjacent direction, but was excluded from internal
  selection by protocol.

The frozen selection rule chose hidden-state index 10 and strength 4. Its
development targeted-control margin advantage was only +0.040 logits and it
changed no top-1 decisions.

## Frozen audit

The exact configuration was committed before audit as
`configs/phase01_frozen.json`. Audit regenerated and verified the full dataset
hash, reloaded the exact model revision, refit probes on train plus
development, and captured only the selected boundary.

Probe results:

| Endpoint | Audit result |
|---|---:|
| Route accuracy | 64.6% |
| Route balanced accuracy | 76.4% |
| Route AUC | 1.000 |
| Leading-digit accuracy | 13.9% |
| Scalar-sum R² | -0.314 |
| Scalar-sum rounded exact | 0/36 |

The perfect route AUC paired with poor fixed-threshold accuracy demonstrates a
substantial score-distribution shift across the unseen compact template. It is
not a robust router.

Causal first-digit results:

| Condition | Top-1 | Margin change |
|---|---:|---:|
| Frozen base | 32/36 | — |
| Targeted digit probe | 33/36 | -0.132 |
| Wrong-digit probe control | 32/36 | -0.184 |
| Random direction control | 32/36 | -0.010 |
| Scalar-sum probe | 32/36 | +0.056 |
| Same-digit donor | 33/36 | +0.063 |
| Digit-logit direction | 32/36 | -0.016 |

The frozen primary targeted-control advantage was **-0.123 logits**. Although
the targeted condition changed one top-1 decision, it damaged the aggregate
correct-digit margin much more than random intervention. It therefore failed
the causal gate.

The same-digit donor direction is a non-primary clue: it changed the same
number of top-1 decisions while modestly improving margin. This result may
motivate a donor-subspace or nonlinear native-state experiment, but it cannot
rescue the failed primary condition.

## Conclusion

Phase 0.1 is an infrastructure and experimental-discipline success with a
negative primary scientific result:

- simple linear route/value probes did not transfer cleanly across template
  families;
- exact typed result information was not recovered;
- the selected internal digit-probe direction did not provide reliable causal
  control;
- output-adjacent steering remains easy but does not answer the latent-interface
  question.

The next justified step is triangulated cartography using donor subspaces,
Jacobian/logit transport, NLA explanations/reconstructions, and activation
patching—not a larger claim about deterministic grafts.

