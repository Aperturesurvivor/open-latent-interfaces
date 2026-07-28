# Phase 9C Phi Iterative Compiler Summary

## Result

The bounded prompt-local compiler passed its frozen exposed-selection gate.
Three relinearizations changed the requested counterfactual leading digit on
all 180 examples while preserving all 180 identity cases.

The selected condition achieved:

- target accuracy: `180/180`;
- identity accuracy: `180/180`;
- wrong-target norm-matched control: `7/180`;
- random norm-matched control: `1/180`;
- target control advantage: `0.9611`;
- target digit-token rate: `1.0`;
- mean cumulative update norm: `0.0973` relative to the original residual.

The fourth allowed iteration produced exactly the same accuracy and mean norm.
This is the expected signature of the exact-zero success gate.

## Interpretation

The Phase 9B one-shot derivative was directionally causal but insufficient.
Relinearizing at the perturbed state resolved that failure completely on the
exposed selection distribution. The leading-token pathway is therefore better
modeled as a curved, locally traversable interface than as one universal
prototype vector.

This mechanism is deterministic and changes no model parameter. It is also
explicitly output-side: the requested token participates in the derivative
calculation. It does not reveal a natural-language thought or establish that
the path is a semantic reasoning circuit.

## Next boundary

The selected compiler may now replace the failed leading prototype in exposed
integrated development. The independently passing wide-distribution tens and
ones writers remain fixed at hidden index 30, rank 32, scale 1.0.

No generalization claim is authorized. The next claim-bearing experiment must
freeze the complete reader → deterministic addition → hybrid writer pipeline
and evaluate it once on a newly generated pair-disjoint audit corpus.
