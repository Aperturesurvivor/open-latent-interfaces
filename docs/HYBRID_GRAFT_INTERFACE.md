# Hybrid Graft Interface

## Purpose

The hybrid-graft interface binds an audited latent reader, an explicit
deterministic mechanism, and bounded hidden-state writers into one
reproducible contract.

The Phi v1 package performs three operations:

1. decode decimal operand digits from hidden-state index 1;
2. add the reconstructed integers with ordinary host-language integer
   addition;
3. write the three-digit result through an iterative leading-token compiler
   at hidden-state index 24 and fixed native suffix coordinates at index 30.

No model parameter is trained or changed.

## Validation

From the repository root:

```bash
oli-hybrid-graft \
  manifests/phi35-mini-hybrid-arithmetic-graft-v1.json \
  --root .
```

The validator checks:

- model, residual-width, prompt, locator, and deterministic-operation
  contracts;
- the nested audited operand-reader manifest and artifact;
- compiler source hash and positive iteration/margin/norm parameters;
- suffix basis and prototype hashes, keys, ranks, and tensor shapes;
- development, correction, dataset, audit-config, and audit-result hashes;
- exactly one passing audit run and its recorded reader/final accuracy.

Python callers can load the components with:

```python
from pathlib import Path

from open_latent_interfaces import HybridGraftManifest

root = Path(".")
manifest = HybridGraftManifest.load(
    root / "manifests/phi35-mini-hybrid-arithmetic-graft-v1.json"
)
manifest.verify(root)
reader = manifest.load_reader(root)
suffix_basis, suffix_prototypes = manifest.load_suffix_components(root)
```

`compile_iterative_margin_deltas` supplies the leading writer implementation.
The audited orchestration, including closed-loop token prefixes and controls,
is preserved in `scripts/run_phase9e_hybrid_graft_audit.py`.

## Audited evidence

On 90 pair- and template-disjoint additions:

- reader and deterministic compute: `90/90`;
- complete latent and oracle answers: `90/90`;
- unmodified base: `80/90`;
- base-error recovery: `10/10`;
- base-correct preservation: `80/80`;
- norm-matched random recovery: `0/10`;
- shuffled requested-target following: `82/90`;
- shuffled true answers: `0/90`.

## Claim boundary

This is an audited deterministic arithmetic graft for one Phi revision,
three-digit addition, the `Answer=` contract, and an external operand-span
locator. The leading compiler uses the requested output token and frozen
downstream Jacobian, so it is an output-side causal compiler—not evidence of a
single natural “math neuron” or a decoded chain of thought.

The reusable contribution is the workflow and typed interface. Cross-model,
cross-operation, longer-answer, autonomous-locator, and free-form-prompt
replications remain future work.
