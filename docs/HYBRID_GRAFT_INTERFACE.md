# Hybrid Graft Interface

## Purpose

The hybrid-graft interface binds an audited latent reader, an explicit
deterministic mechanism, and bounded hidden-state writers into one
reproducible contract.

All three audited packages perform three operations:

1. decode decimal operand digits from hidden-state index 1;
2. add the reconstructed integers with ordinary host-language integer
   addition;
3. write the three-digit result through bounded target-conditioned residual
   mechanisms.

No model parameter is trained or changed.

The model-specific write boundaries are:

| Package | Leading compiler | Iterations | Suffix writer |
|---|---:|---:|---:|
| Phi-3.5-mini | hidden-state index 24 | 3 | index 30, rank 32, scale 1.0 |
| Qwen2.5-1.5B | hidden-state index 23 | 3 | index 27, rank 16, scales 1.25/2.0 |
| SmolLM2-1.7B | hidden-state index 24 | 1 | position-local compilers at index 24, margins 4/8 |

Manifest schema v2 adds per-position suffix scales and artifacts plus
correction-free evidence sources. The validator remains backward-compatible
with the released Phi v1 schema.

SmolLM2 uses
`oli.compiler-graft-interface/v1`, a sibling schema for sequential
prompt-local compilation at all three output positions. Its validator also
requires the preserved nonpass from the native suffix-coordinate search.

## Validation

From the repository root:

```bash
oli-hybrid-graft \
  manifests/phi35-mini-hybrid-arithmetic-graft-v1.json \
  --root .

oli-hybrid-graft \
  manifests/qwen25-15b-hybrid-arithmetic-graft-v1.json \
  --root .

oli-compiler-graft \
  manifests/smollm2-17b-compiler-arithmetic-graft-v1.json \
  --root .
```

The validator checks:

- model, residual-width, prompt, locator, and deterministic-operation
  contracts;
- the nested audited operand-reader manifest and artifact;
- compiler source hash and positive iteration/margin/norm parameters;
- suffix basis and prototype hashes, keys, ranks, and tensor shapes;
- development, optional correction, dataset, compiler-selection, audit-config,
  and audit-result hashes as required by the manifest version;
- exactly one passing audit run and its recorded reader/final accuracy.
- for v2, prospective compiler-depth selection and arbitrary shuffled-target
  accuracy over the norm-matched random control.
- for the compiler-graft schema, all three prospective position selections,
  the earlier native-coordinate nonpass, identity preservation, base-error
  recovery, and shuffled-target specificity.

Python callers can load the components with:

```python
from pathlib import Path

from open_latent_interfaces import HybridGraftManifest

root = Path(".")
manifest = HybridGraftManifest.load(
    root / "manifests/qwen25-15b-hybrid-arithmetic-graft-v1.json"
)
manifest.verify(root)
reader = manifest.load_reader(root)
suffix_basis, suffix_prototypes = manifest.load_suffix_components(root)
```

`compile_iterative_margin_deltas` supplies both leading-writer
implementations. The audited orchestration, including closed-loop token
prefixes and controls, is preserved in
`scripts/run_phase9e_hybrid_graft_audit.py` for Phi and
`scripts/run_phase11_qwen_hybrid_graft_audit.py` for Qwen. The SmolLM2
compiler orchestration is preserved in
`scripts/run_phase13_smollm2_compiler_graft_audit.py`.

The Qwen release bundle can be rebuilt deterministically:

```bash
uv run python scripts/package_phase12_qwen_hybrid_release.py \
  --output-dir /tmp/qwen-hybrid-release

cd /tmp/qwen-hybrid-release
shasum -a 256 -c SHA256SUMS
```

The packager validates both Qwen manifests before copying anything and embeds
the complete repository-relative evidence tree in a fixed-timestamp ZIP.

The SmolLM2 release can be rebuilt with:

```bash
uv run python scripts/package_phase13_smollm2_compiler_release.py \
  --output-dir /tmp/smollm2-compiler-release

cd /tmp/smollm2-compiler-release
shasum -a 256 -c SHA256SUMS
```

## Audited evidence

On 90 pair- and template-disjoint additions, Phi achieved:

- reader and deterministic compute: `90/90`;
- complete latent and oracle answers: `90/90`;
- unmodified base: `80/90`;
- base-error recovery: `10/10`;
- base-correct preservation: `80/80`;
- norm-matched random recovery: `0/10`;
- shuffled requested-target following: `82/90`;
- shuffled true answers: `0/90`.

On a separately sealed set of 90 pair- and template-disjoint additions, Qwen
achieved:

- reader and deterministic compute: `90/90`;
- complete latent and oracle answers: `90/90`;
- unmodified base: `59/90`;
- base-error recovery: `31/31`;
- base-correct preservation: `59/59`;
- norm-matched random recovery: `1/31`;
- wrong-target recovery: `2/31`;
- shuffled requested-target following: `85/90`;
- shuffled random target following: `1/90`;
- shuffled target-following advantage: `93.3` percentage points;
- shuffled true answers: `0/90`.

On a third sealed set of 90 pair- and template-disjoint additions, SmolLM2
achieved:

- reader and deterministic compute: `90/90`;
- complete latent and oracle answers: `90/90`;
- unmodified base: `33/90`;
- base-error recovery: `57/57`;
- base-correct preservation: `33/33`;
- norm-matched random recovery: `1/57`;
- wrong-target recovery: `2/57`;
- shuffled requested-target following: `83/90`;
- shuffled random target following: `0/90`;
- shuffled target-following advantage: `92.2` percentage points;
- shuffled true answers: `0/90`.

## Claim boundary

These are audited deterministic arithmetic grafts for one frozen revision
each of Phi, Qwen, and SmolLM2, three-digit addition, the `Answer=` contract,
and an external operand-span locator. Each prompt-local compiler uses the
requested output token and frozen downstream Jacobian, so it is an output-side
causal compiler—not evidence of a single natural “math neuron” or a decoded
chain of thought.

The three audits establish a cross-family replication of the workflow, not
portable tensors: each model required its own reader fit, boundary discovery,
writer selection, model-specific write mechanism, development gate, and
sealed audit.
Cross-operation, longer-answer, autonomous-locator, free-form-prompt, and
additional-model replications remain future work.
