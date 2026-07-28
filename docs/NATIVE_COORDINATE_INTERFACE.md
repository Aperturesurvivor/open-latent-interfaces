# Native Coordinate Interface

## Purpose

The native-coordinate API packages the smallest mechanism that survived the
Phase 2 one-shot audit:

1. project the current residual state into a frozen causal basis;
2. look up the native coordinate prototype for a requested class;
3. replace the current coordinates with that prototype;
4. reconstruct the residual delta;
5. cap its norm;
6. emit exactly zero when the unmodified model already predicts the requested
   token.

The API is independent of a particular model width, layer count, class count,
or answer position. Model-specific facts live in a declarative manifest and
hashed tensor artifacts. Manifest v2 supports a different causal basis at each
answer position while preserving v1's shared-basis contract.

## Equation

For residual state `h`, row-orthonormal basis `B`, class prototype `p[d]`,
scale `s`, and hard gate `g`:

`raw_delta = (p[d] - h B^T) B`

`delta = 0`, if the base argmax already equals the requested token.

Otherwise:

`delta = norm_cap(s * raw_delta, ||h||)`

This operation changes only coordinates in the declared subspace.

## Python API

```python
from pathlib import Path

from open_latent_interfaces import NativeCoordinateManifest

root = Path(".")
manifest = NativeCoordinateManifest.load(
    root / "manifests/phi35-mini-next-digit-interface-v1.json"
)
manifest.verify(root)
writer = manifest.load_writer(0, root=root)

write = writer.write(
    states,
    requested_digits,
    base_logits=base_logits,
    requested_token_ids=requested_token_ids,
)
delta = write.delta
hard_gate = write.hard_gate
```

The caller remains responsible for:

- capturing `states` at the manifest's hidden-state index;
- computing unmodified `base_logits` on the same prompt and prefix;
- mapping requested classes to tokenizer token IDs;
- adding `delta` at the declared residual boundary;
- preserving the model and tokenizer revisions named by the manifest.

## Fitting prototypes

```python
from open_latent_interfaces import fit_digit_prototypes

prototypes, counts = fit_digit_prototypes(
    fit_states,
    fit_digit_labels,
    basis,
    class_count=10,
)
```

Every class must be represented. Prototype fitting averages native residual
coordinates; it does not train the base model or a neural adapter.

## Manifest validation

```bash
oli-native-interface \
  manifests/qwen25-15b-next-digit-interface-v1.json \
  --root .
```

Validation checks:

- schema version and required manifest fields;
- lowercase SHA-256 syntax;
- existence and hash of the basis and prototype artifacts;
- basis width and per-position rank;
- prototype tensor shape `[10, rank]`;
- passing one-shot audit provenance and its result/config hashes.

The JSON Schema is
[`schemas/native-coordinate-interface-v1.schema.json`](../schemas/native-coordinate-interface-v1.schema.json).
The position-specific v2 contract is
[`schemas/native-coordinate-interface-v2.schema.json`](../schemas/native-coordinate-interface-v2.schema.json).

## Audited Qwen interface

The first manifest declares:

- model: `Qwen/Qwen2.5-1.5B-Instruct`;
- residual width: 1536;
- shared basis rank: 16;
- hidden-state index: 27;
- tens scale: 1.25;
- ones scale: 2.0;
- hard gate: base argmax equals requested digit token;
- one-shot audit result: 88/90 exact complete targets.

Position zero is recorded as an external causal-adapter component because it
has not yet been converted to the shared coordinate interface.

## Audited Phi interface

The v2 Phi manifest declares:

- model: `microsoft/Phi-3.5-mini-instruct`;
- residual width: 3072;
- leading basis: rank 32 at hidden index 24, scale 1.0;
- shared suffix basis: rank 32 at hidden index 30, scale 1.25;
- explicit assistant-prefix token contract: `Answer=`;
- one-shot audit result: 70/90 exact complete targets;
- all three answer positions implemented as native-coordinate writers.

The compact 791 KB tensor package is available from the
[`phase3-phi-native-coordinate-audit-v1` release](https://github.com/Aperturesurvivor/open-latent-interfaces/releases/tag/phase3-phi-native-coordinate-audit-v1).
Its SHA-256 is recorded independently in every position's basis and prototype
reference.

## Claim boundary

A valid manifest proves artifact integrity and records evidence. It does not
make an interface universal.

Cross-model use requires a new discovery and validation run unless basis and
prototype compatibility are independently demonstrated. Token IDs are
tokenizer-specific, and residual coordinates are model-revision-specific.
