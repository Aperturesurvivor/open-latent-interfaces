# Arithmetic Coordinate Interface

## Purpose

This interface packages audited internal arithmetic-state transports rather
than answer-token writers. It currently exposes:

1. a class-conditioned edit of the first operand's ones digit;
2. a class-invariant rank-one carry transition at the second operand's
   contextualized ones digit.

Both are additive residual transports fitted without changing model weights.
Inference requires no donor activation.

## Runtime API

```python
from pathlib import Path

import torch

from open_latent_interfaces import ArithmeticCoordinateManifest

root = Path(".")
manifest = ArithmeticCoordinateManifest.load(
    root / "manifests/phi35-mini-arithmetic-coordinates-v1.json"
)
manifest.verify(root)

operand = manifest.load_writer("operand_increment", root=root)
operand_delta = operand.delta(class_labels=torch.tensor([1, 4]))

carry = manifest.load_writer("carry_on", root=root)
carry_delta = carry.delta(batch_size=2)
```

The returned tensors have shape `[batch, residual_width]`. The caller adds each
row once at the token and hidden-state boundary declared by the corresponding
manifest interface. `one_shot_sequence_residual_intervention` can apply a
padded token-local sequence delta during the prompt forward pass while leaving
generated tokens untouched.

## Token-selection contract

The manifest intentionally describes token locations semantically:

- `operand_increment`: first operand, ones place;
- `carry_on`: second operand, ones place.

A caller must resolve these locations under its exact renderer and tokenizer.
It must reject prompts where the decimal digit does not occupy one distinct
token or where the model/tokenizer revision differs. The audit runners
hash-locked this contract for every evaluated quartet.

## Deterministic graft

The audited write path can be used in a deterministic mechanism graft:

1. resolve and verify both operand digit tokens;
2. obtain the source digit from a trusted parser or a separately audited
   latent reader;
3. let a deterministic mechanism decide whether to request `operand_increment`
   or `carry_on`;
4. request the corresponding fixed residual delta;
5. place it at the declared token and boundary during the prompt forward pass;
6. let the frozen model generate without further intervention;
7. optionally use the separately audited next-digit interface to write an
   externally computed final answer.

The current package audits steps 3–6 when the request is externally supplied.
It does not yet audit a latent reader or a full automatic
latent-read → deterministic-compute → latent-write loop. That remains a
separate gate and must not be inferred from writer success.

## Validation

```bash
oli-arithmetic-interface \
  manifests/phi35-mini-arithmetic-coordinates-v1.json \
  --root .
```

Validation checks artifact hashes, tensor widths and class labels, boundary
metadata, passing corrected-development evidence, and the passing one-shot
audit chain.

## Audited evidence

The sealed audit used 45 unfiltered quartets:

- operand writer: 37/45 exact versus 0/45 wrong-class exact and 2/45 random;
- carry writer: 26/45 target tens versus 11/45 matched no-carry and 2/45
  random;
- all generated outputs parseable;
- repeat audit invocation refused.

The package is 74,448 bytes and contains five tensors. Its SHA-256 is
`9479f3652d6b250a8ff6ae375edfe87c801c8c899dd8c962c204acdb13eb79c4`.
It is published in the
[`phase4-phi-arithmetic-coordinate-audit-v1` release](https://github.com/Aperturesurvivor/open-latent-interfaces/releases/tag/phase4-phi-arithmetic-coordinate-audit-v1).

## Workflow portability

The reusable discovery workflow is:

1. freeze matched causal quartets and behavior gates;
2. reject single-token bottlenecks before escalating to sequence transport;
3. localize effects to semantic token regions under matched controls;
4. fit donor-free transports on behavior-correct fit data only;
5. keep selection, development, and audit unfiltered;
6. freeze layers, scales, token contracts, controls, and metrics;
7. validate once on untouched development;
8. authorize exactly one sealed audit;
9. package only passing coordinates with model revision and evidence hashes.

Vectors are not portable between models. The workflow and evidence contract
are the portable objects.
