# Operand Reader Interface

## Purpose

The operand-reader interface decodes decimal digit values from frozen model
hidden states at externally located operand-token positions. It is the first
audited read bridge in the repository.

The Phi v1 reader uses:

- hidden-state index 1;
- one full-width native-state centroid per decimal digit;
- a fixed nearest-centroid linear decision rule;
- an external semantic locator that supplies ordered operand-A and operand-B
  decimal token positions.

No base-model weight changes, prompt-text digit reads, or token-ID digit reads
occur in the read bridge.

## Runtime

```python
from pathlib import Path

from open_latent_interfaces import OperandReaderManifest

root = Path(".")
manifest = OperandReaderManifest.load(
    root / "manifests/phi35-mini-operand-reader-v1.json"
)
manifest.verify(root)
reader = manifest.load_reader(root)

predicted_digits = reader.predict(selected_hidden_states)
```

`selected_hidden_states` has shape `[digit_tokens, 3072]` and must contain
states captured at hidden-state index 1 in the order declared by the locator
contract.

The helper `locate_operand_digit_tokens` verifies that each decimal character
occupies exactly one token inside the exact rendered user-prompt content. It is
an experimental external locator, not part of the learned reader.

## Validation

```bash
oli-operand-reader \
  manifests/phi35-mini-operand-reader-v1.json \
  --root .
```

Validation checks model metadata, tensor hashes and shapes, all fit counts, the
external-locator declaration, every evidence-file hash, one-shot audit count,
and the recorded audit pair accuracy.

## Evidence

- Selection: 180/180 exact operand pairs.
- Development: 45/45 exact operand pairs.
- Sealed audit: 45/45 exact operand pairs.
- Rotated-label selection control: 0/180 exact pairs.
- Audit run count: one; repeat invocation refused.

The containing read-compute-write audit was an overall nonpass because the
previously audited answer writer did not recover enough base-model errors on
the wider distribution. That does not invalidate the independently measured
reader component, and the manifest states this limitation explicitly.

## Claim boundary

The v1 interface establishes token-local decimal digit decoding under an
external operand-span locator for one Phi revision and the frozen Phase 8
prompt families. It does not autonomously discover operands, infer roles from
free-form text, expose hidden chain of thought, or transfer its centroids to a
different model.
