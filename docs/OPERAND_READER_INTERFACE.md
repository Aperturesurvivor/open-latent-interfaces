# Operand Reader Interface

## Purpose

The operand-reader interface decodes decimal digit values from frozen model
hidden states at externally located operand-token positions. It is the first
audited read bridge in the repository.

The audited Phi and Qwen readers both use:

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
    root / "manifests/qwen25-15b-operand-reader-v1.json"
)
manifest.verify(root)
reader = manifest.load_reader(root)

predicted_digits = reader.predict(selected_hidden_states)
```

`selected_hidden_states` has shape `[digit_tokens, residual_width]`—3072 for
Phi and 1536 for Qwen—and must contain
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

oli-operand-reader \
  manifests/qwen25-15b-operand-reader-v1.json \
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

The initial Phase 8 Phi read-compute-write audit was an overall nonpass because
the earlier answer writer did not recover enough base-model errors. The reader
component passed independently and later participated in the passing Phase 9E
hybrid audit.

The Qwen reader selected on 180 held-out examples at 180/180 pairs and 988/988
digits, with 0/180 for the rotated-label control. It then decoded 90/90 unseen
operand pairs and 498/498 digits in the pair- and template-disjoint Phase 12
one-shot hybrid audit.

Schema v2 represents correction-free evidence as a typed source map. The
runtime remains backward-compatible with the Phi v1 manifest.

## Claim boundary

The manifests establish token-local decimal digit decoding under an external
operand-span locator for one frozen revision each of Phi and Qwen. They do not
autonomously discover operands, infer roles from unrestricted free-form text,
expose hidden chain of thought, or transfer centroids between models.
