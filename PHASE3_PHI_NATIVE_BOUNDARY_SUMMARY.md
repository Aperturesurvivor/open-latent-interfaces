# Phase 3 Phi Native-Boundary Summary

## Outcome

Full-native fit-donor transport passed the frozen selection gate independently
at all three answer positions in Phi-3.5:

| Position | Selected hidden index | Target digit | Strongest control | Advantage | Mean relative norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Leading | 24 | 90/90 | 9/90 | 90.0 points | 66.72% |
| Tens | 30 | 90/90 | 9/90 | 90.0 points | 80.60% |
| Ones | 30 | 90/90 | 11/90 | 87.78 points | 66.08% |

The experiment used 428 fit donors whose complete prefilled answer was correct.
All 90 selection targets were balanced and changed every result digit. Base,
wrong-digit, shuffled-donor, and random controls were evaluated without
removing any recipient.

## Transition map

The leading-digit channel appeared earlier than the suffix channels:

- index 17: 5/90 targeted leading digits;
- index 21: 73/90;
- index 24: 90/90.

For tens and ones, index 21 reached 29/90 and 22/90, index 24 reached 70/90 and
81/90, and index 30 reached 90/90 at both positions. This differs from Qwen's
shared selected hidden index 27 and therefore supports rediscovery rather than
blind parameter transfer.

## Final-normalization caveat

Index 32 is not coordinate-compatible in the current generic capture/hook
pair. Phi applies a final RMS normalization after decoder block 31. Hugging
Face's returned last hidden state is post-normalization, while the generic
index-32 intervention hook modifies the block output before that
normalization. The resulting delta mixes two coordinate systems and the
index-32 measurements must not be interpreted as a biological-looking
late-layer collapse.

The selected indices 24 and 30 are ordinary decoder-block boundaries and are
not affected. Future scans must either omit the terminal hidden-state index or
capture the pre-normalization final block output with a matching hook.

## Frozen provenance

- frozen experiment commit: `349a3eb`
- configuration SHA-256:
  `918bcbbd02744e592e4218f4048469a921d4c358655983a168f64854c3a71bc8`
- result SHA-256:
  `aac7a27200379c47407644edf9adc40c4c30c88958c8e07157339bf2fe962a81`
- target SHA-256:
  `54f954934aae1c812667ea60e68eb49f0a0edd02478ac1e0a6f0f5e3d64be493`
- model revision:
  `2fe192450127e6a83f7441aef6e3ca586c338b77`
- runtime: Python 3.12.12, PyTorch 2.13.0, MPS, float16
- elapsed causal-evaluation time: 637.79 seconds

## Decision and claim boundary

Proceed to intrinsic-rank estimation at hidden index 24 for the leading
position and index 30 for tens and ones. This is a selection-only causal upper
bound using full native donor states. It is not yet a compact or donor-free
Phi interface, and no development or audit example was evaluated.
