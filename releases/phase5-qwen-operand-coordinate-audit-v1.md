# Qwen 2.5 1.5B audited operand coordinate v1

This release publishes the independently audited, donor-free operand-increment
coordinate for `Qwen/Qwen2.5-1.5B-Instruct` at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

The sealed audit reached 43/45 exact target answers. The norm-matched
wrong-class and random controls each reached 0/45 exact. The included artifact
contains four source-digit-conditioned operand vectors for source ones digits
1 through 4, plus labels and fit counts.

The Qwen carry writer did not pass its independent audit and is intentionally
excluded. This release makes no Qwen carry-coordinate claim.

- Artifact:
  `phase5_qwen_operand_coordinate_v1.safetensors`
- SHA-256:
  `adaeb2d34cb7694d9ab08bb971173730addd056eac0fe0bb69d8e1f6e28a58a2`
- Size: 25,216 bytes
- Hidden-state index: 12
- Scale: 1.0
- Residual width: 1,536
- License: Apache-2.0 for repository code and metadata; the upstream model
  retains its own license.
