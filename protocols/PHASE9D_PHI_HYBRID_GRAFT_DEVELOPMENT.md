# Phase 9D: Phi Hybrid Graft Development

## Purpose

Evaluate the complete exposed-development pipeline after replacing its only
localized writer bottleneck:

1. decode both operands from hidden-state index 1 with the independently
   audited nearest-centroid reader;
2. add the decoded integers with an external deterministic mechanism;
3. compile the leading answer token at hidden-state index 24 with the selected
   three-step prompt-local causal compiler;
4. write the tens and ones tokens at hidden-state index 30 with the selected
   wide-distribution rank-32 native coordinates.

Every answer position is generated closed loop. Later positions receive only
the tokens actually produced by earlier positions.

## Frozen sources

- Phase 8 dataset config SHA-256:
  `2cefb24b966ff9423dbb04be6b97f7633da09b0d64b894302ca8acf856c21aa2`;
- operand reader artifact SHA-256:
  `58f84aeda73713e9eb2e8ed0347639fc84f60273ad69557d7718b096cd6ac0c0`;
- iterative compiler result SHA-256:
  `6776b9e315e6ceb56e3e61019774b9520e8ee2f7f85aa895907d117342b26447`;
- wide writer selection SHA-256:
  `c6a57224e2824845e2c7b60d449b3d0b58df9306aad69c15ee2fefee719acfb6`;
- wide prototype artifact SHA-256:
  `62d7302c7bcfe7ebe529f4f20ab91f43b61ebc8bda8bf2df86ccb053868122f0`;
- audited native basis SHA-256:
  `960aac22478678a81a9677f03c0d7e885c60a603f22aa3737e0a19bb41803faa`.

The compiler is fixed at hidden index 24, margin 8.0, three
relinearizations, and cumulative norm cap 0.75. Both suffix positions are
fixed at hidden index 30, rank 32, scale 1.0, and norm cap 1.0.

## Conditions

- unmodified greedy base;
- oracle integer addition followed by the hybrid writer;
- latent operand read, deterministic addition, and hybrid writer;
- cyclically shuffled decoded targets;
- norm-matched random leading and suffix residual updates;
- norm-matched deterministic wrong-target compilation and suffix writes.

The oracle and latent conditions may share the same compiled trace only when
the decoded integer target lists are exactly equal. The result records both
conditions independently.

## Frozen development gate

The pipeline advances only if:

- operand-pair accuracy is at least 0.98;
- deterministic computed-target accuracy is at least 0.98;
- latent full-result exact accuracy is at least 0.95;
- latent-oracle exact gap is at most 0.05;
- at least 0.75 of base errors are recovered;
- at least 0.98 of base-correct rows are preserved;
- net exact improvement over base is at least 0.10;
- base-error recovery exceeds the random control by at least 0.50;
- shuffled-target true accuracy is at most 0.25;
- wrong-target true accuracy is at most 0.25;
- parse and digit-token rates are both 1.0.

This paired gate distinguishes recovery from merely preserving a naturally
correct base answer.

## Claim boundary

This run reuses an already exposed development split. A pass permits freezing
a new pair-disjoint audit only; it does not repair, repeat, or supersede the
closed Phase 8 audit.

## Frozen run outcome

The write-once development run produced:

- reader pair accuracy: `45/45`;
- deterministic computed-target accuracy: `45/45`;
- latent and oracle hybrid output: `45/45`;
- base output: `38/45`;
- latent recovery of base errors: `7/7`;
- latent preservation of base-correct rows: `38/38`;
- random-control recovery of base errors: `0/7`;
- shuffled-target output matching its requested target: `45/45`;
- shuffled-target true accuracy: `2/45`.

The original gate nevertheless recorded a non-pass because wrong-target true
accuracy was `38/45`, above the frozen maximum `0.25`. The wrong-target
condition recovered `0/7` base errors and preserved `38/38` base-correct rows.
This is the expected consequence of norm-matching against a hard-gated target
update: target-update norm is zero on already-correct rows, so the
wrong-target control also applies zero there. Absolute true accuracy therefore
measures base preservation rather than wrong-direction recovery.

The original result is immutable at
`results/phase9d_phi_hybrid_graft_development.json`, SHA-256
`a41a28814499e0eaf0c3190d0aba5ade98070c2b35bd63d070d8216257fad02a`.

A correction, if performed, must not rerun the model or change any output. It
may replace only the structurally mismatched absolute wrong-target check with
paired checks frozen against the existing rows:

- wrong-target base-error recovery at most 0.25;
- latent recovery advantage over wrong-target at least 0.50.

The original non-pass remains part of the record.
