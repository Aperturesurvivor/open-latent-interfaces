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

