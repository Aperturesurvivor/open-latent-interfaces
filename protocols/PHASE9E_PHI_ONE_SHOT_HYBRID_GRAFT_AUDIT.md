# Phase 9E: One-Shot Phi Hybrid-Graft Audit

## Authorization

This protocol authorizes exactly one execution producing
`results/phase9e_phi_hybrid_graft_audit.json`. The runner refuses any other
output path and refuses to overwrite the result.

The 90-example corpus was generated and committed while its dataset
authorization flag was false. No model behavior on those rows was inspected
before this audit configuration froze.

## Frozen pipeline

- operand reader: hidden-state index 1, nearest full-width digit centroid;
- deterministic mechanism: host integer addition over the two decoded values;
- leading writer: hidden-state index 24, margin 8.0, three prompt-local
  relinearizations, cumulative norm cap 0.75;
- tens and ones writers: hidden-state index 30, audited rank-32 suffix basis,
  wide prototypes, scale 1.0, norm cap 1.0;
- generation: greedy and closed loop, with no teacher-forced answer token;
- random seed: 20261206.

The runner, imported execution dependencies, dataset generator, source
artifacts, selected-component results, development result, correction config,
correction result, rendered prompts, operand positions, and token contract are
all hash-locked in the audit config.

## Conditions

- unmodified greedy base;
- oracle integer target through the hybrid writer;
- latent operand read, deterministic addition, and hybrid writer;
- cyclically shuffled latent targets through the same full-strength writer;
- random residual directions norm-matched independently at every answer
  position to the true-target writer.

Shuffled target following measures whether the interface writes requested
content rather than merely improving arithmetic. Shuffled true accuracy
serves as the corresponding specificity control. Random updates test whether
base-error recovery is direction-specific.

## Frozen gate

The audit passes only if:

- reader pair accuracy is at least 0.98;
- decoded deterministic target accuracy is at least 0.98;
- latent and oracle exact accuracy are each at least 0.95;
- latent-oracle exact gap is at most 0.03;
- every latent answer position is at least 0.95 accurate;
- at least 0.98 of base-correct rows are preserved;
- net exact change relative to base is nonnegative;
- shuffled requested-target exact accuracy is at least 0.90;
- shuffled true accuracy is at most 0.15;
- parse and decimal-token rates are 1.0;
- observed mean leading and suffix relative norms remain within their frozen
  per-row caps.

If the base has at least one error, the latent graft must recover at least
0.80 of base errors and exceed random-control recovery by at least 0.50. If
the base has zero errors, those ratios are undefined; the predeclared
alternative branch requires latent accuracy and base-correct preservation
both equal 1.0.

## Claim boundary

A pass supports a one-model claim that the frozen workflow can decode
operands from an early residual stream, route an externally computed integer
result back through a bounded intermediate-state compiler and native suffix
coordinates, and generate the requested answer on a pair- and
template-disjoint corpus.

It does not show that Phi naturally used the external deterministic
mechanism, that the compiled path is a stable semantic neuron set, that
natural-language thoughts were decoded, or that the specific vectors
transfer to another model.

