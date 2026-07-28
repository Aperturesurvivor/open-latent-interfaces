# Phase 9D Phi Hybrid Graft Summary

## End-to-end development result

The complete latent graft reached `45/45` exact answers on the exposed
development split:

- hidden-state operand reader: `45/45` operand pairs;
- deterministic integer addition: `45/45`;
- leading causal compiler: `45/45` generated leading tokens;
- wide native suffix writer: `45/45` tens and `45/45` ones tokens;
- complete latent and oracle output: `45/45`.

The unmodified model produced `38/45`. The graft recovered all seven base
errors and preserved all 38 base-correct rows. The norm-matched random control
recovered none of the seven errors. A cyclic shuffled target was written
exactly on all 45 rows but matched the true answer on only two, separating
requested content from generic perturbation.

## Gate record

The original frozen gate recorded a non-pass because it incorrectly required
low aggregate true accuracy from a wrong-target control whose update norm is
zero on hard-gated base-correct rows. That original result is retained.

A separately frozen, no-inference correction replaced only that mismatched
absolute check with paired base-error recovery checks. Wrong-target updates
recovered `0/7` errors versus `7/7` for the latent target and preserved
`38/38` base-correct rows. The corrected development gate passed.

## Scientific boundary

This establishes a complete exposed-development candidate:

`hidden operand states → decoded integers → deterministic addition →`
`iterative leading residual write → native suffix writes`.

The leading mechanism is explicitly target-conditioned and output-side. The
result does not show that the model naturally performs deterministic
arithmetic internally or that a stable semantic neuron has been found.

No new generalization claim exists yet. The next valid experiment is a
single, hash-locked run on a newly generated corpus whose operand pairs,
templates, and split identities are disjoint from all prior fit, selection,
development, and audit examples.
