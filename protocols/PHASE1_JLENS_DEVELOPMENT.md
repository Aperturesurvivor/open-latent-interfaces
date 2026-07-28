# Phase 1A Protocol: Development Jacobian Lens

## Purpose

Fit a real, generic-corpus Jacobian lens for frozen
`Qwen/Qwen2.5-0.5B-Instruct`, then use it as one member of a triangulated
mathematical-cartography workflow. This is a development artifact, not a frozen
audit and not evidence that the lens readout is causally used.

## Immutable inputs

- target model and exact Hugging Face revision;
- `Salesforce/wikitext` and exact dataset revision;
- deterministic row-selection seed, token-length filter, and row hashes;
- Jacobian-lens source repository, exact commit, and license;
- source layers 0–22, target layer 23;
- maximum sequence length 64, first 16 positions excluded;
- output-dimension batch size 16.

The selected corpus configuration must be committed before fitting begins.
Only row indices, token counts, and text hashes are published; the source text
remains governed by the dataset's own license.

## Separation from the audit

This 24-prompt lens is exclusively for engineering and Phase 1 development.
A later audit must use:

- a disjoint corpus selection and independently fitted lens;
- prompt-family and exact-string-disjoint math tasks;
- a configuration committed before audit labels are examined.

No threshold or layer chosen with the development lens may be reselected on the
audit.

## Required checks

1. Upstream J-lens tests pass at the pinned revision.
2. Every selected corpus row matches its recorded SHA-256.
3. The model and dataset load by immutable revision.
4. The fit checkpoint can resume without changing source/target layers.
5. The final lens file and selection config receive content hashes.
6. Development readouts are compared with vanilla logit lens, probes, donor
   patches, and causal interventions at the same activation sites.

## Interpretation boundary

A Jacobian-lens token is a readout of what an activation is disposed to make
the model say under an average linear transport. It is not a hidden-thought
transcript, a causal mechanism, or an exact typed value. Agreement with another
readout remains correlational until necessity and sufficiency interventions
pass.
