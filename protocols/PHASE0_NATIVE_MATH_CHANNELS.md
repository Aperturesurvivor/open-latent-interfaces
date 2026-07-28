# Phase 0 Pilot Protocol: Native Mathematical Channels

Status: **engineering pilot; not preregistered confirmation**

## Question

Can a completely frozen open-weight language model expose a residual-stream
state that:

1. distinguishes requested addition from matched mathematical and numerical
   contrasts;
2. linearly decodes the scalar sum on exact-pair-disjoint examples; and
3. responds causally when moved toward the externally computed exact result
   along the probe-defined minimum-norm direction?

This is a write-path diagnostic. It is not an end-to-end deterministic implant.

## Frozen target

- Default model: `Qwen/Qwen2.5-0.5B-Instruct`
- Every model parameter is frozen.
- Hugging Face residual hidden-state indices are used: index 0 is the embedding
  output and index `k` is the output of decoder block `k - 1`.

## Dataset

The committed generator creates deterministic, split-disjoint examples:

- positives: direct addition, verbal addition, and addition word problems;
- negatives: matched multiplication, quoted addition strings, and factual
  prompts containing the same numbers;
- operands in every split: 20–499;
- exact ordered operand pairs are unique across splits;
- numeric values and result ranges may overlap, avoiding an accidental
  out-of-range extrapolation requirement;
- all three template families are balanced within every split.

Every pair appears once in each of the four semantic conditions. Future phases
must add a separately labeled template-held-out and numeric-extrapolation audit.

## Observational tests

At five evenly spaced residual boundaries:

- fit a ridge route probe on train examples;
- fit a ridge scalar-sum probe on train positives;
- fit a ten-class leading-result-digit ridge probe;
- choose a pilot layer by development leading-digit accuracy, breaking ties by
  scalar-sum R²;
- report development and untouched test route accuracy, balanced accuracy, and
  AUC;
- report sum R², MAE, and rounded exact recovery.

Probe success establishes decodability, not causal use.

## Causal pilot

On test positives whose first answer digit is a verified single tokenizer token:

1. compute the exact sum outside the model (oracle result);
2. use the fitted categorical probe to calculate the minimum-L2 activation
   shift that would move the exact leading digit ahead of its strongest
   competing digit by a fixed probe margin;
3. cap the intervention norm relative to the original activation norm;
4. add the shift only at the last prompt token and selected residual boundary;
5. compare first-correct-digit rank, top-1 accuracy, and logit margin against:
   - the unmodified frozen model;
   - a norm-comparable shift toward shuffled result labels.

The pilot may sweep strength on the test set for engineering information, but
the chosen value cannot support confirmation. A later protocol must freeze the
layer, strength, cap, templates, and a new audit before a causal claim.

## Interpretation gates

- A route probe does not establish a native router.
- A sum probe does not establish that the model uses that direction.
- A targeted advantage over shuffled-result shifts is causal evidence for the
  write direction, but not proof that the full subspace is a natural API.
- External oracle results bypass operand extraction and routing.
- Successful first-digit steering is not yet multi-token exact arithmetic.
- NLA explanations are not used in Phase 0.

## Required artifacts

- code commit;
- environment and resolved model revision;
- seed and full configuration;
- dataset generator hash;
- per-layer metrics;
- raw aggregate causal metrics, including failed and null outcomes.
