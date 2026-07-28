# Phase 8: Phi Latent Read–Compute–Write Graft

## Purpose

Build the first complete frozen-base-model pipeline in this repository:

`token-local latent operand read → deterministic integer addition → audited native answer write`.

The experiment deliberately starts with a narrow read contract. An external
semantic locator identifies the two operand token spans. Digit values are
decoded from Phi residual states, not read from token IDs or prompt text. The
deterministic mechanism receives only decoded integers.

## Components

### Read bridge

- candidate hidden-state indices: 1, 4, 8, 12, 16, 20, 24, 28, 32;
- one full-width native-state centroid per decimal digit;
- fixed nearest-centroid linear decision rule;
- fit on Phase 8 fit prompts only;
- selection chooses the earliest passing hidden-state index;
- no model weights or reader parameters change after selection.

### Deterministic mechanism

Exact host-language integer addition with no learned parameter:

`target = decoded_operand_a + decoded_operand_b`.

### Write bridge

The already audited
`manifests/phi35-mini-next-digit-interface-v1.json` controller:

- leading digit at hidden-state index 24;
- tens and ones digits at hidden-state index 30;
- frozen ranks, scales, prototypes, hard gate, and norm cap.

## Fresh-data boundary

Phase 8 reuses the pair-disjoint Phase 7 corpus as a model-independent operand
set but evaluates a different model and a different hypothesis. Fit,
selection, development, and audit remain mutually disjoint. Phase 7 audit
prompts have not been run on Phi.

## Reader selection gate

On every unfiltered selection prompt variant:

- digit-token accuracy: at least 0.995;
- operand-A exact accuracy: at least 0.98;
- operand-B exact accuracy: at least 0.98;
- operand-pair exact accuracy: at least 0.98;
- all ten digit classes represented in fit;
- rotated-label control pair accuracy: at most 0.05.

The earliest passing hidden-state index is selected. If none pass, the graft
does not advance.

## Development and audit conditions

- untouched base generation;
- oracle-compute native write, using ground-truth operands;
- latent-read deterministic-compute native write;
- shuffled-read compute/write control;
- random native-subspace control.

The integrated target must meet:

- reader operand-pair exact accuracy at least 0.98;
- deterministic target accuracy at least 0.98;
- final exact output accuracy at least 0.75;
- no more than a 0.05 exact-accuracy gap from oracle-compute native write;
- at least a 0.50 advantage over the strongest shuffled/random control;
- parse and digit-token rates of 1.0.

Development runs once. Audit requires a separately frozen configuration and
one authorized execution.

## Development metric correction

The first development run completed without execution error and produced:

- reader operand pairs: 45/45 exact;
- deterministic sums: 45/45 exact;
- latent graft: 44/45 exact;
- oracle writer: 44/45 exact;
- untouched base: 38/45 exact;
- random native-subspace intervention: 40/45 exact;
- shuffled-read control: 2/45 exact.

The original gate marked this a nonpass only because it required the latent
condition to exceed every control by 23/45 absolute correct answers. That
criterion is not identifiable when base accuracy is already 38/45: a random
intervention can preserve most correct base answers without implementing the
target computation.

Before any audit access, a no-rerun correction replaces only that absolute
control criterion with paired causal-uplift checks:

- recover at least 75% of base errors;
- preserve at least 98% of base-correct examples;
- improve net exact accuracy over base by at least 10 percentage points;
- recover at least 50% more of the base-error set than random;
- shuffled-read true accuracy at most 10%.

All reader, deterministic-compute, final-exact, oracle-gap, parse, and
digit-token checks remain unchanged. The original result and nonpass remain
immutable and hash-referenced. The corrected rule is development-derived and
must be frozen unchanged for the untouched audit.

## Audit authorization

The corrected development package passes all paired-uplift checks. The audit
configuration authorizes exactly one run on the 45 untouched audit
`carry_base` examples. It locks:

- dataset, example-list, and token-contract hashes;
- reader selection result, reader tensor, and hidden-state index;
- audited writer manifest and all writer-internal artifacts through manifest
  verification;
- deterministic operation, random seed, conditions, and corrected thresholds;
- original development config/result and correction config/result hashes;
- audit runner hash, output path, and maximum run count.

An existing audit output is never overwritten. Any failed audit conjunct is a
final nonpass for this package and cannot be used to tune or repeat it.

## Recorded audit outcome

The one authorized audit run completed and the repeat guard refused a second
run. The integrated package is an audit nonpass:

- reader operand pairs: 45/45 exact;
- deterministic sums: 45/45 exact;
- latent graft: 42/45 exact;
- oracle writer: 42/45 exact;
- untouched base: 39/45 exact;
- random native-subspace control: 39/45 exact;
- shuffled-read control: 4/45 exact;
- base-error recovery: 3/6;
- preservation of base-correct examples: 39/39.

Reader, compute, final-exact, oracle-gap, preservation, random advantage,
shuffled, parse, and digit-token checks passed. The frozen 75% base-error
recovery and 10-point net-improvement checks failed at 50% and 6.67 points.
The audit may not be retuned or repeated.

The operand-reader component independently passed its audit criterion at
45/45 exact operand pairs and may be packaged with the narrower claim. The
integrated graft may not be described as audit-passing.

## Claim boundary

A passing audit would establish a complete deterministic latent graft under an
external operand-token locator for one model revision and one three-digit
addition distribution. It would not establish autonomous operand discovery,
free-form prompt parsing, hidden chain-of-thought recovery, unrestricted
arithmetic, or cross-model vector portability.
