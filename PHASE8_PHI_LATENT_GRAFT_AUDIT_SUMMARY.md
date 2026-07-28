# Phase 8 Phi Latent Graft Audit

## Outcome

The repository's first complete
`latent read → deterministic addition → native write` pipeline reached 42/45
exact answers but failed two frozen paired-uplift thresholds. It is an audit
nonpass.

## One-shot audit

- Operand reader: 45/45 exact pairs
- Deterministic computation: 45/45 exact sums
- Latent-read compute/write: 42/45 exact
- Oracle-compute native write: 42/45 exact
- Untouched base: 39/45 exact
- Random native-subspace control: 39/45 exact
- Shuffled-read compute/write: 4/45 exact
- Parse rate: 1.0
- Digit-token rate: 1.0

On the six examples the base model missed, the graft recovered three and
random recovered none. The graft harmed none of the 39 base-correct examples.
It exactly matched the oracle writer, proving that reader and deterministic
computation introduced no additional failures.

## Gate

Passed:

- reader exactness;
- deterministic computation;
- final exact accuracy;
- oracle gap;
- base-correct preservation;
- excess recovery over random;
- shuffled control;
- parse and digit-token contracts.

Failed:

- base-error recovery: 50%, required 75%;
- net exact improvement: 6.67 points, required 10 points.

The audit output SHA-256 is
`1f01247935eb7cd080dc548af55d3a34e892dd435b92f9d75bdd68255bda3d98`.
The existing output path refused a repeat invocation.

## Interpretation

The token-local reader is independently validated on the audit split, and the
full pipeline is functional. The limiting component is the previously audited
native answer writer under this wider prompt/value distribution. A new
integrated audit requires a new writer development boundary and new untouched
examples; this audit cannot be reused.
