# Phase 4 Protocol: Causal Carry Cartography

## Question

Where does a frozen arithmetic-capable model form and causally use an
ones-to-tens carry, distinct from merely representing that an input operand
increased by one?

## Matched quartet

Each quartet fixes the higher digits of both operands and one operand's ones
digit. It contains:

1. a no-carry base whose ones digits sum to 9;
2. the same problem after incrementing operand A by one, making the ones digits
   sum to 10 and creating a carry;
3. a control base whose ones digits sum to 5;
4. the same +1 increment of operand A, leaving the ones column carry-free.

Both experimental and control transitions change the same operand by +1 and
change the result by +1. Only the experimental transition changes carry state.
Their activation difference-in-differences therefore targets carry-specific
computation more narrowly than an ordinary clean/corrupt pair.

## Frozen dataset

- 180 fit quartets;
- 45 selection quartets;
- 45 development quartets;
- 45 sealed audit quartets;
- four examples per quartet;
- balanced leading result digits within each split;
- unique canonical operand pairs;
- exclusion of every capability, Phase 1, Phase 2, and Phase 3 operand pair;
- one prompt family per split;
- frozen Phi `Answer=` assistant prefill.

## Research ladder

1. Verify untouched behavior on every non-audit row and require both high row
   accuracy and high complete-quartet accuracy.
2. Capture prompt-boundary states for all four variants across layers.
3. Measure carry and control increment deltas separately.
4. Compute the difference-in-differences carry residual:
   `(carry_increment - carry_base) -
   (control_increment - control_base)`.
5. Test donor-dependent full deltas, difference-in-differences deltas, matched
   control deltas, shuffled deltas, and random norm-matched deltas causally.
6. Select layers on selection only.
7. Estimate low-rank carry subspaces from fit only.
8. Validate a compact intervention on untouched development before authorizing
   any audit.

## Fit-eligibility amendment after the behavior non-pass

The original behavior gate was executed once and failed on fit while selection
and development passed. The dataset and result are preserved. Subsequent
fit-only representation estimation may use only quartets whose four untouched
answers were all correct, provided at least 100 eligible fit quartets remain.

Selection and development may not be filtered. Every causal report must retain
the original behavior non-pass and report complete unfiltered selection
metrics. This amendment does not authorize audit access.

## Prompt-boundary amendment after the single-token non-pass

The first selection-only causal map patched only the final `Answer=` prompt
token. Full carry-pair, difference-in-differences, matched increment,
shuffled, and random interventions all reached at most 2/45 target tens
digits. That result rejects a single-vector prompt bottleneck but leaves a
distributed prompt-state mechanism unresolved.

The next frozen selection experiment replaces the residual states of every
active prompt token once, before cached autoregressive generation. It does not
patch generated answer tokens. Corresponding quartet prompts must have equal
token lengths, and that length matrix is hash-locked. The conditions are:

1. zero intervention;
2. the complete carry-pair sequence delta;
3. the carry difference-in-differences sequence delta, Frobenius-norm matched
   per example to the complete carry delta;
4. the matched no-carry +1 sequence delta at the same norm;
5. an isotropic random sequence delta at the same norm.

The full sequence condition remains an upper-bound sufficiency test. A
carry-specific claim still requires the difference-in-differences condition
to pass its absolute threshold and outperform both matched controls. Generated
tokens remain unpatched so the experiment cannot teacher-force the answer.

The resulting full-sequence selection map found a strong generic `+1` route
but no carry-specific advantage. A bounded follow-up therefore tests three
causal regions at indices 1, 5, 9, and 13:

1. the single changed first-operand digit token;
2. the single second-operand digit that distinguishes the carry and control
   contexts;
3. every downstream token after that carry-context digit.

Each region is evaluated separately. The generic changed-token gate uses an
isotropic region-matched control. Carry-context and downstream-tail gates must
outperform both their matched no-carry `+1` regional delta and an isotropic
region-matched control. Token identities, changed positions, context
positions, and sequence lengths are hash-locked before selection is opened.

The regional selection result fixed hidden-state index 1 for the generic
changed-operand interface and index 13 for the carry-context interface. The
former reached 39/45 exact targets against 0/45 random; the latter reached
32/45 target tens digits against 17/45 for the matched no-carry regional
update and 1/45 random. Subsequent compact estimation must use only the 127
behavior-correct fit quartets and these preselected token/boundary pairs.

Before rank compression, a frozen donor-free viability gate fits one
full-width mean transport vector per source-digit class from the 127 eligible
fit quartets. The operand writer uses the changed digit at index 1. The carry
writer uses the contextualized second-operand digit at index 13. Source digits
1–4 have 24–41 eligible examples each.

Selection tests scales 0.5, 1.0, and 1.5. The operand prototype must outperform
wrong-class and isotropic controls. The carry prototype must outperform the
matched no-carry class prototype, a wrong-class carry prototype, and an
isotropic control. Rank selection is authorized only if the corresponding
full-width donor-free writer passes.

The operand prototype passed, while the carry prototype failed only the
wrong-class advantage: wrong source-digit carry prototypes generalized as
well as matched ones. This result is preserved as a class-specific non-pass.
A bounded follow-up may test the simpler class-invariant mean carry vector
against universal matched no-carry and isotropic controls before any carry
rank selection.

## Evidence rules

- A carry probe is correlational evidence only.
- Full-state carry-pair transport is a causal upper bound but is not
  carry-specific.
- A difference-in-differences intervention must outperform the matched +1
  control before it can support a carry-specific claim.
- Output recovery alone does not establish a thought transcript.
- Audit data remains sealed until a complete intervention and conjunctive gate
  are committed.

## Long-run bridge

If a compact causal carry coordinate can be read and written, connect it to a
frozen deterministic column-adder and the already audited native-coordinate
answer writer. This would test an end-to-end latent read → deterministic
mechanism → latent write graft without changing model weights.
