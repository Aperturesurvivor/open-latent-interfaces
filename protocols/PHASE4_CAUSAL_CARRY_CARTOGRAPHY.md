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
