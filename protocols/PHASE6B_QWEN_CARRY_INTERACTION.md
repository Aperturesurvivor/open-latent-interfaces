# Phase 6B: Qwen Carry Interaction Coordinate

## Purpose

Test the newly isolated carry interaction after Phase 6 showed that full
increment transports are dominated by a generic direction. The target is a
matched difference in differences:

`(carry_increment - carry_base) - (control_increment - control_base)`.

This removes the token-local effect of incrementing the first operand when no
ones carry occurs and retains only the additional native state change
associated with crossing the carry boundary.

## Data boundary

- Fit uses only the 164 behavior-exact Phase 6 fit quartets.
- Phase 6 selection was used to formulate this distinct hypothesis but is not
  used to fit, choose ranks, choose regularization, set strength, or evaluate
  the writer.
- The untouched Phase 6 development split is a single confirmation run.
- Audit remains sealed unless this fixed one-shot confirmation passes.

## Frozen writer

- Site: Qwen hidden-state index 16, second operand ones token.
- Input: carry-base recipient state plus source ones-digit class.
- Target: the matched carry interaction defined above.
- State ranks: 8, 16, 32.
- Transport ranks: 4, 8, 16, 32.
- Ridge penalties: 1, 10, 100.
- Architecture choice: deterministic five-fold grouped fit-only
  cross-validation minimizing normalized interaction-delta MSE.
- Intervention strength: exactly 1.0; no development scale search.

A no-carry conditional writer of the same selected architecture is fitted only
to provide a control. It is not added to the carry-interaction target.

## Controls

- conditional no-carry delta, norm-matched;
- source-digit-rotated interaction prediction, norm-matched;
- shuffled-recipient interaction prediction, norm-matched;
- deterministic random direction, norm-matched.

## One-shot confirmation gate

- parse rate: 1.0;
- target tens accuracy: at least 0.50;
- target advantage over the strongest control: at least 0.25.

Failure closes this interaction-only writer on the corpus. Passing permits,
but does not itself authorize, a separately hashed one-shot audit.

## Claim boundary

A pass would show that a donor-free approximation of the matched carry
interaction is causally sufficient on an untouched prompt family. It would not
establish a single neuron, a thought transcript, universal transfer, or an
audited coordinate.

## Recorded outcome

The one-shot development confirmation failed. The target corrected 8/45 tens
digits, tied with the matched no-carry control and below the rotated-class
control at 10/45. Audit remains sealed and the interaction-only additive
writer is closed on this corpus.
