# Capability Gate v1 Results

## Outcome

No regime passed. Audit remains sealed.

The 360-condition development sweep crossed five addition regimes, three
prompt families, and raw/chat presentation. Overall exact first-integer
accuracy was 33.1%.

| Regime | Aggregate exact accuracy |
|---|---:|
| Single-digit, no carry | 45.8% |
| Single-digit, with carry | 47.2% |
| Two-digit, no carry | 31.9% |
| Two-digit, with carry | 19.4% |
| Three-digit, mixed | 20.8% |

No regime approached the precommitted 90% aggregate and 80% worst-cell gates.

## Structured result

The aggregate conceals a strong interface effect:

- direct chat: 100% on both single-digit regimes, 91.7% on two-digit no-carry;
- symbolic chat: 0% in every regime;
- raw word problems: 0% in every regime, usually because no integer appeared
  within eight generated tokens;
- word-problem chat ranged from 91.7% on single-digit carry to 0% on
  two-digit carry and three-digit mixed.

Inspection showed that symbolic chat commonly produced a correct sentence such
as `3 + 1 equals 4`, but the frozen first-integer scorer correctly read `3`.
This is a protocol/input-contract defect: the symbolic v1 template did not ask
for answer-only output, unlike the other templates.

The result is not rescored. V1 is preserved as a non-pass.

## Revision decision

V2 makes every template explicitly answer-only, treats the checkpoint's native
chat format as the primary competence interface, and retains raw presentation
as a diagnostic rather than a selection gate. It moves to
Qwen2.5-1.5B-Instruct because the 0.5B model's word-problem arithmetic is too
weak for stable value-channel mapping.

This change is frozen before v2 development. The v1 audit remains unopened and
is retired; it will not be used to select a v2 regime.
