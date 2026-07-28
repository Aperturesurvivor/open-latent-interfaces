# Capability Gate v2 Frozen Audit

## Outcome

**Pass.**

The preselected mixed three-digit addition regime passed every frozen
development threshold on exact-pair-disjoint audit operands.

| Condition | Exact |
|---|---:|
| Primary native-chat aggregate | 24/24 (100%) |
| Diagnostic raw aggregate | 24/24 (100%) |
| Direct chat | 8/8 |
| Symbolic chat | 8/8 |
| Word-problem chat | 8/8 |
| Direct raw | 8/8 |
| Symbolic raw | 8/8 |
| Word-problem raw | 8/8 |

Every continuation contained a parseable integer and every parsed integer was
the exact answer.

## What this licenses

Qwen2.5-1.5B-Instruct at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` has a demonstrated behavioral
competence envelope for the selected addition regime. Phase 1 may now map
route, operands, operation, timing, and result on fresh prompts in this
envelope.

## What this does not license

The audit does not establish:

- that the model uses a human-like arithmetic algorithm;
- that a linear or verbal readout reflects causal internal state;
- that the correct result exists at every layer or prompt position;
- that an intervention can write an exact multi-token result;
- that any deterministic graft preserves unrelated behavior.

Those remain independent Phase 1–4 requirements.

The nonpassing Qwen2.5-0.5B three-digit regime is retained as a useful negative
control: a method that reports equally specific "answers" in both regimes is
likely reading task priors or output format rather than successful computation.
