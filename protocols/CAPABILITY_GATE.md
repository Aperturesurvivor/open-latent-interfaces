# Arithmetic Capability Gate

## Purpose

Identify a nontrivial addition regime that the untouched target model solves
reliably before claiming to map its correct mathematical reasoning. This gate
was added after Phase 1A showed strong generic numeric readout but chance-level
value identity on a regime the base model did not solve.

## Frozen factors

- five regimes: single-digit no-carry, single-digit carry, two-digit no-carry,
  two-digit carry, and mixed three-digit;
- three prompt families: direct, symbolic, and word problem;
- raw and native chat-template presentation;
- 12 development and 8 audit canonical operand pairs per regime;
- reversed operands cannot cross splits;
- greedy generation with a fixed maximum of eight new tokens;
- exact first-parsed-integer scoring.

Every pair appears in all six template/presentation conditions. Development and
audit pairs are exact- and commutative-disjoint.

## Selection rule

Development may select the easiest regime satisfying both:

1. at least 90% aggregate exact accuracy;
2. at least 80% exact accuracy in every template/presentation cell.

Prefer the hardest regime that passes. No individual example may be removed.
The selected regime and rule must be committed before authorizing the audit.

The audit passes at the same aggregate and worst-cell thresholds. Failure
returns to development; the audit may not be used to pick another regime.

## Claim boundary

Passing establishes a behavioral competence envelope only. It does not show
where a value is represented or that any readout is causal. Failing regimes
remain useful negative controls for distinguishing generic task context from
correct value-specific computation.
