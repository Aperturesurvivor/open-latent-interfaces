# Phase 2 Protocol: Cross-Position Ones Prototype Transfer

## Question

Is the published rank-16 late-layer digit interface reusable at the ones
position without fitting a new donor-delta basis?

The same interface produced 90/90 tens control and composed perfectly in
closed-loop generation. A successful cross-position transfer would show that
the subspace represents next-digit identity rather than a tens-specific
arithmetic feature.

## Frozen representation

- Boundary: hidden-state index 27 / decoder block 26.
- Basis: the published rank-16 tens native-delta basis.
- Fit examples: 450.
- Fit views: the same four arithmetic templates.
- Base-model parameters: frozen.
- Neural-network training: none.

For each fit view, capture the native residual after both correct prefix digits
and average its rank-16 coordinates by the correct ones digit.

At inference, teacher-force the requested leading-plus-tens prefix and replace
the recipient's 16 coordinates with the requested ones-digit prototype.

## Hard gate and selection

If the unmodified model already predicts the requested ones digit, apply zero
delta. Otherwise apply the prototype delta with a per-row 100% residual-norm
cap.

Select one scale from:

`0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0`

using the frozen target/identity lexicographic rule from the tens experiment.
Selection uses only the selection split.

## Development controls and gate

- no intervention;
- ones prototype writer;
- cyclic wrong-ones prototype, norm matched;
- shuffled target prototype, norm matched;
- random direction in the same rank-16 subspace, norm matched;
- hard-gated identity.

The diagnostic gate requires:

- target ones accuracy at least 70%;
- at least 25 points over every matched control;
- identity ones accuracy at least 90%;
- mean relative norm at most 1.0;
- target digit-token rate 100%.

If every criterion passes, substitute this artifact into the frozen closed-loop
hybrid without further selection. If transfer fails, return to an ones-specific
native boundary and rank map.

The audit remains sealed.
