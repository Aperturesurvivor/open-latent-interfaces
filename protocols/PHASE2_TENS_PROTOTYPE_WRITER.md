# Phase 2 Protocol: Donor-Free Tens Prototype Writer

## Question

Can fit-derived native class prototypes predict the 16 causal tens
coefficients without executing a donor at inference?

The rank study showed that a fit-only rank-16 basis retains 89/90 native tens
effects. This experiment replaces donor coefficients with deterministic class
means.

## Frozen representation

- Boundary: hidden-state index 27 / decoder block 26.
- Basis: the published fit-only native-delta basis, truncated to rank 16.
- Fit examples: 450.
- Fit prompt views: four frozen arithmetic templates.
- Base-model parameters: frozen.
- Learned neural-network parameters: none.

For every fit view, capture the native residual after the correct leading digit
and project it into the 16-dimensional basis.

Fit two prototype dictionaries:

1. one mean coordinate vector for each tens digit;
2. one mean coordinate vector for each leading-plus-tens prefix.

At inference, subtract the recipient's current 16 coordinates from the
requested class prototype and reconstruct the residual delta in the frozen
basis.

## Deterministic identity gate

Compute the unmodified next-token prediction. If it already equals the
requested digit, apply exactly zero delta. This makes "do nothing when already
correct" an explicit mechanism rather than a behavior the writer must learn.

Otherwise scale the prototype delta and cap each row at 100% of recipient
residual norm.

## Selection

Compare digit and prefix prototypes at scales:

`0.5, 0.75, 1.0, 1.25, 1.5`

Selection uses teacher-forced target and identity tens prompts. Choose by:

1. maximum of the smaller target/identity accuracy;
2. target accuracy;
3. identity accuracy;
4. target logit margin;
5. lower target norm;
6. simpler digit prototype on an exact tie.

## Development controls and gate

Evaluate once with:

- no intervention;
- selected prototype writer;
- cyclic wrong-tens prototype, norm matched;
- shuffled target prototype, norm matched;
- random direction in the rank-16 subspace, norm matched;
- hard-gated identity prompts.

The teacher-forced diagnostic gate requires:

- target tens accuracy at least 70%;
- at least 25 points over every matched control;
- identity tens accuracy at least 90%;
- mean relative norm at most 1.0;
- targeted digit-token rate 100%.

Passing this gate authorizes a closed-loop development experiment, not audit.
The audit remains sealed.
