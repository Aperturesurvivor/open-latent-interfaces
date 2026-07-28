# Phase 2 Protocol: Tens Native-Boundary Map

## Question

At which late residual boundary can a native state most reliably overwrite the
tens digit under the balanced all-digits-changed target construction?

The scale sweep showed that the rank-64 compressed writer saturates at 31/90
selection tens digits and loses preservation as amplitude increases. This
experiment tests the native causal geometry before any larger adapter is
trained.

## Frozen data and donors

- Recipients: Phase 2 selection, followed by one development evaluation at the
  selected boundary.
- Donor pool: Phase 2 fit only.
- Target: the balanced synthetic result's tens digit.
- Recipient context: recipient prompt plus the synthetic target's leading
  digit.
- Targeted donor: deterministic fit example whose native result has the same
  leading-plus-tens prefix as the synthetic target.
- Wrong-tens donor: deterministic fit example with the same target leading
  digit and cyclic next tens digit.
- Donor prompts: re-rendered in the recipient split's template family.

All target and donor-assignment hashes are frozen in the configuration. The fit
pool covers all 90 legal leading-plus-tens prefixes.

## Selection boundary sweep

Sweep Hugging Face hidden-state indices:

`17, 19, 21, 23, 25, 27, 28`

These correspond to residual boundaries after decoder blocks 16, 18, 20, 22,
24, 26, and 27.

For each boundary, evaluate:

- no intervention;
- targeted native donor replacement;
- wrong-tens donor, norm matched;
- shuffled targeted donor, norm matched;
- random direction, norm matched.

Select lexicographically by:

1. targeted tens accuracy;
2. advantage over the strongest non-base matched control;
3. targeted mean logit margin;
4. lower mean relative intervention norm.

No development metric participates in boundary selection.

## Development

Evaluate the same five teacher-forced conditions once at the selected boundary.
This is a native upper-bound and localization experiment. It is not a compact
writer and cannot advance to audit by itself.

Evidence for a usable boundary requires:

- at least 70% targeted tens accuracy;
- at least 25 percentage points over every matched control;
- 100% parseable single-token digit outputs.

## Decision rule

- If native control passes, locate the smallest sufficient residual subspace at
  that boundary with direct residual optimization and rank truncation.
- If native control fails, expand the boundary sweep earlier before changing
  adapter capacity.

The audit remains sealed.
