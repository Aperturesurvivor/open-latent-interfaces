# Phase 6B Qwen Carry Interaction Development

## Outcome

The fixed difference-in-differences carry interaction failed its one-shot
development gate. Audit remained sealed.

- Fit target: `(carry increment - carry base) - (control increment - control base)`
- Selected fit-only architecture: state rank 8, transport rank 32, ridge 10
- Cross-validated normalized interaction MSE: 0.3427
- Cross-validated mean cosine similarity: 0.7793
- Intervention scale: 1.0, fixed before development

On 45 untouched development quartets:

- target: 8/45 target tens, 6/45 exact
- matched no-carry: 8/45 target tens, 5/45 exact
- rotated source class: 10/45 target tens, 8/45 exact
- shuffled recipient: 6/45 target tens, 4/45 exact
- random: 2/45 target tens, 0/45 exact

The isolated interaction was neither sufficiently effective nor class
specific. Together with Phase 6, this indicates that the effective full native
transport cannot be decomposed into a generic additive increment plus a
separately additive carry residual at this boundary using these linear
bridges.

The result hash is
`0c9efa861e31eae5b8df887d7e6557a5774364d07c5433f702a252d2856aa479`.
The fit artifact hash is
`721a9f5959e59a7034c6f08b02331b306060d707d450030d33f76cdb11b3ab53`.
