# Phase 7 Qwen Target-State Selection

## Outcome

The target-state overwrite failed selection specificity. Development and audit
remained sealed.

The fresh corpus contains 315 balanced quartets and excludes every canonical
operand pair from Phase 4/5 and Phase 6. Untouched selection and development
behavior each passed at 167/180 exact rows and 36/45 complete quartets. The
wider fit pool supplied 129 behavior-exact complete quartets.

Fit-only between-centroid geometry selected coordinate rank 8 at the frozen
95% explained-energy rule. On 45 selection quartets:

- scale 0.5: target 5/45 tens; strongest control 6/45
- scale 1.0: target 13/45; strongest control 13/45
- scale 1.5: target 15/45; strongest control 16/45
- scale 2.0: target 15/45; strongest control 15/45
- random control: 1/45 at every scale

Structured subspace movement was causal, but the requested digit label was not
specific: target, identity, and wrong-digit overwrites behaved interchangeably.
Hidden-state index 16 is therefore not supported as a typed result-tens
register by this experiment.

The result SHA-256 is
`ddd5a8ab47bfedc97afd0104da2781c978d2c8022c85b39562309799ce29b967`.
The fit artifact SHA-256 is
`8a83f4d4b661c051cabd348b7c28877196f9b809c66814b975368947418d7787`.
