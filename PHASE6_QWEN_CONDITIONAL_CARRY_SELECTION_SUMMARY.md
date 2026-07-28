# Phase 6 Qwen Conditional Carry Selection

## Outcome

The fresh behavior gate passed, but the recipient-conditioned carry writer
failed its frozen specificity gate and is closed.

- Fresh corpus: 315 quartets / 1,260 rows
- Historical overlap: zero canonical operand pairs with Phase 4/5
- Fully correct eligible fit quartets: 164/180
- Selected fit-only architecture: state rank 32, transport rank 32, ridge 100
- Cross-validated normalized delta MSE: 0.2283
- Cross-validated mean cosine similarity: 0.8845

At scale 1.5:

- target: 32/45 target tens, 30/45 exact
- matched no-carry: 19/45 target tens, 16/45 exact
- rotated source class: 33/45 target tens, 31/45 exact
- shuffled recipient: 32/45 target tens, 30/45 exact
- random: 2/45 target tens, 1/45 exact

The target was effective but not conditional: wrong-class and shuffled-state
predictions transferred equally well. This is evidence that the regression
retained a dominant population-level increment direction rather than
recovering a recipient-specific carry coordinate.

The result hash is
`c1183b605a37a55c0d9472f53bff5a842003ba882bdab39610e4d7a7919ecd05`.
The emitted fit artifact hash is
`cfa1c92cc13b11708a634a41875d94c2b96182dc4eaa9f1a552093c32bbff416`.
Neither development nor audit was opened.
