# Phase 2 Protocol: Scaled Native Transport Adapter

## Objective

Train a compact, donor-free nonlinear bridge at the causal native write
boundary established in Phase 1. The bridge accepts a frozen recipient state
and a deterministic target digit, then emits a bounded residual transport
without changing base-model weights.

Phase 2 exists because full native donor replacement passed 38/45 development
examples while every compressed Phase 1 bridge was fitted from only 60
examples. Increasing model flexibility on that small corpus did not close the
gap.

## Frozen model and boundary

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- HF hidden-state index: 23
- Decoder block: 22
- Base-model parameters: frozen

## Dataset

The deterministic Phase 2 corpus contains:

| Split | Examples | Pairs per leading digit | Use |
|---|---:|---:|---|
| Fit | 450 | 50 | activation capture and adapter fitting |
| Selection | 90 | 10 | architecture, checkpoint, and scale selection |
| Development | 90 | 10 | one post-freeze development evaluation |
| Audit | 90 | 10 | sealed until every gate below is frozen |

Every split has a distinct prompt-template family. Canonical unordered operand
pairs are disjoint across splits and exclude the capability gate and all Phase
1 examples.

## Training pairs

Each fit recipient receives four deterministic target donors spanning
alternative leading-result classes, plus an identity example:

- targeted pair: target-prefix recipient state → matched native donor state;
- identity pair: native recipient state → zero transport.

Pairs are generated independently at answer positions zero, one, and two.
Donor identities and native states are training data only; inference accepts no
donor execution.

## Adapter

For each answer position:

1. project recipient states into a frozen PCA state basis;
2. concatenate a one-hot target digit;
3. predict reduced-rank transport coefficients with a two-layer GELU MLP;
4. reconstruct the residual delta in a frozen transport PCA basis;
5. apply a frozen norm cap relative to the recipient residual.

Training minimizes:

- transport coefficient mean-squared error;
- next-state reconstruction error in the transport basis;
- identity transport error;
- excess relative-norm penalty.

The base language model is never updated.

## Selection

Fit data may select checkpoints by a fixed combined objective. The selection
split chooses one adapter width, bottleneck rank, and output scale per answer
position from a preregistered grid.

Development is opened once after those choices are immutable.

## Development controls

- untouched base;
- trained targeted adapter;
- shuffled target digit, norm-matched;
- shuffled recipient state, norm-matched;
- random direction, norm-matched;
- identity/same-digit preservation;
- Phase 1G linear conditional bridge;
- full native donor upper bound.

## Development advancement gate

The adapter may advance toward audit only if:

- closed-loop exact target result is at least 50%;
- every answer position transfers at least 70% of target digits;
- targeted exactness exceeds every matched control by at least 25 points;
- identity preservation is at least 90%;
- median relative intervention norm is at most 1.0 at every position;
- all outputs remain parseable;
- no hyperparameter or threshold was selected using development.

Failure leaves the audit sealed.

## Audit gate

Before audit, commit:

- dataset and rendered-prompt hashes;
- model revision;
- activation hashes;
- adapter weights and hashes;
- architecture and checkpoint;
- all scales and norm caps;
- exact metric code;
- all pass thresholds.

The audit is a single run. It cannot select or repair the adapter.
