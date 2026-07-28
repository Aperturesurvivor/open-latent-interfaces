# Phase 3 Protocol: Phi Cross-Model Replication

## Question

Can the donor-free native-coordinate write interface discovered in
Qwen2.5-1.5B be independently rediscovered in a different frozen model family
without assuming Qwen's layer, basis, rank, or coordinate prototypes?

## Frozen target

- model: `microsoft/Phi-3.5-mini-instruct`
- revision: `2fe192450127e6a83f7441aef6e3ca586c338b77`
- architecture: 32-layer Phi-3 decoder, width 3072
- model parameters: frozen
- execution: deterministic float16 inference on MPS

The upstream model card records an MIT license. The model passed the separate
behavioral capability screen before this protocol was written.

## Tokenizer contract

Phi's SentencePiece tokenizer inserts an otherwise empty word-start token when
an answer begins directly with a digit. Phase 3 therefore uses an explicit
assistant prefill, `Answer=`, and asks the model to continue with the integer.
In that context every decimal digit is exactly one token and a three-digit
answer is exactly three continuation tokens.

The prefill is part of the frozen interface contract. It must be used for base,
targeted, identity, and control conditions alike. A prefilled-answer behavioral
check must pass before causal mapping.

## Dataset

Generate 720 deterministic three-digit-addition examples:

- 450 fit;
- 90 selection;
- 90 development;
- 90 sealed audit.

Every split is balanced across the nine leading result digits and uses a
different prompt family. Canonicalized operand pairs are disjoint across
splits and exclude all capability-gate, Phase 1, and Phase 2 pairs. Reversed
operands cannot cross a boundary.

The audit split may be generated and hashed but not evaluated until a complete
controller, all thresholds, and a one-shot audit runner are committed.

## Rediscovery ladder

1. Verify exact base-model continuation under the frozen `Answer=` prefill on
   fit, selection, and development only.
2. Map full-native donor control independently at each answer position across
   a precommitted late-layer boundary grid.
3. Select boundaries on the selection split only.
4. Estimate the intrinsic rank of effective native transports using fit-only
   states and selection-only rank choice.
5. Fit digit-coordinate prototypes from fit only.
6. Select scales, hard gates, and any position-specific parameters on
   selection only.
7. Run one untouched development evaluation with matched wrong-digit,
   shuffled, random, and identity controls.
8. Freeze the complete package before opening the audit exactly once.

Qwen's hidden index 27, rank 16, and prototype vectors are hypotheses to test,
not defaults that may be silently inherited.

## Advancement gates

The complete development and audit controller must satisfy all of:

- exact target result at least 50%;
- every answer position at least 70%;
- exact advantage over every matched control at least 25 points;
- identity preservation at least 90%;
- mean intervention norm no greater than one recipient residual norm at every
  position;
- parse rate 100%.

## Claim boundary

A pass would establish cross-family reproducibility of the workflow and a
second model-specific native interface. It would not establish that coordinate
vectors transfer directly between models, that the interface captures the
model's internal arithmetic algorithm, or that it generalizes beyond this
three-digit answer channel.
