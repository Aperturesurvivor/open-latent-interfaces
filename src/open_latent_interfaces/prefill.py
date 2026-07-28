from __future__ import annotations

from typing import Any


def render_prefilled_chat(
    tokenizer: Any,
    user_prompt: str,
    *,
    assistant_prefix: str,
) -> str:
    """Render a chat whose assistant response ends at a frozen text prefix."""
    if not assistant_prefix:
        raise ValueError("assistant prefix cannot be empty")
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prefix},
        ],
        tokenize=False,
        continue_final_message=True,
    )


def contextual_continuation_ids(
    tokenizer: Any,
    rendered_prefix: str,
    continuation: str,
) -> list[int]:
    """Return tokens added by a continuation, rejecting boundary retokenization."""
    prefix_ids = tokenizer(
        rendered_prefix,
        add_special_tokens=False,
    )["input_ids"]
    complete_ids = tokenizer(
        rendered_prefix + continuation,
        add_special_tokens=False,
    )["input_ids"]
    if complete_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("continuation retokenizes the frozen prompt boundary")
    return complete_ids[len(prefix_ids) :]


def verify_decimal_digit_contract(tokenizer: Any, rendered_prefix: str) -> dict[int, int]:
    """Verify that every decimal digit is one stable contextual token."""
    mapping = {}
    for digit in range(10):
        token_ids = contextual_continuation_ids(
            tokenizer,
            rendered_prefix,
            str(digit),
        )
        if len(token_ids) != 1:
            raise ValueError(
                f"digit {digit} requires {len(token_ids)} continuation tokens"
            )
        mapping[digit] = token_ids[0]
    if len(set(mapping.values())) != 10:
        raise ValueError("decimal digits do not map to ten distinct token IDs")
    return mapping


def result_digit_token_ids(
    tokenizer: Any,
    rendered_prefix: str,
    results: list[int],
) -> list[list[int]]:
    """Tokenize fixed-width decimal results under a verified prompt contract."""
    mapping = verify_decimal_digit_contract(tokenizer, rendered_prefix)
    widths = {len(str(result)) for result in results}
    if len(widths) != 1:
        raise ValueError("results must have one fixed decimal width")
    rows = [[mapping[int(digit)] for digit in str(result)] for result in results]
    for result, expected in zip(results, rows, strict=True):
        observed = contextual_continuation_ids(
            tokenizer,
            rendered_prefix,
            str(result),
        )
        if observed != expected:
            raise ValueError(
                f"result {result} does not compose from contextual digit tokens"
            )
    return rows
