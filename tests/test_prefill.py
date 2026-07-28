from typing import Any

import pytest

from open_latent_interfaces.prefill import (
    contextual_continuation_ids,
    render_prefilled_chat,
    result_digit_token_ids,
    verify_decimal_digit_contract,
)


class CharacterTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        continue_final_message: bool,
    ) -> str:
        assert not tokenize
        assert continue_final_message
        return f"U:{messages[0]['content']}|A:{messages[1]['content']}"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def test_prefilled_digit_contract_and_results() -> None:
    tokenizer = CharacterTokenizer()
    rendered = render_prefilled_chat(
        tokenizer,
        "2 + 3",
        assistant_prefix="Answer=",
    )
    assert rendered == "U:2 + 3|A:Answer="
    assert contextual_continuation_ids(tokenizer, rendered, "5") == [ord("5")]
    mapping = verify_decimal_digit_contract(tokenizer, rendered)
    assert mapping[0] == ord("0")
    assert result_digit_token_ids(tokenizer, rendered, [123, 590]) == [
        [ord("1"), ord("2"), ord("3")],
        [ord("5"), ord("9"), ord("0")],
    ]


def test_prefilled_chat_rejects_empty_prefix() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        render_prefilled_chat(CharacterTokenizer(), "2 + 3", assistant_prefix="")
