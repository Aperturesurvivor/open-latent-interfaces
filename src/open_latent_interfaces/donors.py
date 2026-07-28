from __future__ import annotations

from typing import Any


def donor_distance(recipient: Any, donor: Any) -> tuple[int, int, int]:
    return (
        abs((recipient.result % 100) - (donor.result % 100)),
        int(recipient.ones_carry != donor.ones_carry)
        + int(recipient.tens_carry != donor.tens_carry),
        abs(recipient.operand_a - donor.operand_a)
        + abs(recipient.operand_b - donor.operand_b),
    )


def choose_donors(examples: list[Any]) -> tuple[list[int], list[int]]:
    """Choose deterministic different-leading and same-leading native donors."""

    targeted = []
    same_leading = []
    for recipient_index, recipient in enumerate(examples):
        original_digit = int(str(recipient.result)[0])
        desired_digit = original_digit % 9 + 1
        target_candidates = [
            (index, donor)
            for index, donor in enumerate(examples)
            if int(str(donor.result)[0]) == desired_digit
        ]
        same_candidates = [
            (index, donor)
            for index, donor in enumerate(examples)
            if index != recipient_index and int(str(donor.result)[0]) == original_digit
        ]
        targeted.append(
            min(target_candidates, key=lambda item: donor_distance(recipient, item[1]))[
                0
            ]
        )
        same_leading.append(
            min(same_candidates, key=lambda item: donor_distance(recipient, item[1]))[
                0
            ]
        )
    return targeted, same_leading
