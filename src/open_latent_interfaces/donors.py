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


def choose_multi_donors(examples: list[Any]) -> list[list[int]]:
    """Choose one matched donor for every alternative leading-result digit."""

    selections = []
    for recipient in examples:
        original_digit = int(str(recipient.result)[0])
        donor_indices = []
        for desired_digit in range(1, 10):
            if desired_digit == original_digit:
                continue
            candidates = [
                (index, donor)
                for index, donor in enumerate(examples)
                if int(str(donor.result)[0]) == desired_digit
            ]
            if not candidates:
                raise ValueError(f"no donor available for leading digit {desired_digit}")
            donor_indices.append(
                min(candidates, key=lambda item: donor_distance(recipient, item[1]))[0]
            )
        selections.append(donor_indices)
    return selections
