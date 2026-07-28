from __future__ import annotations

import hashlib
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


def choose_cyclic_donors(
    examples: list[Any],
    *,
    offsets: tuple[int, ...],
) -> list[list[int]]:
    """Choose matched donors at fixed cyclic leading-digit offsets."""

    if not offsets or any(offset < 1 or offset > 8 for offset in offsets):
        raise ValueError("cyclic donor offsets must be between 1 and 8")
    if len(set(offsets)) != len(offsets):
        raise ValueError("cyclic donor offsets must be unique")
    selections = []
    for recipient in examples:
        original_digit = int(str(recipient.result)[0])
        row = []
        for offset in offsets:
            desired_digit = (original_digit - 1 + offset) % 9 + 1
            candidates = [
                (index, donor)
                for index, donor in enumerate(examples)
                if int(str(donor.result)[0]) == desired_digit
            ]
            if not candidates:
                raise ValueError(f"no donor available for leading digit {desired_digit}")
            row.append(
                min(candidates, key=lambda item: donor_distance(recipient, item[1]))[0]
            )
        selections.append(row)
    return selections


def choose_prefix_donors(
    donor_pool: list[Any],
    recipients: list[Any],
    targets: list[int],
    *,
    prefix_length: int,
) -> list[int]:
    """Choose stable donor-pool indices matching each requested result prefix."""
    if len(recipients) != len(targets):
        raise ValueError("recipients and targets must align")
    if prefix_length < 1:
        raise ValueError("prefix length must be positive")
    selections = []
    for recipient, target in zip(recipients, targets, strict=True):
        prefix = str(target)[:prefix_length]
        candidates = sorted(
            (
                (donor.example_id, index)
                for index, donor in enumerate(donor_pool)
                if str(donor.result).startswith(prefix)
            ),
        )
        if not candidates:
            raise ValueError(f"no donor available for result prefix {prefix}")
        digest = hashlib.sha256(
            f"{recipient.example_id}:{prefix}".encode()
        ).digest()
        slot = int.from_bytes(digest[:8], "big") % len(candidates)
        selections.append(candidates[slot][1])
    return selections


def choose_position_donors(
    donor_pool: list[Any],
    recipients: list[Any],
    targets: list[int],
    *,
    position: int,
    wrong_digit: bool = False,
) -> list[int]:
    """Choose stable donors by a requested decimal digit at one position."""
    if len(recipients) != len(targets):
        raise ValueError("recipients and targets must align")
    if position not in (0, 1, 2):
        raise ValueError("position must be 0, 1, or 2")
    selections = []
    for recipient, target in zip(recipients, targets, strict=True):
        rendered_target = str(target)
        if len(rendered_target) != 3:
            raise ValueError("position donors require three-digit targets")
        desired = int(rendered_target[position])
        if wrong_digit:
            desired = desired % 9 + 1 if position == 0 else (desired + 1) % 10
        candidates = sorted(
            (
                (donor.example_id, index)
                for index, donor in enumerate(donor_pool)
                if int(str(donor.result)[position]) == desired
            ),
        )
        if not candidates:
            raise ValueError(
                f"no donor available for digit {desired} at position {position}"
            )
        digest = hashlib.sha256(
            f"{recipient.example_id}:{target}:{position}:{int(wrong_digit)}".encode()
        ).digest()
        slot = int.from_bytes(digest[:8], "big") % len(candidates)
        selections.append(candidates[slot][1])
    return selections
