from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Literal

from open_latent_interfaces.phase3_data import prior_canonical_pairs
from open_latent_interfaces.phase4_data import build_phase4_carry_quartets
from open_latent_interfaces.phase6_data import (
    REFERENCE_PARAMETERS,
    build_phase6_carry_quartets,
)

Phase7Split = Literal["fit", "selection", "development", "audit"]
CarryVariant = Literal[
    "carry_base",
    "carry_increment",
    "control_base",
    "control_increment",
]

PHASE7_TEMPLATES = {
    "fit": "Determine {a} plus {b}. Give only Answer=<integer>.",
    "selection": "Sum {a} and {b}, responding only as Answer=<integer>.",
    "development": "Compute the combined value of {a} and {b}: Answer=<integer>.",
    "audit": (
        "One register holds {a} items and a second holds {b}. "
        "Report their combined count only as Answer=<integer>."
    ),
}


@dataclass(frozen=True)
class Phase7CarryExample:
    example_id: str
    quartet_id: str
    split: Phase7Split
    variant: CarryVariant
    operand_a: int
    operand_b: int
    result: int
    ones_carry: int
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def historical_canonical_pairs() -> set[tuple[int, int]]:
    historical = build_phase4_carry_quartets(**REFERENCE_PARAMETERS)
    phase6 = build_phase6_carry_quartets()
    return prior_canonical_pairs() | {
        tuple(sorted((row.operand_a, row.operand_b)))
        for row in historical + phase6
    }


def _candidate_quartets() -> list[tuple[int, int, int]]:
    return [
        (a_high, b_high, a_ones)
        for a_high in range(2, 50)
        for b_high in range(a_high, 99)
        for a_ones in range(1, 5)
        if 10 <= a_high + b_high <= 98
    ]


def _quartet_rows(
    split: Phase7Split,
    quartet_index: int,
    a_high: int,
    b_high: int,
    a_ones: int,
) -> list[Phase7CarryExample]:
    a = 10 * a_high + a_ones
    incremented_a = a + 1
    carry_b = 10 * b_high + (9 - a_ones)
    control_b = 10 * b_high + (5 - a_ones)
    values: tuple[tuple[CarryVariant, int, int, int], ...] = (
        ("carry_base", a, carry_b, 0),
        ("carry_increment", incremented_a, carry_b, 1),
        ("control_base", a, control_b, 0),
        ("control_increment", incremented_a, control_b, 0),
    )
    quartet_id = f"phase7-{split}-q{quartet_index:04d}"
    template = PHASE7_TEMPLATES[split]
    return [
        Phase7CarryExample(
            example_id=f"{quartet_id}-{variant}",
            quartet_id=quartet_id,
            split=split,
            variant=variant,
            operand_a=operand_a,
            operand_b=operand_b,
            result=operand_a + operand_b,
            ones_carry=ones_carry,
            prompt=template.format(a=operand_a, b=operand_b),
        )
        for variant, operand_a, operand_b, ones_carry in values
    ]


def build_phase7_carry_quartets(
    *,
    seed: int = 20261001,
    fit_quartets_per_digit: int = 20,
    selection_quartets_per_digit: int = 5,
    development_quartets_per_digit: int = 5,
    audit_quartets_per_digit: int = 5,
) -> list[Phase7CarryExample]:
    counts = (
        fit_quartets_per_digit,
        selection_quartets_per_digit,
        development_quartets_per_digit,
        audit_quartets_per_digit,
    )
    if min(counts) < 1:
        raise ValueError("every Phase 7 split requires quartets")
    rng = random.Random(seed)
    candidates = _candidate_quartets()
    rng.shuffle(candidates)
    used = historical_canonical_pairs()
    split_counts: tuple[tuple[Phase7Split, int], ...] = (
        ("fit", fit_quartets_per_digit),
        ("selection", selection_quartets_per_digit),
        ("development", development_quartets_per_digit),
        ("audit", audit_quartets_per_digit),
    )
    total_per_digit = sum(counts)
    selected_by_digit: dict[int, list[tuple[int, int, int]]] = {
        digit: [] for digit in range(1, 10)
    }
    for a_high, b_high, a_ones in candidates:
        rows = _quartet_rows("fit", 0, a_high, b_high, a_ones)
        leading = int(str(rows[0].result)[0])
        if len(selected_by_digit[leading]) >= total_per_digit:
            continue
        canonical = {
            tuple(sorted((row.operand_a, row.operand_b))) for row in rows
        }
        if len(canonical) != 4 or canonical & used:
            continue
        used.update(canonical)
        selected_by_digit[leading].append((a_high, b_high, a_ones))
        if all(
            len(selected) == total_per_digit
            for selected in selected_by_digit.values()
        ):
            break
    if any(
        len(selected) != total_per_digit
        for selected in selected_by_digit.values()
    ):
        available = {
            digit: len(selected)
            for digit, selected in selected_by_digit.items()
        }
        raise ValueError(f"insufficient fresh Phase 7 carry quartets: {available}")

    examples: list[Phase7CarryExample] = []
    offsets = {digit: 0 for digit in range(1, 10)}
    for split, count_per_digit in split_counts:
        quartet_index = 0
        for digit in range(1, 10):
            start = offsets[digit]
            stop = start + count_per_digit
            for a_high, b_high, a_ones in selected_by_digit[digit][start:stop]:
                rows = _quartet_rows(
                    split,
                    quartet_index,
                    a_high,
                    b_high,
                    a_ones,
                )
                examples.extend(rows)
                quartet_index += 1
            offsets[digit] = stop
    assert_phase7_carry_invariants(examples)
    return examples


def assert_phase7_carry_invariants(
    examples: list[Phase7CarryExample],
) -> None:
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError("Phase 7 example IDs must be unique")
    historical = historical_canonical_pairs()
    by_split: dict[str, set[tuple[int, int]]] = {}
    quartets: dict[str, list[Phase7CarryExample]] = {}
    for row in examples:
        canonical = tuple(sorted((row.operand_a, row.operand_b)))
        if canonical in historical:
            raise ValueError("Phase 7 pair overlaps historical data")
        if canonical in by_split.setdefault(row.split, set()):
            raise ValueError("Phase 7 pair repeats within a split")
        by_split[row.split].add(canonical)
        quartets.setdefault(row.quartet_id, []).append(row)
        observed_carry = int(
            (row.operand_a % 10) + (row.operand_b % 10) >= 10
        )
        if observed_carry != row.ones_carry:
            raise ValueError(f"wrong carry label for {row.example_id}")
        if row.result != row.operand_a + row.operand_b:
            raise ValueError(f"wrong result for {row.example_id}")
    splits = sorted(by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if by_split[left] & by_split[right]:
                raise ValueError(f"Phase 7 pair leakage: {left}/{right}")
    expected_variants = {
        "carry_base",
        "carry_increment",
        "control_base",
        "control_increment",
    }
    for quartet_id, rows in quartets.items():
        by_variant = {row.variant: row for row in rows}
        if set(by_variant) != expected_variants:
            raise ValueError(f"incomplete quartet: {quartet_id}")
        carry_base = by_variant["carry_base"]
        carry_increment = by_variant["carry_increment"]
        control_base = by_variant["control_base"]
        control_increment = by_variant["control_increment"]
        if carry_increment.operand_a != carry_base.operand_a + 1:
            raise ValueError(f"carry increment mismatch: {quartet_id}")
        if control_increment.operand_a != control_base.operand_a + 1:
            raise ValueError(f"control increment mismatch: {quartet_id}")
        if carry_increment.operand_b != carry_base.operand_b:
            raise ValueError(f"carry operand mismatch: {quartet_id}")
        if control_increment.operand_b != control_base.operand_b:
            raise ValueError(f"control operand mismatch: {quartet_id}")
        if (carry_base.ones_carry, carry_increment.ones_carry) != (0, 1):
            raise ValueError(f"carry transition mismatch: {quartet_id}")
        if (control_base.ones_carry, control_increment.ones_carry) != (0, 0):
            raise ValueError(f"control transition mismatch: {quartet_id}")
    for split in splits:
        bases = [
            row
            for row in examples
            if row.split == split and row.variant == "carry_base"
        ]
        counts = {
            digit: sum(int(str(row.result)[0]) == digit for row in bases)
            for digit in range(1, 10)
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"Phase 7 quartets are not balanced: {split}")


def phase7_carry_sha256(examples: list[Phase7CarryExample]) -> str:
    encoded = json.dumps(
        [example.to_dict() for example in examples],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
