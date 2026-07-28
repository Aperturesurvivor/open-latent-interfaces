from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Literal

from open_latent_interfaces.phase3_data import prior_canonical_pairs

Phase4Split = Literal["fit", "selection", "development", "audit"]
CarryVariant = Literal[
    "carry_base",
    "carry_increment",
    "control_base",
    "control_increment",
]

PHASE4_TEMPLATES = {
    "fit": "Add {a} and {b}. Return exactly Answer=<integer>.",
    "selection": "Compute {a} plus {b}. Return exactly Answer=<integer>.",
    "development": "Find the sum of {a} and {b}. Return exactly Answer=<integer>.",
    "audit": (
        "A bin contains {a} parts and another contains {b} parts. "
        "Return the total exactly as Answer=<integer>."
    ),
}


@dataclass(frozen=True)
class Phase4CarryExample:
    example_id: str
    quartet_id: str
    split: Phase4Split
    variant: CarryVariant
    operand_a: int
    operand_b: int
    result: int
    ones_carry: int
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _quartet_rows(
    split: Phase4Split,
    quartet_index: int,
    a_high: int,
    b_high: int,
    a_ones: int,
) -> list[Phase4CarryExample]:
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
    quartet_id = f"phase4-{split}-q{quartet_index:04d}"
    template = PHASE4_TEMPLATES[split]
    return [
        Phase4CarryExample(
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


def _candidate_quartets() -> list[tuple[int, int, int]]:
    return [
        (a_high, b_high, a_ones)
        for a_high in range(2, 50)
        for b_high in range(a_high, 50)
        for a_ones in range(1, 5)
        if 10 <= a_high + b_high <= 98
    ]


def build_phase4_carry_quartets(
    *,
    seed: int = 20260801,
    fit_quartets_per_digit: int = 20,
    selection_quartets_per_digit: int = 5,
    development_quartets_per_digit: int = 5,
    audit_quartets_per_digit: int = 5,
) -> list[Phase4CarryExample]:
    counts = (
        fit_quartets_per_digit,
        selection_quartets_per_digit,
        development_quartets_per_digit,
        audit_quartets_per_digit,
    )
    if min(counts) < 1:
        raise ValueError("every Phase 4 split requires quartets")
    rng = random.Random(seed)
    candidates = _candidate_quartets()
    rng.shuffle(candidates)
    excluded = prior_canonical_pairs()
    used = set(excluded)
    split_counts: tuple[tuple[Phase4Split, int], ...] = (
        ("fit", fit_quartets_per_digit),
        ("selection", selection_quartets_per_digit),
        ("development", development_quartets_per_digit),
        ("audit", audit_quartets_per_digit),
    )
    examples = []
    candidate_cursor = 0
    for split, count_per_digit in split_counts:
        selected_by_digit = {digit: 0 for digit in range(1, 10)}
        quartet_index = 0
        while set(selected_by_digit.values()) != {count_per_digit}:
            if candidate_cursor >= len(candidates):
                raise ValueError("insufficient carry quartets for requested balance")
            a_high, b_high, a_ones = candidates[candidate_cursor]
            candidate_cursor += 1
            rows = _quartet_rows(
                split,
                quartet_index,
                a_high,
                b_high,
                a_ones,
            )
            leading = int(str(rows[0].result)[0])
            if selected_by_digit[leading] >= count_per_digit:
                continue
            canonical = {
                tuple(sorted((row.operand_a, row.operand_b))) for row in rows
            }
            if len(canonical) != 4 or canonical & used:
                continue
            used.update(canonical)
            selected_by_digit[leading] += 1
            examples.extend(rows)
            quartet_index += 1
    assert_phase4_carry_invariants(examples)
    return examples


def assert_phase4_carry_invariants(
    examples: list[Phase4CarryExample],
) -> None:
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("Phase 4 example IDs must be unique")
    prior = prior_canonical_pairs()
    by_split: dict[str, set[tuple[int, int]]] = {}
    quartets: dict[str, list[Phase4CarryExample]] = {}
    for example in examples:
        canonical = tuple(sorted((example.operand_a, example.operand_b)))
        if canonical in prior:
            raise ValueError("Phase 4 pair overlaps prior data")
        if canonical in by_split.setdefault(example.split, set()):
            raise ValueError("Phase 4 pair repeats within a split")
        by_split[example.split].add(canonical)
        quartets.setdefault(example.quartet_id, []).append(example)
        observed_carry = int(
            (example.operand_a % 10) + (example.operand_b % 10) >= 10
        )
        if observed_carry != example.ones_carry:
            raise ValueError(f"wrong carry label for {example.example_id}")
        if example.result != example.operand_a + example.operand_b:
            raise ValueError(f"wrong result for {example.example_id}")
    splits = sorted(by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if by_split[left] & by_split[right]:
                raise ValueError(f"Phase 4 pair leakage between {left} and {right}")
    for quartet_id, rows in quartets.items():
        by_variant = {row.variant: row for row in rows}
        if set(by_variant) != {
            "carry_base",
            "carry_increment",
            "control_base",
            "control_increment",
        }:
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
            raise ValueError(f"carry operand-b mismatch: {quartet_id}")
        if control_increment.operand_b != control_base.operand_b:
            raise ValueError(f"control operand-b mismatch: {quartet_id}")
        if carry_increment.result != carry_base.result + 1:
            raise ValueError(f"carry result increment mismatch: {quartet_id}")
        if control_increment.result != control_base.result + 1:
            raise ValueError(f"control result increment mismatch: {quartet_id}")
        if (carry_base.ones_carry, carry_increment.ones_carry) != (0, 1):
            raise ValueError(f"carry transition mismatch: {quartet_id}")
        if (control_base.ones_carry, control_increment.ones_carry) != (0, 0):
            raise ValueError(f"control transition mismatch: {quartet_id}")
    for split in splits:
        split_quartets = {
            row.quartet_id: row
            for row in examples
            if row.split == split and row.variant == "carry_base"
        }
        counts = {
            digit: sum(
                int(str(row.result)[0]) == digit
                for row in split_quartets.values()
            )
            for digit in range(1, 10)
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"carry quartets are not leading-digit balanced: {split}")


def phase4_carry_sha256(examples: list[Phase4CarryExample]) -> str:
    encoded = json.dumps(
        [example.to_dict() for example in examples],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
