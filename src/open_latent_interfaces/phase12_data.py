from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Literal

from open_latent_interfaces.phase11_audit_data import (
    build_phase11_audit,
    phase11_audit_sha256,
)
from open_latent_interfaces.phase11_audit_data import (
    prior_canonical_pairs as phase11_prior_canonical_pairs,
)
from open_latent_interfaces.phase11_audit_data import (
    prior_dataset_hashes as phase11_prior_dataset_hashes,
)

Phase12Split = Literal["selection", "development"]

PHASE12_TEMPLATES = {
    "selection": (
        (
            "Produce the integer sum for {a} plus {b}; the complete response "
            "must be Answer=<integer>."
        ),
        (
            "Resolve the addition of {a} with {b}. Return only "
            "Answer=<integer>."
        ),
        (
            "Combine ledger values {a} and {b}, then print exactly "
            "Answer=<integer>."
        ),
    ),
    "development": (
        (
            "Find the total of counters {a} and {b}. No work; use "
            "Answer=<integer> only."
        ),
        (
            "Give the result of adding {a} to {b} in the form "
            "Answer=<integer> and nothing more."
        ),
        (
            "An accumulator receives {a} and {b}. Report its final value as "
            "Answer=<integer>."
        ),
    ),
}


@dataclass(frozen=True)
class Phase12Example:
    example_id: str
    split: Phase12Split
    template_family: str
    operand_a: int
    operand_b: int
    result: int
    leading_digit: int
    tens_digit: int
    ones_digit: int
    ones_carry: int
    tens_carry: int
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_pairs(examples: list[object]) -> set[tuple[int, int]]:
    return {
        tuple(sorted((example.operand_a, example.operand_b)))
        for example in examples
    }


@lru_cache(maxsize=1)
def _prior_dataset_hash_items() -> tuple[tuple[str, str], ...]:
    hashes = phase11_prior_dataset_hashes()
    phase11 = build_phase11_audit()
    hashes["phase11"] = phase11_audit_sha256(phase11)
    return tuple(hashes.items())


def prior_dataset_hashes() -> dict[str, str]:
    return dict(_prior_dataset_hash_items())


@lru_cache(maxsize=1)
def _prior_canonical_pairs() -> frozenset[tuple[int, int]]:
    phase11 = build_phase11_audit()
    return frozenset(
        phase11_prior_canonical_pairs() | _canonical_pairs(phase11)
    )


def prior_canonical_pairs() -> set[tuple[int, int]]:
    return set(_prior_canonical_pairs())


def _carry_labels(a: int, b: int) -> tuple[int, int]:
    ones = int((a % 10) + (b % 10) >= 10)
    tens = int(((a // 10) % 10) + ((b // 10) % 10) + ones >= 10)
    return ones, tens


def build_phase12_examples(
    *,
    seed: int = 20261307,
) -> list[Phase12Example]:
    rng = random.Random(seed)
    used = prior_canonical_pairs()
    selected: set[tuple[int, int]] = set()
    examples = []
    split_parameters: tuple[
        tuple[Phase12Split, int, int],
        ...,
    ] = (
        ("selection", 5, 1),
        ("development", 2, 3),
    )
    for split, multiplier, offset in split_parameters:
        for leading_digit in range(1, 10):
            feasible_tens = [
                tens
                for tens in range(10)
                if (tens + multiplier * leading_digit + offset) % 10 != 9
            ]
            rotation = (leading_digit + offset) % len(feasible_tens)
            rotated = feasible_tens[rotation:] + feasible_tens[:rotation]
            carry_tens = set(rotated[:5])
            for tens_digit in range(10):
                ones_digit = (
                    tens_digit + multiplier * leading_digit + offset
                ) % 10
                result = (
                    100 * leading_digit + 10 * tens_digit + ones_digit
                )
                desired_ones_carry = int(tens_digit in carry_tens)
                candidates = list(range(20, result - 19))
                rng.shuffle(candidates)
                pair: tuple[int, int] | None = None
                for candidate_a in candidates:
                    candidate_b = result - candidate_a
                    canonical = tuple(sorted((candidate_a, candidate_b)))
                    if (
                        candidate_a == candidate_b
                        or canonical in used
                        or canonical in selected
                    ):
                        continue
                    ones_carry, _ = _carry_labels(candidate_a, candidate_b)
                    if ones_carry != desired_ones_carry:
                        continue
                    pair = (candidate_a, candidate_b)
                    selected.add(canonical)
                    break
                if pair is None:
                    raise ValueError(
                        "could not construct fresh Phase 12 pair for "
                        f"split={split}, result={result}, "
                        f"carry={desired_ones_carry}"
                    )
                a, b = pair
                if rng.randrange(2):
                    a, b = b, a
                ones_carry, tens_carry = _carry_labels(a, b)
                templates = PHASE12_TEMPLATES[split]
                template_index = (
                    leading_digit + 2 * tens_digit + offset
                ) % len(templates)
                examples.append(
                    Phase12Example(
                        example_id=(
                            f"phase12-{split}-l{leading_digit}-"
                            f"t{tens_digit}-o{ones_digit}"
                        ),
                        split=split,
                        template_family=f"phase12-{split}-{template_index}",
                        operand_a=a,
                        operand_b=b,
                        result=result,
                        leading_digit=leading_digit,
                        tens_digit=tens_digit,
                        ones_digit=ones_digit,
                        ones_carry=ones_carry,
                        tens_carry=tens_carry,
                        prompt=templates[template_index].format(a=a, b=b),
                    )
                )
    rng.shuffle(examples)
    assert_phase12_invariants(examples)
    return examples


def assert_phase12_invariants(examples: list[Phase12Example]) -> None:
    if len(examples) != 180:
        raise ValueError("Phase 12 must contain exactly 180 examples")
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError("Phase 12 example IDs must be unique")
    if len({row.prompt for row in examples}) != len(examples):
        raise ValueError("Phase 12 prompts must be unique")
    canonical = _canonical_pairs(examples)
    if len(canonical) != len(examples):
        raise ValueError("Phase 12 canonical pairs must be unique")
    if canonical & prior_canonical_pairs():
        raise ValueError("Phase 12 pair overlaps prior data")
    selection_pairs = _canonical_pairs(
        [row for row in examples if row.split == "selection"]
    )
    development_pairs = _canonical_pairs(
        [row for row in examples if row.split == "development"]
    )
    if selection_pairs & development_pairs:
        raise ValueError("Phase 12 selection/development pair leakage")
    for row in examples:
        if row.result != row.operand_a + row.operand_b:
            raise ValueError(f"incorrect result for {row.example_id}")
        digits = [int(digit) for digit in str(row.result)]
        if len(digits) != 3 or digits != [
            row.leading_digit,
            row.tens_digit,
            row.ones_digit,
        ]:
            raise ValueError(f"incorrect digit labels for {row.example_id}")
        if (row.ones_carry, row.tens_carry) != _carry_labels(
            row.operand_a,
            row.operand_b,
        ):
            raise ValueError(f"incorrect carry labels for {row.example_id}")
    for split in ("selection", "development"):
        split_rows = [row for row in examples if row.split == split]
        expected_counts = {
            "leading": (
                {digit: 10 for digit in range(1, 10)},
                [row.leading_digit for row in split_rows],
            ),
            "tens": (
                {digit: 9 for digit in range(10)},
                [row.tens_digit for row in split_rows],
            ),
            "ones": (
                {digit: 9 for digit in range(10)},
                [row.ones_digit for row in split_rows],
            ),
            "ones_carry": (
                {0: 45, 1: 45},
                [row.ones_carry for row in split_rows],
            ),
            "template": (
                {
                    f"phase12-{split}-{index}": 30
                    for index in range(3)
                },
                [row.template_family for row in split_rows],
            ),
        }
        for name, (expected, values) in expected_counts.items():
            observed = {value: values.count(value) for value in set(values)}
            if observed != expected:
                raise ValueError(
                    f"Phase 12 {split} {name} distribution mismatch: "
                    f"{observed}"
                )


def phase12_sha256(examples: list[Phase12Example]) -> str:
    encoded = json.dumps(
        [example.to_dict() for example in examples],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
