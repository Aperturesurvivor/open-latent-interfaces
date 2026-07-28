from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Literal

from open_latent_interfaces.phase12_audit_data import (
    build_phase12_audit,
    phase12_audit_sha256,
)
from open_latent_interfaces.phase12_audit_data import (
    prior_canonical_pairs as phase12_audit_prior_canonical_pairs,
)
from open_latent_interfaces.phase12_audit_data import (
    prior_dataset_hashes as phase12_audit_prior_dataset_hashes,
)

Phase13Split = Literal["fit", "selection", "development"]

PHASE13_TEMPLATES = {
    "fit": (
        (
            "Add the values in register A={a} and register B={b}. Return only "
            "Answer=<integer>."
        ),
        (
            "What integer results when {a} is combined with {b}? Use exactly "
            "Answer=<integer>."
        ),
        (
            "A counter holds {a} and receives {b}. State the new count as "
            "Answer=<integer>."
        ),
    ),
    "selection": (
        (
            "Evaluate the integer total for inputs {a} and {b}; reply solely "
            "Answer=<integer>."
        ),
        (
            "Place {a} and {b} into an adder and report its output in the form "
            "Answer=<integer>."
        ),
        (
            "Two bins contain {a} and {b} items. Give their joint count only "
            "as Answer=<integer>."
        ),
    ),
    "development": (
        (
            "Return the exact sum of quantities {a} and {b}, formatted only "
            "as Answer=<integer>."
        ),
        (
            "An integer accumulator starts at {a}, then increases by {b}. "
            "Output Answer=<integer>."
        ),
        (
            "Resolve {a} added to {b}. The response must contain exactly "
            "Answer=<integer>."
        ),
    ),
}


@dataclass(frozen=True)
class Phase13Example:
    example_id: str
    split: Phase13Split
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
    hashes = phase12_audit_prior_dataset_hashes()
    phase12_audit = build_phase12_audit()
    hashes["phase12_audit"] = phase12_audit_sha256(phase12_audit)
    return tuple(hashes.items())


def prior_dataset_hashes() -> dict[str, str]:
    return dict(_prior_dataset_hash_items())


@lru_cache(maxsize=1)
def _prior_canonical_pairs() -> frozenset[tuple[int, int]]:
    phase12_audit = build_phase12_audit()
    return frozenset(
        phase12_audit_prior_canonical_pairs()
        | _canonical_pairs(phase12_audit)
    )


def prior_canonical_pairs() -> set[tuple[int, int]]:
    return set(_prior_canonical_pairs())


def _carry_labels(a: int, b: int) -> tuple[int, int]:
    ones = int((a % 10) + (b % 10) >= 10)
    tens = int(((a // 10) % 10) + ((b // 10) % 10) + ones >= 10)
    return ones, tens


def build_phase13_examples(
    *,
    seed: int = 20261313,
) -> list[Phase13Example]:
    rng = random.Random(seed)
    used = prior_canonical_pairs()
    selected: set[tuple[int, int]] = set()
    examples = []
    split_parameters: tuple[
        tuple[Phase13Split, int, int, int],
        ...,
    ] = (
        ("fit", 7, 2, 1),
        ("selection", 3, 6, 4),
        ("development", 9, 1, 7),
    )
    for split, multiplier, offset, carry_rotation in split_parameters:
        for leading_digit in range(1, 10):
            feasible_tens = [
                tens
                for tens in range(10)
                if (tens + multiplier * leading_digit + offset) % 10 != 9
            ]
            rotation = (
                leading_digit + carry_rotation
            ) % len(feasible_tens)
            rotated_tens = (
                feasible_tens[rotation:] + feasible_tens[:rotation]
            )
            carry_tens = set(rotated_tens[:5])
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
                        "could not construct fresh Phase 13 pair for "
                        f"split={split}, result={result}, "
                        f"carry={desired_ones_carry}"
                    )
                a, b = pair
                if rng.randrange(2):
                    a, b = b, a
                ones_carry, tens_carry = _carry_labels(a, b)
                templates = PHASE13_TEMPLATES[split]
                grid_index = (leading_digit - 1) * 10 + tens_digit
                template_index = grid_index % len(templates)
                examples.append(
                    Phase13Example(
                        example_id=(
                            f"phase13-{split}-l{leading_digit}-"
                            f"t{tens_digit}-o{ones_digit}"
                        ),
                        split=split,
                        template_family=f"phase13-{split}-{template_index}",
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
    assert_phase13_invariants(examples)
    return examples


def assert_phase13_invariants(examples: list[Phase13Example]) -> None:
    if len(examples) != 270:
        raise ValueError("Phase 13 must contain exactly 270 examples")
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError("Phase 13 example IDs must be unique")
    if len({row.prompt for row in examples}) != len(examples):
        raise ValueError("Phase 13 prompts must be unique")
    canonical = _canonical_pairs(examples)
    if len(canonical) != len(examples):
        raise ValueError("Phase 13 canonical pairs must be unique")
    if canonical & prior_canonical_pairs():
        raise ValueError("Phase 13 pair overlaps prior data")
    split_pairs = {
        split: _canonical_pairs(
            [row for row in examples if row.split == split]
        )
        for split in ("fit", "selection", "development")
    }
    for left, right in (
        ("fit", "selection"),
        ("fit", "development"),
        ("selection", "development"),
    ):
        if split_pairs[left] & split_pairs[right]:
            raise ValueError(f"Phase 13 {left}/{right} pair leakage")
    for row in examples:
        if row.result != row.operand_a + row.operand_b:
            raise ValueError(f"incorrect result for {row.example_id}")
        digits = [int(digit) for digit in str(row.result)]
        if digits != [
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
    for split in ("fit", "selection", "development"):
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
                    f"phase13-{split}-{index}": 30
                    for index in range(3)
                },
                [row.template_family for row in split_rows],
            ),
        }
        for name, (expected, values) in expected_counts.items():
            observed = {value: values.count(value) for value in set(values)}
            if observed != expected:
                raise ValueError(
                    f"Phase 13 {split} {name} distribution mismatch: "
                    f"{observed}"
                )


def phase13_sha256(examples: list[Phase13Example]) -> str:
    encoded = json.dumps(
        [example.to_dict() for example in examples],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
