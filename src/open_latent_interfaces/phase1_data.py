from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Literal

from open_latent_interfaces.capability import build_capability_sweep

Phase1Split = Literal["train", "development", "audit"]

PHASE1_TEMPLATES = {
    "train": "What is {a} + {b}? Respond with only the integer.",
    "development": "Compute {a} + {b}. Respond with only the integer.",
    "audit": (
        "There are {a} red pieces and {b} blue pieces. "
        "How many pieces are there in total? Respond with only the integer."
    ),
}


@dataclass(frozen=True)
class Phase1Addition:
    example_id: str
    split: Phase1Split
    template_family: str
    operand_a: int
    operand_b: int
    result: int
    ones_carry: int
    tens_carry: int
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _carry_labels(a: int, b: int) -> tuple[int, int]:
    ones = int((a % 10) + (b % 10) >= 10)
    tens = int(((a // 10) % 10) + ((b // 10) % 10) + ones >= 10)
    return ones, tens


def _excluded_capability_pairs() -> set[tuple[int, int]]:
    return {
        tuple(sorted((example.operand_a, example.operand_b)))
        for example in build_capability_sweep(protocol_version="v2")
    }


def _balanced_pairs(
    rng: random.Random,
    *,
    count_per_leading_digit: int,
    used: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    pairs = []
    for leading_digit in range(1, 10):
        selected: set[tuple[int, int]] = set()
        while len(selected) < count_per_leading_digit:
            maximum = 998 if leading_digit == 9 else leading_digit * 100 + 99
            result = rng.randint(leading_digit * 100, maximum)
            minimum_a = max(20, result - 499)
            maximum_a = min(499, result - 20)
            a = rng.randint(minimum_a, maximum_a)
            b = result - a
            canonical = tuple(sorted((a, b)))
            if a == b or canonical in used or canonical in selected:
                continue
            if rng.randrange(2):
                a, b = b, a
            selected.add(canonical)
            pairs.append((a, b))
        used.update(selected)
    rng.shuffle(pairs)
    return pairs


def build_phase1_additions(
    *,
    seed: int = 20260730,
    train_pairs_per_digit: int = 10,
    development_pairs_per_digit: int = 5,
    audit_pairs_per_digit: int = 5,
) -> list[Phase1Addition]:
    if min(
        train_pairs_per_digit,
        development_pairs_per_digit,
        audit_pairs_per_digit,
    ) < 1:
        raise ValueError("every Phase 1 split requires at least one pair per digit")
    rng = random.Random(seed)
    used = _excluded_capability_pairs()
    split_counts: tuple[tuple[Phase1Split, int], ...] = (
        ("train", train_pairs_per_digit),
        ("development", development_pairs_per_digit),
        ("audit", audit_pairs_per_digit),
    )
    examples = []
    for split, count in split_counts:
        pairs = _balanced_pairs(rng, count_per_leading_digit=count, used=used)
        for index, (a, b) in enumerate(pairs):
            result = a + b
            ones_carry, tens_carry = _carry_labels(a, b)
            examples.append(
                Phase1Addition(
                    example_id=f"phase1-{split}-{index:03d}",
                    split=split,
                    template_family=split,
                    operand_a=a,
                    operand_b=b,
                    result=result,
                    ones_carry=ones_carry,
                    tens_carry=tens_carry,
                    prompt=PHASE1_TEMPLATES[split].format(a=a, b=b),
                )
            )
    assert_phase1_addition_invariants(examples)
    return examples


def assert_phase1_addition_invariants(examples: list[Phase1Addition]) -> None:
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("Phase 1 example IDs must be unique")
    excluded = _excluded_capability_pairs()
    by_split: dict[str, set[tuple[int, int]]] = {}
    leading_counts: dict[str, dict[int, int]] = {}
    for example in examples:
        canonical = tuple(sorted((example.operand_a, example.operand_b)))
        if canonical in excluded:
            raise ValueError("Phase 1 pair overlaps the capability gate")
        by_split.setdefault(example.split, set()).add(canonical)
        leading = int(str(example.result)[0])
        leading_counts.setdefault(example.split, {}).setdefault(leading, 0)
        leading_counts[example.split][leading] += 1
        expected_carries = _carry_labels(example.operand_a, example.operand_b)
        if (example.ones_carry, example.tens_carry) != expected_carries:
            raise ValueError(f"wrong carry labels for {example.example_id}")
    splits = sorted(by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if by_split[left] & by_split[right]:
                raise ValueError(f"Phase 1 pair leakage between {left} and {right}")
    for split, counts in leading_counts.items():
        if len(counts) != 9 or len(set(counts.values())) != 1:
            raise ValueError(f"leading digits are not balanced in {split}")


def phase1_addition_sha256(examples: list[Phase1Addition]) -> str:
    payload = [example.to_dict() for example in examples]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
