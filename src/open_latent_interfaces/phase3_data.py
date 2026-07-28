from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Literal

from open_latent_interfaces.capability import build_capability_sweep
from open_latent_interfaces.phase1_data import build_phase1_additions
from open_latent_interfaces.phase2_data import build_phase2_additions

Phase3Split = Literal["fit", "selection", "development", "audit"]

PHASE3_TEMPLATES = {
    "fit": "Add {a} and {b}. Return exactly Answer=<integer>.",
    "selection": "Compute the sum {a} + {b}. Return exactly Answer=<integer>.",
    "development": "Find the total of {a} and {b}. Return exactly Answer=<integer>.",
    "audit": (
        "One shelf contains {a} items and another contains {b} items. "
        "Return their combined count exactly as Answer=<integer>."
    ),
}


@dataclass(frozen=True)
class Phase3Addition:
    example_id: str
    split: Phase3Split
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


def prior_canonical_pairs() -> set[tuple[int, int]]:
    sources = (
        build_capability_sweep(protocol_version="v2"),
        build_phase1_additions(),
        build_phase2_additions(),
    )
    return {
        tuple(sorted((example.operand_a, example.operand_b)))
        for examples in sources
        for example in examples
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


def build_phase3_additions(
    *,
    seed: int = 20260729,
    fit_pairs_per_digit: int = 50,
    selection_pairs_per_digit: int = 10,
    development_pairs_per_digit: int = 10,
    audit_pairs_per_digit: int = 10,
) -> list[Phase3Addition]:
    counts = (
        fit_pairs_per_digit,
        selection_pairs_per_digit,
        development_pairs_per_digit,
        audit_pairs_per_digit,
    )
    if min(counts) < 1:
        raise ValueError("every Phase 3 split requires at least one pair per digit")
    rng = random.Random(seed)
    used = prior_canonical_pairs()
    split_counts: tuple[tuple[Phase3Split, int], ...] = (
        ("fit", fit_pairs_per_digit),
        ("selection", selection_pairs_per_digit),
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
                Phase3Addition(
                    example_id=f"phase3-{split}-{index:04d}",
                    split=split,
                    template_family=f"phase3-{split}",
                    operand_a=a,
                    operand_b=b,
                    result=result,
                    ones_carry=ones_carry,
                    tens_carry=tens_carry,
                    prompt=PHASE3_TEMPLATES[split].format(a=a, b=b),
                )
            )
    assert_phase3_addition_invariants(examples)
    return examples


def assert_phase3_addition_invariants(examples: list[Phase3Addition]) -> None:
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("Phase 3 example IDs must be unique")
    excluded = prior_canonical_pairs()
    by_split: dict[str, set[tuple[int, int]]] = {}
    rendered_by_split: dict[str, set[str]] = {}
    leading_counts: dict[str, dict[int, int]] = {}
    for example in examples:
        canonical = tuple(sorted((example.operand_a, example.operand_b)))
        if canonical in excluded:
            raise ValueError("Phase 3 pair overlaps prior data")
        by_split.setdefault(example.split, set()).add(canonical)
        rendered_by_split.setdefault(example.split, set()).add(example.prompt)
        leading = int(str(example.result)[0])
        leading_counts.setdefault(example.split, {}).setdefault(leading, 0)
        leading_counts[example.split][leading] += 1
        if (example.ones_carry, example.tens_carry) != _carry_labels(
            example.operand_a,
            example.operand_b,
        ):
            raise ValueError(f"wrong carry labels for {example.example_id}")
    splits = sorted(by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            if by_split[left] & by_split[right]:
                raise ValueError(f"Phase 3 pair leakage between {left} and {right}")
            if rendered_by_split[left] & rendered_by_split[right]:
                raise ValueError(f"Phase 3 prompt leakage between {left} and {right}")
    for split, counts in leading_counts.items():
        if len(counts) != 9 or len(set(counts.values())) != 1:
            raise ValueError(f"leading digits are not balanced in {split}")


def phase3_addition_sha256(examples: list[Phase3Addition]) -> str:
    payload = [example.to_dict() for example in examples]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
