from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache

from open_latent_interfaces.phase12_data import (
    build_phase12_examples,
    phase12_sha256,
)
from open_latent_interfaces.phase12_data import (
    prior_canonical_pairs as phase12_prior_canonical_pairs,
)
from open_latent_interfaces.phase12_data import (
    prior_dataset_hashes as phase12_prior_dataset_hashes,
)

PHASE12_AUDIT_TEMPLATES = (
    (
        "Calculate the sum stored in cells {a} and {b}. Respond with only "
        "Answer=<integer>."
    ),
    (
        "Merge quantities {a} and {b} and provide the resulting integer as "
        "Answer=<integer>."
    ),
    (
        "A tally receives batches of {a} and {b}. Output its total exactly as "
        "Answer=<integer>."
    ),
)


@dataclass(frozen=True)
class Phase12AuditExample:
    example_id: str
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
    hashes = phase12_prior_dataset_hashes()
    phase12 = build_phase12_examples()
    hashes["phase12"] = phase12_sha256(phase12)
    return tuple(hashes.items())


def prior_dataset_hashes() -> dict[str, str]:
    return dict(_prior_dataset_hash_items())


@lru_cache(maxsize=1)
def _prior_canonical_pairs() -> frozenset[tuple[int, int]]:
    phase12 = build_phase12_examples()
    return frozenset(
        phase12_prior_canonical_pairs() | _canonical_pairs(phase12)
    )


def prior_canonical_pairs() -> set[tuple[int, int]]:
    return set(_prior_canonical_pairs())


def _carry_labels(a: int, b: int) -> tuple[int, int]:
    ones = int((a % 10) + (b % 10) >= 10)
    tens = int(((a // 10) % 10) + ((b // 10) % 10) + ones >= 10)
    return ones, tens


def build_phase12_audit(
    *,
    seed: int = 20261310,
) -> list[Phase12AuditExample]:
    rng = random.Random(seed)
    used = prior_canonical_pairs()
    selected: set[tuple[int, int]] = set()
    examples = []
    for leading_digit in range(1, 10):
        feasible_tens = [
            tens
            for tens in range(10)
            if (tens + 3 * leading_digit + 4) % 10 != 9
        ]
        rotation = (3 * leading_digit) % len(feasible_tens)
        rotated = feasible_tens[rotation:] + feasible_tens[:rotation]
        carry_tens = set(rotated[:5])
        for tens_digit in range(10):
            ones_digit = (tens_digit + 3 * leading_digit + 4) % 10
            result = 100 * leading_digit + 10 * tens_digit + ones_digit
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
                    "could not construct fresh Phase 12 audit pair for "
                    f"result={result}, carry={desired_ones_carry}"
                )
            a, b = pair
            if rng.randrange(2):
                a, b = b, a
            ones_carry, tens_carry = _carry_labels(a, b)
            grid_index = (leading_digit - 1) * 10 + tens_digit
            template_index = grid_index % len(PHASE12_AUDIT_TEMPLATES)
            examples.append(
                Phase12AuditExample(
                    example_id=(
                        f"phase12-audit-l{leading_digit}-"
                        f"t{tens_digit}-o{ones_digit}"
                    ),
                    template_family=f"phase12-audit-{template_index}",
                    operand_a=a,
                    operand_b=b,
                    result=result,
                    leading_digit=leading_digit,
                    tens_digit=tens_digit,
                    ones_digit=ones_digit,
                    ones_carry=ones_carry,
                    tens_carry=tens_carry,
                    prompt=PHASE12_AUDIT_TEMPLATES[template_index].format(
                        a=a,
                        b=b,
                    ),
                )
            )
    rng.shuffle(examples)
    assert_phase12_audit_invariants(examples)
    return examples


def assert_phase12_audit_invariants(
    examples: list[Phase12AuditExample],
) -> None:
    if len(examples) != 90:
        raise ValueError("Phase 12 audit must contain exactly 90 examples")
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError("Phase 12 audit IDs must be unique")
    if len({row.prompt for row in examples}) != len(examples):
        raise ValueError("Phase 12 audit prompts must be unique")
    canonical = _canonical_pairs(examples)
    if len(canonical) != len(examples):
        raise ValueError("Phase 12 audit pairs must be unique")
    if canonical & prior_canonical_pairs():
        raise ValueError("Phase 12 audit pair overlaps prior data")
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
    expected_counts = {
        "leading": (
            {digit: 10 for digit in range(1, 10)},
            [row.leading_digit for row in examples],
        ),
        "tens": (
            {digit: 9 for digit in range(10)},
            [row.tens_digit for row in examples],
        ),
        "ones": (
            {digit: 9 for digit in range(10)},
            [row.ones_digit for row in examples],
        ),
        "ones_carry": (
            {0: 45, 1: 45},
            [row.ones_carry for row in examples],
        ),
        "template": (
            {f"phase12-audit-{index}": 30 for index in range(3)},
            [row.template_family for row in examples],
        ),
    }
    for name, (expected, values) in expected_counts.items():
        observed = {value: values.count(value) for value in set(values)}
        if observed != expected:
            raise ValueError(
                f"Phase 12 audit {name} distribution mismatch: {observed}"
            )


def phase12_audit_sha256(examples: list[Phase12AuditExample]) -> str:
    encoded = json.dumps(
        [example.to_dict() for example in examples],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
