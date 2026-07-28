from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache

from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
)
from open_latent_interfaces.phase4_data import (
    build_phase4_carry_quartets,
    phase4_carry_sha256,
)
from open_latent_interfaces.phase6_data import (
    REFERENCE_PARAMETERS,
    build_phase6_carry_quartets,
    phase6_carry_sha256,
)
from open_latent_interfaces.phase7_data import (
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)

PHASE9E_TEMPLATES = (
    (
        "Without showing work, combine the quantities {a} and {b}; "
        "emit only Answer=<integer>."
    ),
    (
        "Evaluate the total obtained from {a} together with {b}. "
        "Use only Answer=<integer>."
    ),
    (
        "A ledger lists entries of {a} and {b}. Give their aggregate as "
        "Answer=<integer> and nothing else."
    ),
)

PHASE7_PARAMETERS = {
    "seed": 20261001,
    "fit_quartets_per_digit": 20,
    "selection_quartets_per_digit": 5,
    "development_quartets_per_digit": 5,
    "audit_quartets_per_digit": 5,
}


@dataclass(frozen=True)
class Phase9EAuditExample:
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
    phase3 = build_phase3_additions()
    phase4 = build_phase4_carry_quartets(**REFERENCE_PARAMETERS)
    phase6 = build_phase6_carry_quartets()
    phase7 = build_phase7_carry_quartets(**PHASE7_PARAMETERS)
    return tuple(
        {
            "phase3": phase3_addition_sha256(phase3),
            "phase4": phase4_carry_sha256(phase4),
            "phase6": phase6_carry_sha256(phase6),
            "phase7": phase7_carry_sha256(phase7),
        }.items()
    )


def prior_dataset_hashes() -> dict[str, str]:
    return dict(_prior_dataset_hash_items())


@lru_cache(maxsize=1)
def _prior_canonical_pairs() -> frozenset[tuple[int, int]]:
    sources = (
        build_phase3_additions(),
        build_phase4_carry_quartets(**REFERENCE_PARAMETERS),
        build_phase6_carry_quartets(),
        build_phase7_carry_quartets(**PHASE7_PARAMETERS),
    )
    return frozenset().union(*(_canonical_pairs(rows) for rows in sources))


def prior_canonical_pairs() -> set[tuple[int, int]]:
    return set(_prior_canonical_pairs())


def _carry_labels(a: int, b: int) -> tuple[int, int]:
    ones = int((a % 10) + (b % 10) >= 10)
    tens = int(((a // 10) % 10) + ((b // 10) % 10) + ones >= 10)
    return ones, tens


def build_phase9e_audit(
    *,
    seed: int = 20261205,
) -> list[Phase9EAuditExample]:
    rng = random.Random(seed)
    used = prior_canonical_pairs()
    selected: set[tuple[int, int]] = set()
    examples = []
    for leading_digit in range(1, 10):
        carry_feasible_tens = [
            tens
            for tens in range(10)
            if (tens + 3 * leading_digit) % 10 != 9
        ]
        rotation = leading_digit % len(carry_feasible_tens)
        rotated = (
            carry_feasible_tens[rotation:]
            + carry_feasible_tens[:rotation]
        )
        carry_tens = set(rotated[:5])
        for tens_digit in range(10):
            ones_digit = (tens_digit + 3 * leading_digit) % 10
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
                    "could not construct fresh Phase 9E audit pair for "
                    f"result={result}, carry={desired_ones_carry}"
                )
            a, b = pair
            if rng.randrange(2):
                a, b = b, a
            ones_carry, tens_carry = _carry_labels(a, b)
            template_index = (leading_digit + tens_digit) % len(
                PHASE9E_TEMPLATES
            )
            examples.append(
                Phase9EAuditExample(
                    example_id=(
                        f"phase9e-audit-l{leading_digit}-"
                        f"t{tens_digit}-o{ones_digit}"
                    ),
                    template_family=f"phase9e-audit-{template_index}",
                    operand_a=a,
                    operand_b=b,
                    result=result,
                    leading_digit=leading_digit,
                    tens_digit=tens_digit,
                    ones_digit=ones_digit,
                    ones_carry=ones_carry,
                    tens_carry=tens_carry,
                    prompt=PHASE9E_TEMPLATES[template_index].format(a=a, b=b),
                )
            )
    rng.shuffle(examples)
    assert_phase9e_invariants(examples)
    return examples


def assert_phase9e_invariants(examples: list[Phase9EAuditExample]) -> None:
    if len(examples) != 90:
        raise ValueError("Phase 9E audit must contain exactly 90 examples")
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError("Phase 9E example IDs must be unique")
    if len({row.prompt for row in examples}) != len(examples):
        raise ValueError("Phase 9E prompts must be unique")
    canonical = _canonical_pairs(examples)
    if len(canonical) != len(examples):
        raise ValueError("Phase 9E canonical pairs must be unique")
    if canonical & prior_canonical_pairs():
        raise ValueError("Phase 9E pair overlaps prior data")
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
            {f"phase9e-audit-{index}": 30 for index in range(3)},
            [row.template_family for row in examples],
        ),
    }
    for name, (expected, values) in expected_counts.items():
        observed = {value: values.count(value) for value in set(values)}
        if observed != expected:
            raise ValueError(
                f"Phase 9E {name} distribution mismatch: {observed}"
            )


def phase9e_audit_sha256(examples: list[Phase9EAuditExample]) -> str:
    encoded = json.dumps(
        [example.to_dict() for example in examples],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
