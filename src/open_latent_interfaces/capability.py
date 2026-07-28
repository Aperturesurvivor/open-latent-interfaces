from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Literal

CapabilitySplit = Literal["development", "audit"]

CAPABILITY_TEMPLATES_V1 = {
    "direct": "What is {a} + {b}? Respond with only the integer.",
    "symbolic": "{a} + {b} =",
    "word_problem": (
        "A box has {a} red pieces and {b} blue pieces. "
        "How many pieces are there in total? Respond with only the integer."
    ),
}
CAPABILITY_TEMPLATES_V2 = {
    "direct": "What is {a} + {b}? Respond with only the integer.",
    "symbolic": "Compute {a} + {b}. Respond with only the integer.",
    "word_problem": (
        "There are {a} red pieces and {b} blue pieces. "
        "How many pieces are there in total? Respond with only the integer."
    ),
}
# Backward-compatible public alias for the already-run v1 sweep.
CAPABILITY_TEMPLATES = CAPABILITY_TEMPLATES_V1


@dataclass(frozen=True)
class CapabilityExample:
    example_id: str
    split: CapabilitySplit
    regime: str
    template_family: str
    presentation: Literal["raw", "chat"]
    operand_a: int
    operand_b: int
    result: int
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_first_integer(text: str) -> int | None:
    match = re.search(r"(?<!\d)-?\d+", text.replace(",", ""))
    return None if match is None else int(match.group())


def _candidate_pairs(regime: str) -> list[tuple[int, int]]:
    if regime == "single_digit_no_carry":
        return [
            (a, b)
            for a in range(1, 10)
            for b in range(a, 10)
            if a + b <= 9
        ]
    if regime == "single_digit_with_carry":
        return [
            (a, b)
            for a in range(1, 10)
            for b in range(a, 10)
            if a + b >= 10
        ]
    if regime == "two_digit_no_carry":
        return [
            (a, b)
            for a in range(10, 90)
            for b in range(a, 90)
            if a + b <= 99
            and (a % 10) + (b % 10) <= 9
            and (a // 10) + (b // 10) <= 9
        ]
    if regime == "two_digit_with_carry":
        return [
            (a, b)
            for a in range(10, 90)
            for b in range(a, 90)
            if a + b <= 99 and (a % 10) + (b % 10) >= 10
        ]
    if regime == "three_digit_mixed":
        return [
            (a, b)
            for a in range(20, 500)
            for b in range(a, 500)
            if 100 <= a + b <= 998
        ]
    raise ValueError(f"unknown capability regime: {regime}")


def build_capability_sweep(
    *,
    seed: int = 20260728,
    development_pairs: int = 12,
    audit_pairs: int = 8,
    protocol_version: str = "v1",
) -> list[CapabilityExample]:
    if min(development_pairs, audit_pairs) < 1:
        raise ValueError("both splits require at least one pair")
    regimes = (
        "single_digit_no_carry",
        "single_digit_with_carry",
        "two_digit_no_carry",
        "two_digit_with_carry",
        "three_digit_mixed",
    )
    if protocol_version == "v1":
        templates = CAPABILITY_TEMPLATES_V1
    elif protocol_version == "v2":
        templates = CAPABILITY_TEMPLATES_V2
    else:
        raise ValueError(f"unknown capability protocol version: {protocol_version}")
    examples: list[CapabilityExample] = []
    for regime_index, regime in enumerate(regimes):
        pairs = _candidate_pairs(regime)
        rng = random.Random(seed + regime_index)
        rng.shuffle(pairs)
        required = development_pairs + audit_pairs
        if len(pairs) < required:
            raise ValueError(f"{regime} has only {len(pairs)} canonical pairs")
        selections = (
            ("development", pairs[:development_pairs]),
            ("audit", pairs[development_pairs:required]),
        )
        for split, split_pairs in selections:
            for pair_index, canonical_pair in enumerate(split_pairs):
                a, b = canonical_pair
                if rng.randrange(2):
                    a, b = b, a
                for family, template in templates.items():
                    base_prompt = template.format(a=a, b=b)
                    for presentation in ("raw", "chat"):
                        examples.append(
                            CapabilityExample(
                                example_id=(
                                    f"capability-{split}-{regime}-{pair_index:02d}-"
                                    f"{family}-{presentation}"
                                ),
                                split=split,  # type: ignore[arg-type]
                                regime=regime,
                                template_family=family,
                                presentation=presentation,  # type: ignore[arg-type]
                                operand_a=a,
                                operand_b=b,
                                result=a + b,
                                prompt=base_prompt,
                            )
                        )
    assert_capability_invariants(examples)
    return examples


def assert_capability_invariants(examples: list[CapabilityExample]) -> None:
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("capability example IDs must be unique")
    conditions: dict[tuple[str, str, int, int], set[tuple[str, str]]] = {}
    pairs_by_regime_split: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for example in examples:
        canonical = tuple(sorted((example.operand_a, example.operand_b)))
        pairs_by_regime_split.setdefault((example.regime, example.split), set()).add(
            canonical
        )
        conditions.setdefault(
            (example.regime, example.split, *canonical), set()
        ).add((example.template_family, example.presentation))
        if example.result != example.operand_a + example.operand_b:
            raise ValueError(f"wrong result for {example.example_id}")
    expected_conditions = {
        (family, presentation)
        for family in {example.template_family for example in examples}
        for presentation in ("raw", "chat")
    }
    if any(value != expected_conditions for value in conditions.values()):
        raise ValueError("every pair must have all template/presentation conditions")
    for regime in {example.regime for example in examples}:
        development = pairs_by_regime_split[(regime, "development")]
        audit = pairs_by_regime_split[(regime, "audit")]
        if development & audit:
            raise ValueError(f"canonical-pair leakage in {regime}")


def capability_dataset_sha256(examples: list[CapabilityExample]) -> str:
    payload = [example.to_dict() for example in examples]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
