from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

Split = Literal["train", "development", "test", "audit"]
Kind = Literal["addition", "multiplication", "quoted", "factual"]


ADDITION_TEMPLATES = {
    "direct": "Q: What is {a} + {b}?\nA: ",
    "calculate": "Calculate {a} plus {b}. Answer with only the integer: ",
    "word_problem": (
        "A shelf has {a} red parts and {b} blue parts. "
        "How many parts are there in total? Answer with only the integer: "
    ),
    "compact": "Find the sum of {a} and {b}. Result: ",
}

MULTIPLICATION_TEMPLATES = {
    "direct": "Q: What is {a} * {b}?\nA: ",
    "calculate": "Calculate {a} times {b}. Answer with only the integer: ",
    "word_problem": (
        "There are {a} boxes with {b} parts in each box. "
        "How many parts are there? Answer with only the integer: "
    ),
    "compact": "Find the product of {a} and {b}. Result: ",
}

QUOTED_TEMPLATES = {
    "direct": 'Repeat this text without solving it: "{a} + {b}"\nText:',
    "calculate": 'The string "{a} plus {b}" is a label, not a request. Repeat the label:',
    "word_problem": (
        'Quote the following unfinished exercise without answering it: '
        '"A shelf has {a} red parts and {b} blue parts."'
    ),
    "compact": 'Copy the label "{a} + {b}" without solving it. Label: ',
}

FACTUAL_TEMPLATES = {
    "direct": "Write one sentence that mentions the years {a} and {b}:",
    "calculate": "List the two identifiers {a} and {b} without combining them:",
    "word_problem": "Compare item {a} with item {b} in one short sentence:",
    "compact": "List identifiers {a} and {b} separately: ",
}


@dataclass(frozen=True)
class Example:
    example_id: str
    split: Split
    kind: Kind
    template_family: str
    prompt: str
    operand_a: int
    operand_b: int
    result: int | None
    route: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sample_operands(rng: random.Random, count: int) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < count:
        a = rng.randint(20, 499)
        b = rng.randint(20, 499)
        if a != b:
            pairs.add((a, b))
    values = list(pairs)
    rng.shuffle(values)
    return values


def build_phase0_dataset(
    *,
    seed: int = 240727,
    train_pairs: int = 96,
    development_pairs: int = 32,
    test_pairs: int = 32,
) -> list[Example]:
    """Build template- and operand-disjoint positive/contrast examples.

    Each operand pair produces one addition and three matched negatives. The
    exact operand pairs are disjoint across splits while the numeric range is
    shared, so a value probe is evaluated by interpolation rather than accidental
    range extrapolation. Template family is balanced inside every split.
    """

    if min(train_pairs, development_pairs, test_pairs) < 3:
        raise ValueError("each split needs at least three operand pairs")
    rng = random.Random(seed)
    examples: list[Example] = []
    counts: dict[Split, int] = {
        "train": train_pairs,
        "development": development_pairs,
        "test": test_pairs,
    }
    families = tuple(ADDITION_TEMPLATES)
    template_sets: Mapping[str, Mapping[str, str]] = {
        "addition": ADDITION_TEMPLATES,
        "multiplication": MULTIPLICATION_TEMPLATES,
        "quoted": QUOTED_TEMPLATES,
        "factual": FACTUAL_TEMPLATES,
    }

    total_pairs = sum(counts.values())
    all_pairs = _sample_operands(rng, total_pairs)
    offset = 0
    for split, count in counts.items():
        split_pairs = all_pairs[offset : offset + count]
        offset += count
        for index, (a, b) in enumerate(split_pairs):
            family = families[index % len(families)]
            for kind, templates in template_sets.items():
                route = int(kind == "addition")
                result = a + b if kind == "addition" else None
                examples.append(
                    Example(
                        example_id=f"{split}-{index:04d}-{kind}",
                        split=split,
                        kind=kind,  # type: ignore[arg-type]
                        template_family=family,
                        prompt=templates[family].format(a=a, b=b),
                        operand_a=a,
                        operand_b=b,
                        result=result,
                        route=route,
                    )
                )
    return examples


def assert_dataset_invariants(examples: list[Example]) -> None:
    if not examples:
        raise ValueError("dataset is empty")
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("example IDs must be unique")

    pairs_by_split: dict[str, set[tuple[int, int]]] = {}
    for example in examples:
        pairs_by_split.setdefault(example.split, set()).add(
            (example.operand_a, example.operand_b)
        )
        if example.route and example.result != example.operand_a + example.operand_b:
            raise ValueError(f"wrong deterministic result for {example.example_id}")
        if not example.route and example.result is not None:
            raise ValueError(f"negative has a result label: {example.example_id}")

    split_names = sorted(pairs_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            if pairs_by_split[left] & pairs_by_split[right]:
                raise ValueError(f"operand-pair leakage between {left} and {right}")


def _leading_digit(value: int) -> int:
    return int(str(value)[0])


def _balanced_pairs(
    rng: random.Random,
    *,
    count_per_digit: int,
    used: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Sample exact-pair-disjoint additions balanced over leading result 1–9."""

    pairs: list[tuple[int, int]] = []
    for digit in range(1, 10):
        digit_pairs: set[tuple[int, int]] = set()
        attempts = 0
        while len(digit_pairs) < count_per_digit:
            attempts += 1
            if attempts > count_per_digit * 10_000:
                raise RuntimeError(f"could not sample enough pairs for leading digit {digit}")
            maximum = 998 if digit == 9 else digit * 100 + 99
            total = rng.randint(digit * 100, maximum)
            minimum_a = max(20, total - 499)
            maximum_a = min(499, total - 20)
            if minimum_a > maximum_a:
                continue
            a = rng.randint(minimum_a, maximum_a)
            b = total - a
            pair = (a, b)
            if pair in used or pair in digit_pairs or a == b:
                continue
            if _leading_digit(a + b) != digit:
                continue
            digit_pairs.add(pair)
        used.update(digit_pairs)
        pairs.extend(sorted(digit_pairs))
    rng.shuffle(pairs)
    return pairs


def build_phase01_dataset(
    *,
    seed: int = 240801,
    train_pairs_per_digit: int = 24,
    development_pairs_per_digit: int = 8,
    audit_pairs_per_digit: int = 8,
) -> list[Example]:
    """Build the balanced, template-held-out Phase 0.1 dataset."""

    if min(
        train_pairs_per_digit,
        development_pairs_per_digit,
        audit_pairs_per_digit,
    ) < 1:
        raise ValueError("every split needs at least one pair per leading digit")
    rng = random.Random(seed)
    used: set[tuple[int, int]] = set()
    split_specs: tuple[tuple[Split, int, tuple[str, ...]], ...] = (
        ("train", train_pairs_per_digit, ("direct", "calculate")),
        ("development", development_pairs_per_digit, ("word_problem",)),
        ("audit", audit_pairs_per_digit, ("compact",)),
    )
    template_sets: Mapping[str, Mapping[str, str]] = {
        "addition": ADDITION_TEMPLATES,
        "multiplication": MULTIPLICATION_TEMPLATES,
        "quoted": QUOTED_TEMPLATES,
        "factual": FACTUAL_TEMPLATES,
    }
    examples: list[Example] = []
    for split, count_per_digit, families in split_specs:
        pairs = _balanced_pairs(
            rng,
            count_per_digit=count_per_digit,
            used=used,
        )
        for index, (a, b) in enumerate(pairs):
            family = families[index % len(families)]
            for kind, templates in template_sets.items():
                route = int(kind == "addition")
                examples.append(
                    Example(
                        example_id=f"phase01-{split}-{index:04d}-{kind}",
                        split=split,
                        kind=kind,  # type: ignore[arg-type]
                        template_family=family,
                        prompt=templates[family].format(a=a, b=b),
                        operand_a=a,
                        operand_b=b,
                        result=a + b if route else None,
                        route=route,
                    )
                )
    assert_phase01_invariants(examples)
    return examples


def assert_phase01_invariants(examples: list[Example]) -> None:
    assert_dataset_invariants(examples)
    expected_families = {
        "train": {"direct", "calculate"},
        "development": {"word_problem"},
        "audit": {"compact"},
    }
    observed_families: dict[str, set[str]] = {}
    digit_counts: dict[str, dict[int, int]] = {}
    pair_conditions: dict[tuple[str, int, int], set[str]] = {}
    for example in examples:
        observed_families.setdefault(example.split, set()).add(example.template_family)
        pair_conditions.setdefault(
            (example.split, example.operand_a, example.operand_b),
            set(),
        ).add(example.kind)
        if example.route:
            assert example.result is not None
            digit = _leading_digit(example.result)
            digit_counts.setdefault(example.split, {}).setdefault(digit, 0)
            digit_counts[example.split][digit] += 1

    if observed_families != expected_families:
        raise ValueError(
            f"template families are not held out as specified: {observed_families}"
        )
    expected_conditions = {"addition", "multiplication", "quoted", "factual"}
    if any(conditions != expected_conditions for conditions in pair_conditions.values()):
        raise ValueError("every operand pair must have four matched semantic conditions")
    for split, counts in digit_counts.items():
        if set(counts) != set(range(1, 10)) or len(set(counts.values())) != 1:
            raise ValueError(f"leading-digit support is not balanced in {split}: {counts}")
