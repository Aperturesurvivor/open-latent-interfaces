from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

Split = Literal["train", "development", "test"]
Kind = Literal["addition", "multiplication", "quoted", "factual"]


ADDITION_TEMPLATES = {
    "direct": "Q: What is {a} + {b}?\nA: ",
    "calculate": "Calculate {a} plus {b}. Answer with only the integer: ",
    "word_problem": (
        "A shelf has {a} red parts and {b} blue parts. "
        "How many parts are there in total? Answer with only the integer: "
    ),
}

MULTIPLICATION_TEMPLATES = {
    "direct": "Q: What is {a} * {b}?\nA: ",
    "calculate": "Calculate {a} times {b}. Answer with only the integer: ",
    "word_problem": (
        "There are {a} boxes with {b} parts in each box. "
        "How many parts are there? Answer with only the integer: "
    ),
}

QUOTED_TEMPLATES = {
    "direct": 'Repeat this text without solving it: "{a} + {b}"\nText:',
    "calculate": 'The string "{a} plus {b}" is a label, not a request. Repeat the label:',
    "word_problem": (
        'Quote the following unfinished exercise without answering it: '
        '"A shelf has {a} red parts and {b} blue parts."'
    ),
}

FACTUAL_TEMPLATES = {
    "direct": "Write one sentence that mentions the years {a} and {b}:",
    "calculate": "List the two identifiers {a} and {b} without combining them:",
    "word_problem": "Compare item {a} with item {b} in one short sentence:",
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
