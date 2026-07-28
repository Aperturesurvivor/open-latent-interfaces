from open_latent_interfaces.dataset import (
    assert_dataset_invariants,
    build_phase0_dataset,
    build_phase01_dataset,
)


def test_phase0_dataset_is_deterministic_and_split_disjoint() -> None:
    first = build_phase0_dataset(
        seed=7,
        train_pairs=6,
        development_pairs=3,
        test_pairs=3,
    )
    second = build_phase0_dataset(
        seed=7,
        train_pairs=6,
        development_pairs=3,
        test_pairs=3,
    )
    assert first == second
    assert_dataset_invariants(first)
    assert len(first) == 48


def test_each_pair_has_one_positive_and_three_matched_negatives() -> None:
    examples = build_phase0_dataset(
        seed=8,
        train_pairs=3,
        development_pairs=3,
        test_pairs=3,
    )
    groups: dict[tuple[str, int, int], list[object]] = {}
    for example in examples:
        groups.setdefault(
            (example.split, example.operand_a, example.operand_b),
            [],
        ).append(example)
    assert all(len(group) == 4 for group in groups.values())
    assert all(sum(example.route for example in group) == 1 for group in groups.values())


def test_phase01_is_digit_balanced_pair_disjoint_and_template_held_out() -> None:
    examples = build_phase01_dataset(
        seed=9,
        train_pairs_per_digit=3,
        development_pairs_per_digit=2,
        audit_pairs_per_digit=2,
    )
    positives = [example for example in examples if example.route]
    family_by_split: dict[str, set[str]] = {}
    counts: dict[str, dict[int, int]] = {}
    pairs: dict[str, set[tuple[int, int]]] = {}
    for example in positives:
        family_by_split.setdefault(example.split, set()).add(example.template_family)
        digit = int(str(example.result)[0])
        counts.setdefault(example.split, {}).setdefault(digit, 0)
        counts[example.split][digit] += 1
        pairs.setdefault(example.split, set()).add((example.operand_a, example.operand_b))
    assert family_by_split == {
        "train": {"direct", "calculate"},
        "development": {"word_problem"},
        "audit": {"compact"},
    }
    assert all(set(split_counts.values()) == {expected} for split_counts, expected in (
        (counts["train"], 3),
        (counts["development"], 2),
        (counts["audit"], 2),
    ))
    assert not (pairs["train"] & pairs["development"])
    assert not (pairs["train"] & pairs["audit"])
    assert not (pairs["development"] & pairs["audit"])
