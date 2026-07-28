from open_latent_interfaces.dataset import (
    assert_dataset_invariants,
    build_phase0_dataset,
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

