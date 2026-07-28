from open_latent_interfaces.phase3_data import (
    build_phase3_additions,
    phase3_addition_sha256,
    prior_canonical_pairs,
)


def canonical_pairs(examples: list[object]) -> set[tuple[int, int]]:
    return {
        tuple(sorted((example.operand_a, example.operand_b)))  # type: ignore[attr-defined]
        for example in examples
    }


def test_phase3_additions_are_balanced_and_prior_disjoint() -> None:
    examples = build_phase3_additions()
    assert len(examples) == 9 * (50 + 10 + 10 + 10)
    prior = prior_canonical_pairs()
    by_split = {}
    for split, expected in (
        ("fit", 450),
        ("selection", 90),
        ("development", 90),
        ("audit", 90),
    ):
        selected = [example for example in examples if example.split == split]
        assert len(selected) == expected
        by_split[split] = canonical_pairs(selected)
        assert not by_split[split] & prior
        counts = {
            digit: sum(str(example.result)[0] == str(digit) for example in selected)
            for digit in range(1, 10)
        }
        assert set(counts.values()) == {expected // 9}
    for index, left in enumerate(sorted(by_split)):
        for right in sorted(by_split)[index + 1 :]:
            assert not by_split[left] & by_split[right]


def test_phase3_dataset_is_deterministic() -> None:
    assert phase3_addition_sha256(
        build_phase3_additions()
    ) == phase3_addition_sha256(build_phase3_additions())
