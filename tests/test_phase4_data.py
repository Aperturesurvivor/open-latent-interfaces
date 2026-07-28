from open_latent_interfaces.phase3_data import prior_canonical_pairs
from open_latent_interfaces.phase4_data import (
    build_phase4_carry_quartets,
    phase4_carry_sha256,
)


def test_phase4_quartets_are_matched_balanced_and_prior_disjoint() -> None:
    examples = build_phase4_carry_quartets()
    assert len(examples) == 4 * 9 * (20 + 5 + 5 + 5)
    prior = prior_canonical_pairs()
    observed = set()
    for example in examples:
        canonical = tuple(sorted((example.operand_a, example.operand_b)))
        assert canonical not in prior
        assert canonical not in observed
        observed.add(canonical)
    for split, expected_quartets in (
        ("fit", 180),
        ("selection", 45),
        ("development", 45),
        ("audit", 45),
    ):
        carry_base = [
            example
            for example in examples
            if example.split == split and example.variant == "carry_base"
        ]
        assert len(carry_base) == expected_quartets
        counts = {
            digit: sum(int(str(example.result)[0]) == digit for example in carry_base)
            for digit in range(1, 10)
        }
        assert set(counts.values()) == {expected_quartets // 9}


def test_phase4_dataset_is_deterministic() -> None:
    assert phase4_carry_sha256(
        build_phase4_carry_quartets()
    ) == phase4_carry_sha256(build_phase4_carry_quartets())
