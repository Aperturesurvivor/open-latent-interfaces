from open_latent_interfaces.phase4_data import build_phase4_carry_quartets
from open_latent_interfaces.phase6_data import (
    REFERENCE_PARAMETERS,
    build_phase6_carry_quartets,
)
from open_latent_interfaces.phase7_data import (
    assert_phase7_carry_invariants,
    build_phase7_carry_quartets,
    phase7_carry_sha256,
)


def canonical_pairs(examples: list[object]) -> set[tuple[int, int]]:
    return {
        tuple(sorted((row.operand_a, row.operand_b)))
        for row in examples
    }


def test_phase7_dataset_is_deterministic_and_historically_disjoint() -> None:
    first = build_phase7_carry_quartets()
    second = build_phase7_carry_quartets()
    historical = (
        build_phase4_carry_quartets(**REFERENCE_PARAMETERS)
        + build_phase6_carry_quartets()
    )
    assert first == second
    assert len(first) == 1260
    assert len({row.quartet_id for row in first}) == 315
    assert phase7_carry_sha256(first) == phase7_carry_sha256(second)
    assert not (canonical_pairs(first) & canonical_pairs(historical))
    assert all(100 <= row.result <= 999 for row in first)
    assert all(0 < row.operand_a < 1000 for row in first)
    assert all(0 < row.operand_b < 1000 for row in first)
    assert_phase7_carry_invariants(first)


def test_phase7_split_pairs_are_mutually_disjoint() -> None:
    examples = build_phase7_carry_quartets()
    split_pairs = {
        split: canonical_pairs([row for row in examples if row.split == split])
        for split in ("fit", "selection", "development", "audit")
    }
    names = list(split_pairs)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert not (split_pairs[left] & split_pairs[right])
