from open_latent_interfaces.capability import build_capability_sweep
from open_latent_interfaces.phase1_data import (
    build_phase1_additions,
    phase1_addition_sha256,
)


def test_phase1_additions_are_balanced_and_capability_disjoint() -> None:
    examples = build_phase1_additions()
    assert len(examples) == 9 * (10 + 5 + 5)
    capability_pairs = {
        tuple(sorted((example.operand_a, example.operand_b)))
        for example in build_capability_sweep(protocol_version="v2")
    }
    for split, expected in (("train", 90), ("development", 45), ("audit", 45)):
        selected = [example for example in examples if example.split == split]
        assert len(selected) == expected
        assert not {
            tuple(sorted((example.operand_a, example.operand_b)))
            for example in selected
        } & capability_pairs


def test_phase1_dataset_is_deterministic() -> None:
    assert phase1_addition_sha256(
        build_phase1_additions()
    ) == phase1_addition_sha256(build_phase1_additions())
