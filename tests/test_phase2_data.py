from open_latent_interfaces.capability import build_capability_sweep
from open_latent_interfaces.phase1_data import build_phase1_additions
from open_latent_interfaces.phase2_data import (
    balanced_counterfactual_results,
    build_phase2_additions,
    phase2_addition_sha256,
)


def canonical_pairs(examples: list[object]) -> set[tuple[int, int]]:
    return {
        tuple(sorted((example.operand_a, example.operand_b)))  # type: ignore[attr-defined]
        for example in examples
    }


def test_phase2_additions_are_balanced_and_prior_disjoint() -> None:
    examples = build_phase2_additions()
    assert len(examples) == 9 * (50 + 10 + 10 + 10)
    prior_pairs = canonical_pairs(
        build_capability_sweep(protocol_version="v2")
    ) | canonical_pairs(build_phase1_additions())
    split_pairs = {}
    for split, expected in (
        ("fit", 450),
        ("selection", 90),
        ("development", 90),
        ("audit", 90),
    ):
        selected = [example for example in examples if example.split == split]
        assert len(selected) == expected
        split_pairs[split] = canonical_pairs(selected)
        assert not split_pairs[split] & prior_pairs
        counts = {
            digit: sum(str(example.result)[0] == str(digit) for example in selected)
            for digit in range(1, 10)
        }
        assert set(counts.values()) == {expected // 9}
    assert not split_pairs["fit"] & split_pairs["selection"]
    assert not split_pairs["fit"] & split_pairs["development"]
    assert not split_pairs["fit"] & split_pairs["audit"]
    assert not split_pairs["selection"] & split_pairs["development"]
    assert not split_pairs["selection"] & split_pairs["audit"]
    assert not split_pairs["development"] & split_pairs["audit"]


def test_phase2_dataset_is_deterministic() -> None:
    assert phase2_addition_sha256(
        build_phase2_additions()
    ) == phase2_addition_sha256(build_phase2_additions())


def test_balanced_counterfactual_results_change_every_digit() -> None:
    examples = build_phase2_additions()
    for split in ("fit", "selection", "development", "audit"):
        selected = [example for example in examples if example.split == split]
        targets = balanced_counterfactual_results(selected)
        assert all(
            all(
                left != right
                for left, right in zip(
                    str(example.result), str(target), strict=True
                )
            )
            for example, target in zip(selected, targets, strict=True)
        )
        assert {len(str(target)) for target in targets} == {3}
        for position, digits in enumerate((range(1, 10), range(10), range(10))):
            counts = {
                digit: sum(str(target)[position] == str(digit) for target in targets)
                for digit in digits
            }
            assert set(counts.values()) == {len(selected) // len(counts)}


def test_balanced_counterfactual_results_are_order_invariant() -> None:
    examples = [
        example
        for example in build_phase2_additions()
        if example.split == "development"
    ]
    forward = dict(
        zip(
            (example.example_id for example in examples),
            balanced_counterfactual_results(examples),
            strict=True,
        )
    )
    reversed_examples = list(reversed(examples))
    backward = dict(
        zip(
            (example.example_id for example in reversed_examples),
            balanced_counterfactual_results(reversed_examples),
            strict=True,
        )
    )
    assert forward == backward
