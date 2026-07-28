from open_latent_interfaces.capability import (
    CAPABILITY_TEMPLATES,
    build_capability_sweep,
    capability_dataset_sha256,
    parse_first_integer,
)


def test_capability_sweep_is_balanced_and_split_disjoint() -> None:
    examples = build_capability_sweep()
    assert len(examples) == 5 * (12 + 8) * len(CAPABILITY_TEMPLATES) * 2
    for regime in {example.regime for example in examples}:
        for split, pair_count in (("development", 12), ("audit", 8)):
            subset = [
                example
                for example in examples
                if example.regime == regime and example.split == split
            ]
            pairs = {
                tuple(sorted((example.operand_a, example.operand_b)))
                for example in subset
            }
            assert len(pairs) == pair_count


def test_capability_dataset_is_deterministic() -> None:
    assert capability_dataset_sha256(
        build_capability_sweep()
    ) == capability_dataset_sha256(build_capability_sweep())


def test_parse_first_integer() -> None:
    assert parse_first_integer("42") == 42
    assert parse_first_integer("The result is 1,024.") == 1024
    assert parse_first_integer("No numeric answer") is None
