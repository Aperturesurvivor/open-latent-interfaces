from __future__ import annotations

from open_latent_interfaces.phase9e_data import PHASE9E_TEMPLATES
from open_latent_interfaces.phase11_audit_data import PHASE11_AUDIT_TEMPLATES
from open_latent_interfaces.phase12_data import (
    PHASE12_TEMPLATES,
    build_phase12_examples,
    phase12_sha256,
    prior_canonical_pairs,
    prior_dataset_hashes,
)


def test_phase12_is_deterministic_balanced_and_split() -> None:
    first = build_phase12_examples()
    second = build_phase12_examples()
    assert first == second
    assert phase12_sha256(first) == phase12_sha256(second)
    assert len(first) == 180
    for split in ("selection", "development"):
        rows = [row for row in first if row.split == split]
        assert len(rows) == 90
        assert {row.leading_digit for row in rows} == set(range(1, 10))
        assert {row.tens_digit for row in rows} == set(range(10))
        assert {row.ones_digit for row in rows} == set(range(10))
        assert sum(row.ones_carry for row in rows) == 45


def test_phase12_pairs_are_disjoint_from_prior_and_between_splits() -> None:
    rows = build_phase12_examples()
    prior = prior_canonical_pairs()
    by_split = {
        split: {
            tuple(sorted((row.operand_a, row.operand_b)))
            for row in rows
            if row.split == split
        }
        for split in ("selection", "development")
    }
    assert not by_split["selection"] & prior
    assert not by_split["development"] & prior
    assert not by_split["selection"] & by_split["development"]
    assert set(prior_dataset_hashes()) == {
        "phase3",
        "phase4",
        "phase6",
        "phase7",
        "phase9e",
        "phase11",
    }


def test_phase12_templates_are_new_and_split_disjoint() -> None:
    historical = set(PHASE9E_TEMPLATES) | set(PHASE11_AUDIT_TEMPLATES)
    selection = set(PHASE12_TEMPLATES["selection"])
    development = set(PHASE12_TEMPLATES["development"])
    assert len(selection) == 3
    assert len(development) == 3
    assert not selection & development
    assert not selection & historical
    assert not development & historical
