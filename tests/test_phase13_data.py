from __future__ import annotations

from open_latent_interfaces.phase9e_data import PHASE9E_TEMPLATES
from open_latent_interfaces.phase11_audit_data import PHASE11_AUDIT_TEMPLATES
from open_latent_interfaces.phase12_audit_data import PHASE12_AUDIT_TEMPLATES
from open_latent_interfaces.phase12_data import PHASE12_TEMPLATES
from open_latent_interfaces.phase13_data import (
    PHASE13_TEMPLATES,
    build_phase13_examples,
    phase13_sha256,
    prior_canonical_pairs,
    prior_dataset_hashes,
)


def test_phase13_is_deterministic_balanced_and_split() -> None:
    first = build_phase13_examples()
    second = build_phase13_examples()
    assert first == second
    assert phase13_sha256(first) == phase13_sha256(second)
    assert len(first) == 270
    for split in ("fit", "selection", "development"):
        rows = [row for row in first if row.split == split]
        assert len(rows) == 90
        assert {row.leading_digit for row in rows} == set(range(1, 10))
        assert {row.tens_digit for row in rows} == set(range(10))
        assert {row.ones_digit for row in rows} == set(range(10))
        assert sum(row.ones_carry for row in rows) == 45


def test_phase13_pairs_are_disjoint_from_prior_and_between_splits() -> None:
    rows = build_phase13_examples()
    prior = prior_canonical_pairs()
    by_split = {
        split: {
            tuple(sorted((row.operand_a, row.operand_b)))
            for row in rows
            if row.split == split
        }
        for split in ("fit", "selection", "development")
    }
    assert all(not pairs & prior for pairs in by_split.values())
    assert not by_split["fit"] & by_split["selection"]
    assert not by_split["fit"] & by_split["development"]
    assert not by_split["selection"] & by_split["development"]
    assert set(prior_dataset_hashes()) == {
        "phase3",
        "phase4",
        "phase6",
        "phase7",
        "phase9e",
        "phase11",
        "phase12",
        "phase12_audit",
    }


def test_phase13_templates_are_new_and_split_disjoint() -> None:
    historical = (
        set(PHASE9E_TEMPLATES)
        | set(PHASE11_AUDIT_TEMPLATES)
        | set(PHASE12_TEMPLATES["selection"])
        | set(PHASE12_TEMPLATES["development"])
        | set(PHASE12_AUDIT_TEMPLATES)
    )
    split_templates = {
        split: set(PHASE13_TEMPLATES[split])
        for split in ("fit", "selection", "development")
    }
    assert all(len(rows) == 3 for rows in split_templates.values())
    assert all(not rows & historical for rows in split_templates.values())
    assert not split_templates["fit"] & split_templates["selection"]
    assert not split_templates["fit"] & split_templates["development"]
    assert not split_templates["selection"] & split_templates["development"]
