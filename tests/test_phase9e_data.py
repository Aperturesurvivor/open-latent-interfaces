from __future__ import annotations

from open_latent_interfaces.phase9e_data import (
    PHASE9E_TEMPLATES,
    build_phase9e_audit,
    phase9e_audit_sha256,
    prior_canonical_pairs,
    prior_dataset_hashes,
)


def test_phase9e_is_deterministic_and_balanced() -> None:
    first = build_phase9e_audit()
    second = build_phase9e_audit()
    assert first == second
    assert phase9e_audit_sha256(first) == phase9e_audit_sha256(second)
    assert len(first) == 90
    assert {row.leading_digit for row in first} == set(range(1, 10))
    assert {row.tens_digit for row in first} == set(range(10))
    assert {row.ones_digit for row in first} == set(range(10))
    assert sum(row.ones_carry for row in first) == 45


def test_phase9e_pairs_are_disjoint_from_all_prior_sources() -> None:
    prior = prior_canonical_pairs()
    audit = {
        tuple(sorted((row.operand_a, row.operand_b)))
        for row in build_phase9e_audit()
    }
    assert not audit & prior
    assert len(audit) == 90
    assert set(prior_dataset_hashes()) == {
        "phase3",
        "phase4",
        "phase6",
        "phase7",
    }


def test_phase9e_uses_only_new_template_families() -> None:
    examples = build_phase9e_audit()
    assert len(PHASE9E_TEMPLATES) == 3
    assert all(row.template_family.startswith("phase9e-audit-") for row in examples)
    assert all("Answer=<integer>" in row.prompt for row in examples)
