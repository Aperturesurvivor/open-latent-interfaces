from __future__ import annotations

from open_latent_interfaces.phase9e_data import PHASE9E_TEMPLATES
from open_latent_interfaces.phase11_audit_data import PHASE11_AUDIT_TEMPLATES
from open_latent_interfaces.phase12_audit_data import (
    PHASE12_AUDIT_TEMPLATES,
    build_phase12_audit,
    phase12_audit_sha256,
    prior_canonical_pairs,
    prior_dataset_hashes,
)
from open_latent_interfaces.phase12_data import PHASE12_TEMPLATES


def test_phase12_audit_is_deterministic_and_balanced() -> None:
    first = build_phase12_audit()
    second = build_phase12_audit()
    assert first == second
    assert phase12_audit_sha256(first) == phase12_audit_sha256(second)
    assert len(first) == 90
    assert {row.leading_digit for row in first} == set(range(1, 10))
    assert {row.tens_digit for row in first} == set(range(10))
    assert {row.ones_digit for row in first} == set(range(10))
    assert sum(row.ones_carry for row in first) == 45


def test_phase12_audit_pairs_are_disjoint_from_every_prior_source() -> None:
    prior = prior_canonical_pairs()
    audit = {
        tuple(sorted((row.operand_a, row.operand_b)))
        for row in build_phase12_audit()
    }
    assert not audit & prior
    assert len(audit) == 90
    assert set(prior_dataset_hashes()) == {
        "phase3",
        "phase4",
        "phase6",
        "phase7",
        "phase9e",
        "phase11",
        "phase12",
    }


def test_phase12_audit_uses_only_new_template_families() -> None:
    examples = build_phase12_audit()
    historical = (
        set(PHASE9E_TEMPLATES)
        | set(PHASE11_AUDIT_TEMPLATES)
        | set(PHASE12_TEMPLATES["selection"])
        | set(PHASE12_TEMPLATES["development"])
    )
    assert len(PHASE12_AUDIT_TEMPLATES) == 3
    assert not set(PHASE12_AUDIT_TEMPLATES) & historical
    assert all(
        row.template_family.startswith("phase12-audit-") for row in examples
    )
    assert all("Answer=<integer>" in row.prompt for row in examples)
