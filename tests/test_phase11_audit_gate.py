from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

audit_gate = import_module("run_phase11_qwen_hybrid_graft_audit").audit_gate


RULE = {
    "minimum_reader_pair_accuracy": 0.98,
    "minimum_computed_target_accuracy": 0.98,
    "minimum_final_exact_accuracy": 0.95,
    "maximum_oracle_exact_gap": 0.03,
    "minimum_position_accuracy": 0.95,
    "minimum_base_correct_preservation": 0.98,
    "minimum_net_improvement_over_base": 0.0,
    "minimum_base_error_recovery": 0.8,
    "minimum_excess_base_error_recovery_over_random": 0.5,
    "minimum_excess_base_error_recovery_over_wrong_target": 0.5,
    "minimum_shuffled_target_accuracy": 0.9,
    "maximum_shuffled_true_accuracy": 0.15,
    "maximum_shuffled_random_target_accuracy": 0.15,
    "minimum_shuffled_target_advantage_over_random": 0.7,
    "require_parse_rate": True,
    "require_digit_token_rate": True,
}


def condition(
    *,
    true_accuracy: float = 1.0,
    target_accuracy: float = 1.0,
) -> dict[str, object]:
    return {
        "true_result_accuracy": true_accuracy,
        "target_full_result_accuracy": target_accuracy,
        "step_target_accuracy": [1.0, 1.0, 1.0],
        "parse_rate": 1.0,
        "digit_token_rate": 1.0,
        "mean_relative_norm_by_step": [0.1, 0.2, 0.2],
    }


def paired(
    *,
    errors: int,
    latent_recovery: float,
    random_recovery: float,
    wrong_recovery: float,
) -> dict[str, dict[str, object]]:
    def row(recovery: float, improvement: float) -> dict[str, object]:
        return {
            "base_error_count": errors,
            "base_correct_count": 90 - errors,
            "base_error_recovery": recovery,
            "base_correct_preservation": 1.0,
            "net_exact_improvement_rate": improvement,
        }

    return {
        "latent_read_compute_hybrid_write": row(
            latent_recovery,
            0.1 if errors else 0.0,
        ),
        "random_norm_matched": row(random_recovery, 0.0),
        "wrong_target_norm_matched": row(wrong_recovery, 0.0),
    }


def passing_conditions() -> dict[str, dict[str, object]]:
    return {
        "latent_read_compute_hybrid_write": condition(),
        "oracle_compute_hybrid_write": condition(),
        "shuffled_read_compute_hybrid_write": condition(
            true_accuracy=0.05,
            target_accuracy=0.98,
        ),
        "shuffled_random_norm_matched": condition(
            true_accuracy=0.4,
            target_accuracy=0.05,
        ),
    }


def test_phase11_audit_gate_uses_both_recovery_controls() -> None:
    result = audit_gate(
        passing_conditions(),
        paired(
            errors=10,
            latent_recovery=1.0,
            random_recovery=0.1,
            wrong_recovery=0.3,
        ),
        reader_pair_accuracy=1.0,
        computed_accuracy=1.0,
        rule=RULE,
        leading_norm_cap=0.25,
        suffix_norm_cap=1.0,
    )
    assert result["recovery_branch"] == "observed_base_errors"
    assert result["checks"]["excess_recovery_over_random"]
    assert result["checks"]["excess_recovery_over_wrong"]
    assert result["passes"]


def test_phase11_audit_gate_rejects_random_target_following() -> None:
    conditions = passing_conditions()
    conditions["shuffled_random_norm_matched"] = condition(
        true_accuracy=0.1,
        target_accuracy=0.4,
    )
    result = audit_gate(
        conditions,
        paired(
            errors=10,
            latent_recovery=1.0,
            random_recovery=0.1,
            wrong_recovery=0.3,
        ),
        reader_pair_accuracy=1.0,
        computed_accuracy=1.0,
        rule=RULE,
        leading_norm_cap=0.25,
        suffix_norm_cap=1.0,
    )
    assert not result["checks"]["shuffled_random_target_control"]
    assert not result["passes"]


def test_phase11_audit_gate_zero_error_branch_requires_no_harm() -> None:
    result = audit_gate(
        passing_conditions(),
        paired(
            errors=0,
            latent_recovery=1.0,
            random_recovery=1.0,
            wrong_recovery=1.0,
        ),
        reader_pair_accuracy=1.0,
        computed_accuracy=1.0,
        rule=RULE,
        leading_norm_cap=0.25,
        suffix_norm_cap=1.0,
    )
    assert result["recovery_branch"] == "zero_base_errors"
    assert result["checks"]["zero_error_full_exact"]
    assert result["checks"]["zero_error_no_harm"]
    assert result["passes"]
