from pathlib import Path

import pytest

from open_latent_interfaces.model_onboarding import (
    ModelOnboardingSpec,
    candidate_hidden_state_indices,
)


@pytest.mark.parametrize(
    ("config_name", "model_id", "layers"),
    (
        (
            "model_onboarding_phi35_mini_audited.json",
            "microsoft/Phi-3.5-mini-instruct",
            32,
        ),
        (
            "model_onboarding_qwen25_15b_audited.json",
            "Qwen/Qwen2.5-1.5B-Instruct",
            28,
        ),
    ),
)
def test_audited_model_onboarding_specs_bind_evidence(
    config_name: str,
    model_id: str,
    layers: int,
) -> None:
    root = Path(__file__).parents[1]
    spec = ModelOnboardingSpec.load(root / "configs" / config_name)
    spec.verify_evidence(root)
    assert spec.model_id == model_id
    assert spec.expected_num_hidden_layers == layers
    assert len(candidate_hidden_state_indices(layers)) == 8


def test_candidate_hidden_state_indices_are_bounded_and_deterministic() -> None:
    first = candidate_hidden_state_indices(28)
    second = candidate_hidden_state_indices(28)
    assert first == second
    assert first[0] == 1
    assert all(1 <= index <= 28 for index in first)
    assert first == sorted(set(first))


def test_model_onboarding_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        ModelOnboardingSpec.from_dict({"schema_version": "wrong"})
