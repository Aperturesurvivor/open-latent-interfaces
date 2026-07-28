import pytest
import torch

from open_latent_interfaces.typed_writer import (
    build_conditional_transport_design,
    build_full_result_transport_design,
    encode_three_digit_results,
    fit_digit_subspace,
    fit_transport_subspace,
)


def test_digit_writer_replaces_coordinates_inside_fitted_subspace() -> None:
    states = torch.tensor(
        [
            [-2.0, 0.0, 4.0],
            [-1.0, 0.0, 6.0],
            [1.0, 0.0, 3.0],
            [2.0, 0.0, 7.0],
        ]
    )
    digits = torch.tensor([0, 0, 1, 1])
    writer = fit_digit_subspace(states, digits)
    recipients = torch.tensor([[-3.0, 2.0, 10.0], [3.0, -2.0, -10.0]])
    delta = writer.write_delta(
        recipients,
        torch.tensor([1, 0]),
        rank=1,
        scale=1.0,
    )
    written = recipients + delta
    assert written[:, 0].tolist() == pytest.approx([1.5, -1.5], abs=1e-5)
    assert torch.allclose(written[:, 1:], recipients[:, 1:], atol=1e-5)


def test_digit_writer_rejects_unfitted_digit() -> None:
    writer = fit_digit_subspace(
        torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([2, 3]),
    )
    with pytest.raises(ValueError, match="was not fitted"):
        writer.write_delta(
            torch.zeros(1, 2),
            torch.tensor([4]),
            rank=1,
            scale=1.0,
        )


def test_transport_writer_returns_projected_class_delta() -> None:
    deltas = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
    )
    writer = fit_transport_subspace(deltas, torch.tensor([1, 1, 2, 2]))
    written = writer.write_delta(
        torch.tensor([1, 2]),
        rank=2,
        scale=0.5,
    )
    assert torch.allclose(
        written,
        torch.tensor([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]),
        atol=1e-5,
    )


def test_conditional_transport_learns_state_digit_interaction() -> None:
    values = torch.linspace(-3.0, 3.0, 20)
    states = torch.stack((values, torch.ones_like(values)), dim=1)
    digits = torch.tensor([0, 1] * 10)
    deltas = torch.stack(
        (
            torch.where(digits == 0, 2.0 * values, -3.0 * values),
            digits.float(),
        ),
        dim=1,
    )
    design = build_conditional_transport_design(
        states,
        deltas,
        digits,
        state_rank=1,
        max_transport_rank=2,
    )
    model = design.fit(transport_rank=2, ridge=1e-5)
    predicted = model.predict(states, digits)
    assert torch.allclose(predicted, deltas, atol=1e-3)


def test_three_digit_result_encoding_is_position_specific() -> None:
    encoded = encode_three_digit_results(torch.tensor([105, 150]))
    assert encoded.shape == (2, 29)
    assert encoded.sum(dim=1).tolist() == [3.0, 3.0]
    assert not torch.equal(encoded[0], encoded[1])
    with pytest.raises(ValueError, match="three-digit"):
        encode_three_digit_results(torch.tensor([99]))


def test_full_result_transport_learns_state_result_interaction() -> None:
    values = torch.linspace(-2.0, 2.0, 90)
    states = torch.stack((values, torch.ones_like(values)), dim=1)
    hundreds = torch.tensor([(index % 9) + 1 for index in range(90)])
    tens = torch.tensor([(index * 3) % 10 for index in range(90)])
    ones = torch.tensor([(index * 7) % 10 for index in range(90)])
    results = hundreds * 100 + tens * 10 + ones
    deltas = torch.stack(
        (
            values * hundreds + tens,
            values * ones + hundreds,
        ),
        dim=1,
    )
    design = build_full_result_transport_design(
        states,
        deltas,
        results,
        state_rank=1,
        max_transport_rank=2,
    )
    model = design.fit(transport_rank=2, ridge=1e-5)
    predicted = model.predict(states, results)
    assert torch.allclose(predicted, deltas, atol=2e-3)
