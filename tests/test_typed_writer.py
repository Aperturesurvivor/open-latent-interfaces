import pytest
import torch

from open_latent_interfaces.typed_writer import (
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
