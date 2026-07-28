from pathlib import Path

import pytest
import torch

from open_latent_interfaces.arithmetic_coordinates import (
    ArithmeticCoordinateManifest,
    TokenLocalTransportWriter,
)


def test_fixed_transport_writer_repeats_scaled_vector() -> None:
    writer = TokenLocalTransportWriter(torch.tensor([1.0, 2.0]), scale=1.5)
    delta = writer.delta(batch_size=2)
    assert delta.shape == (2, 2)
    assert torch.allclose(delta, torch.tensor([[1.5, 3.0], [1.5, 3.0]]))


def test_class_conditioned_writer_uses_explicit_labels() -> None:
    writer = TokenLocalTransportWriter(
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        scale=2.0,
        class_labels=torch.tensor([3, 7]),
    )
    delta = writer.delta(class_labels=torch.tensor([7, 3]))
    assert torch.equal(delta, torch.tensor([[0.0, 4.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="unsupported transport class"):
        writer.delta(class_labels=torch.tensor([5]))


def test_audited_arithmetic_manifest_and_writers() -> None:
    root = Path(__file__).parents[1]
    manifest = ArithmeticCoordinateManifest.load(
        root / "manifests/phi35-mini-arithmetic-coordinates-v1.json"
    )
    manifest.verify(root)
    assert manifest.model_id == "microsoft/Phi-3.5-mini-instruct"
    assert set(manifest.interfaces) == {"operand_increment", "carry_on"}
    operand = manifest.load_writer("operand_increment", root=root)
    carry = manifest.load_writer("carry_on", root=root)
    assert (operand.coordinate_count, operand.residual_width, operand.scale) == (
        4,
        3072,
        1.5,
    )
    assert (carry.coordinate_count, carry.residual_width, carry.scale) == (
        1,
        3072,
        1.0,
    )
