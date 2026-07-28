from pathlib import Path

import pytest
import torch

from open_latent_interfaces.operand_reader import (
    NearestCentroidDigitReader,
    OperandReaderManifest,
    fit_nearest_centroid_digit_reader,
    reconstruct_decimal_digits,
)


def test_nearest_centroid_reader_decodes_fitted_classes() -> None:
    states = torch.tensor(
        [
            [1.0, 0.0],
            [1.2, 0.1],
            [0.0, 1.0],
            [0.1, 1.2],
        ]
    )
    digits = torch.tensor([3, 3, 7, 7])
    reader, counts = fit_nearest_centroid_digit_reader(
        states,
        digits,
        classes=(3, 7),
    )
    assert counts.tolist() == [2, 2]
    assert reader.predict(torch.tensor([[0.9, 0.0], [0.0, 0.9]])).tolist() == [
        3,
        7,
    ]


def test_reader_validates_width_and_decimal_reconstruction() -> None:
    reader = NearestCentroidDigitReader(
        classes=torch.tensor([0, 1]),
        centroids=torch.eye(2),
    )
    with pytest.raises(ValueError, match="residual width"):
        reader.predict(torch.ones(1, 3))
    assert reconstruct_decimal_digits([4, 0, 7]) == 407
    with pytest.raises(ValueError, match="at least one"):
        reconstruct_decimal_digits([])


def test_audited_phi_operand_reader_manifest() -> None:
    root = Path(__file__).parents[1]
    manifest = OperandReaderManifest.load(
        root / "manifests/phi35-mini-operand-reader-v1.json"
    )
    manifest.verify(root)
    reader = manifest.load_reader(root)
    assert manifest.hidden_state_index == 1
    assert reader.classes.tolist() == list(range(10))
    assert reader.centroids.shape == (10, 3072)


def test_audited_qwen_operand_reader_manifest() -> None:
    root = Path(__file__).parents[1]
    manifest = OperandReaderManifest.load(
        root / "manifests/qwen25-15b-operand-reader-v1.json"
    )
    manifest.verify(root)
    reader = manifest.load_reader(root)
    assert manifest.schema_version == "oli.operand-reader-interface/v2"
    assert manifest.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert manifest.hidden_state_index == 1
    assert reader.classes.tolist() == list(range(10))
    assert reader.centroids.shape == (10, 1536)
