import pytest
import torch

from open_latent_interfaces.operand_reader import (
    NearestCentroidDigitReader,
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
