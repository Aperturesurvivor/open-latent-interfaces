from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_phase13_smollm2_suffix_prototype_selection import (  # noqa: E402
    fit_digit_subspace,
    prototype_delta,
    rotate_position,
)


def test_rotate_position_changes_only_requested_suffix_digit() -> None:
    assert rotate_position([109, 987], position=1) == [119, 997]
    assert rotate_position([109, 987], position=2) == [100, 988]


def test_fit_digit_subspace_is_orthonormal_and_counted() -> None:
    states = torch.eye(10).repeat_interleave(2, dim=0)
    labels = torch.arange(10).repeat_interleave(2)
    basis, centroids, counts = fit_digit_subspace(states, labels)
    assert basis.shape == (9, 10)
    assert centroids.shape == (10, 10)
    assert counts.tolist() == [2] * 10
    torch.testing.assert_close(basis @ basis.T, torch.eye(9))


def test_prototype_delta_reaches_projected_centroid() -> None:
    states = torch.eye(10)
    labels = torch.arange(10)
    basis, centroids, _ = fit_digit_subspace(states, labels)
    desired = torch.arange(10).roll(1)
    delta = prototype_delta(states, desired, centroids, basis)
    torch.testing.assert_close(
        (states + delta) @ basis.T,
        centroids[desired] @ basis.T,
        atol=1e-6,
        rtol=1e-6,
    )
