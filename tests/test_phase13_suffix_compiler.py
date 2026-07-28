from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_phase13_smollm2_suffix_compiler_selection import (  # noqa: E402
    wrong_position_results,
)


def test_wrong_position_results_rotates_only_requested_digit() -> None:
    assert wrong_position_results([109, 987], position=1) == [119, 997]
    assert wrong_position_results([109, 987], position=2) == [100, 988]


def test_wrong_position_results_rejects_leading_position() -> None:
    with pytest.raises(ValueError, match="suffix control"):
        wrong_position_results([109], position=0)
