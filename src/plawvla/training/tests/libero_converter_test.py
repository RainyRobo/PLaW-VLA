# Copyright 2026 PLaW-VLA authors. SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.data.libero.convert_libero_to_lerobot import build_absolute_next_targets


def _state(num_frames: int) -> np.ndarray:
    return np.arange(num_frames * 8, dtype=np.float32).reshape(num_frames, 8)


def test_build_absolute_next_targets_stride_one() -> None:
    state = _state(4)

    targets, validity = build_absolute_next_targets(state, stride=1)

    np.testing.assert_array_equal(targets[:-1], state[1:])
    np.testing.assert_array_equal(targets[-1], state[-1])
    np.testing.assert_array_equal(validity, [True, True, True, False])
    assert targets.dtype == state.dtype
    assert validity.dtype == np.bool_


def test_build_absolute_next_targets_stride_greater_than_one() -> None:
    state = _state(7)

    targets, validity = build_absolute_next_targets(state, stride=3)

    sampled_state = state[[0, 3, 6]]
    np.testing.assert_array_equal(targets, sampled_state[[1, 2, 2]])
    np.testing.assert_array_equal(validity, [True, True, False])


def test_build_absolute_next_targets_single_sample_is_invalid_and_unchanged() -> None:
    state = _state(2)

    targets, validity = build_absolute_next_targets(state, stride=10)

    np.testing.assert_array_equal(targets, state[:1])
    np.testing.assert_array_equal(validity, [False])


@pytest.mark.parametrize(
    ("state", "error", "match"),
    [
        (np.zeros(8, dtype=np.float32), ValueError, r"shape \[num_frames, 8\]"),
        (np.zeros((2, 7), dtype=np.float32), ValueError, r"shape \[num_frames, 8\]"),
        (np.zeros((2, 8, 1), dtype=np.float32), ValueError, r"shape \[num_frames, 8\]"),
        (np.zeros((0, 8), dtype=np.float32), ValueError, "at least one frame"),
        ([[0.0] * 8], TypeError, "numpy.ndarray"),
    ],
)
def test_build_absolute_next_targets_rejects_invalid_state(
    state: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        build_absolute_next_targets(state, stride=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("stride", "error"),
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ],
)
def test_build_absolute_next_targets_rejects_invalid_stride(
    stride: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="stride"):
        build_absolute_next_targets(_state(3), stride)  # type: ignore[arg-type]
