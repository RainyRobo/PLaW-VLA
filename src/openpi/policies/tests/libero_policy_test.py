import numpy as np

from openpi.policies import libero_policy


def test_canonicalize_stored_libero_state_normalizes_physical_gripper() -> None:
    state = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04], dtype=np.float32)

    result = libero_policy._canonicalize_stored_libero_state(state)

    np.testing.assert_allclose(result[:-1], state[:-1])
    np.testing.assert_allclose(result[-1], 1.0)


def test_canonical_ee_deltas_to_absolute_maps_signed_gripper_commands() -> None:
    state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    actions = np.array(
        [
            [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = libero_policy._canonical_ee_deltas_to_absolute(actions, state)

    np.testing.assert_allclose(result[:, 0], [1.1, 1.2])
    np.testing.assert_allclose(result[:, 7], [1.0, 0.0])


def test_absolute_to_libero_command_accepts_padded_canonical_state() -> None:
    actions = np.array([[1.5, 1.5, 4.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    state = np.zeros((32,), dtype=np.float32)
    state[:8] = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    result = libero_policy._absolute_to_libero_command(actions, state)

    np.testing.assert_allclose(
        result,
        np.array([[0.5, -0.5, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32),
    )


def test_absolute_to_libero_command_restores_release_sign() -> None:
    actions = np.array([[1.5, 1.5, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    result = libero_policy._absolute_to_libero_command(actions, state)

    np.testing.assert_allclose(
        result,
        np.array([[0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_command_to_libero_absolute_actions_accepts_padded_canonical_state() -> None:
    actions = np.array([[0.5, -0.5, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
    state = np.zeros((32,), dtype=np.float32)
    state[:8] = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    result = libero_policy._command_to_libero_absolute_actions(actions, state)

    np.testing.assert_allclose(
        result,
        np.array([[1.5, 1.5, 4.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_command_round_trip_preserves_open_and_close_gripper_signs() -> None:
    actions = np.array(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    raw_command_actions = np.array(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    absolute_actions = libero_policy._command_to_libero_absolute_actions(raw_command_actions, state)
    round_trip = libero_policy._absolute_to_libero_command(absolute_actions, state)

    np.testing.assert_allclose(round_trip, actions, atol=1e-6)


def test_command_to_libero_absolute_actions_interprets_negative_sign_as_open() -> None:
    actions = np.array([[0.5, -0.5, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
    state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    result = libero_policy._command_to_libero_absolute_actions(actions, state)

    np.testing.assert_allclose(
        result,
        np.array([[1.5, 1.5, 4.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_command_to_libero_absolute_actions_supports_binary_gripper_targets() -> None:
    actions = np.array(
        [
            [0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    result = libero_policy._command_to_libero_absolute_actions(
        actions,
        state,
        gripper_actions_are_binary_targets=True,
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1.5, 1.5, 4.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.5, 1.5, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_command_to_libero_absolute_actions_rejects_non_binary_gripper_targets() -> None:
    actions = np.array([[0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 2.0]], dtype=np.float32)
    state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, r"Expected LIBERO gripper target values in \[0, 1\]"):
        libero_policy._command_to_libero_absolute_actions(
            actions,
            state,
            gripper_actions_are_binary_targets=True,
        )
