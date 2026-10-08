import numpy as np

from plawvla.policies import libero_policy


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


def test_command_to_libero_absolute_actions_accepts_padded_canonical_state() -> None:
    actions = np.array([[0.5, -0.5, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
    state = np.zeros((32,), dtype=np.float32)
    state[:8] = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    result = libero_policy._command_to_libero_absolute_actions(actions, state)

    np.testing.assert_allclose(
        result,
        np.array([[1.025, 1.975, 3.05, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_command_to_libero_absolute_actions_interprets_negative_sign_as_open() -> None:
    actions = np.array([[0.5, -0.5, 1.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)
    state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    result = libero_policy._command_to_libero_absolute_actions(actions, state)

    np.testing.assert_allclose(
        result,
        np.array([[1.025, 1.975, 3.05, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
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
                [1.025, 1.975, 3.05, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.025, 1.975, 3.05, 1.0, 0.0, 0.0, 0.0, 0.0],
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


def test_libero_inputs_combines_singleton_validity_and_padding_masks() -> None:
    transform = libero_policy.LiberoInputs(
        enable_world_model=False,
        canonicalize_ee_pose_gripper=True,
        state_input_format="canonical",
        dataset_state_gripper_format="physical_width",
        dataset_action_gripper_format="absolute_physical_width",
    )
    state = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04], dtype=np.float32)
    actions = np.repeat(state[None], 3, axis=0)

    result = transform(
        {
            "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "observation/state": state,
            "actions": actions,
            "actions_is_pad": np.array([False, False, True]),
            "action_validity": np.array([[True], [False], [True]]),
        }
    )

    np.testing.assert_array_equal(result["action_loss_mask"], [True, False, False])
