import importlib.util
import pathlib
import sys

import numpy as np


_MAIN_PATH = pathlib.Path(__file__).with_name("main.py")
_SPEC = importlib.util.spec_from_file_location("libero_eval_main", _MAIN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)


def test_legacy_gripper_state_for_policy_cancels_server_openness_scaling() -> None:
    raw_state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04], dtype=np.float32)

    compat_state = main._legacy_gripper_state_for_policy(raw_state)

    np.testing.assert_allclose(compat_state[:-2], raw_state[:-2])
    np.testing.assert_allclose(compat_state[-2:], [0.0016, -0.0016])
    # The server collapses the opposing fingers and divides by 0.04.  The
    # resulting scalar therefore matches the legacy dataset's 0.04-metre state.
    server_scalar = 0.5 * (compat_state[-2] - compat_state[-1]) / 0.04
    np.testing.assert_allclose(server_scalar, 0.04)


def test_legacy_gripper_actions_restore_robosuite_command_sign() -> None:
    decoded = np.zeros((2, 7), dtype=np.float32)
    decoded[:, 6] = [1.0, -1.0]

    corrected = main._legacy_gripper_actions_for_env(decoded)

    np.testing.assert_array_equal(corrected[:, 6], [-1.0, 1.0])
    np.testing.assert_array_equal(decoded[:, 6], [1.0, -1.0])
