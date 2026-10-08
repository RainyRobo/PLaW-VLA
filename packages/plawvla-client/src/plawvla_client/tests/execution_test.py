import numpy as np
import pytest

from plawvla_client.execution import AbsoluteEefTrajectory
from plawvla_client.execution import ExecutionAdapter
from plawvla_client.execution import RobosuiteOSCAdapter
from plawvla_client.execution import TimedHistoryBuffer
from plawvla_client.execution import TimedValueBuffer
from plawvla_client.execution import libero_state_to_canonical
from plawvla_client.execution import normalize_quaternion
from plawvla_client.execution import quaternion_conjugate
from plawvla_client.execution import quaternion_delta
from plawvla_client.execution import quaternion_multiply
from plawvla_client.execution import quaternion_slerp
from plawvla_client.execution import quaternion_to_rotvec
from plawvla_client.execution import rotvec_to_quaternion


def _pose(xyz=(0.0, 0.0, 0.0), quaternion=(1.0, 0.0, 0.0, 0.0), gripper=1.0):
    return np.asarray((*xyz, *quaternion, gripper), dtype=np.float64)


def test_quaternion_operations_use_wxyz_and_normalize():
    quarter_turn_z = rotvec_to_quaternion(np.array([0.0, 0.0, np.pi / 2.0]))
    half_turn_z = quaternion_multiply(quarter_turn_z * 3.0, quarter_turn_z)

    np.testing.assert_allclose(
        quarter_turn_z,
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        quaternion_to_rotvec(half_turn_z),
        [0.0, 0.0, np.pi],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        quaternion_multiply(quarter_turn_z, quaternion_conjugate(quarter_turn_z)),
        [1.0, 0.0, 0.0, 0.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(np.linalg.norm(normalize_quaternion(quarter_turn_z * 9.0)), 1.0)


def test_quaternion_delta_and_slerp_handle_antipodal_inputs():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    antipodal_identity = -identity

    np.testing.assert_allclose(
        quaternion_delta(identity, antipodal_identity),
        identity,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        quaternion_slerp(identity, antipodal_identity, 0.5),
        identity,
        atol=1e-7,
    )

    half_turn_z = rotvec_to_quaternion(np.array([0.0, 0.0, np.pi]))
    midpoint = quaternion_slerp(identity, -half_turn_z, 0.5)
    np.testing.assert_allclose(
        quaternion_to_rotvec(midpoint),
        [0.0, 0.0, np.pi / 2.0],
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "function,value",
    [
        (normalize_quaternion, [0.0, 0.0, 0.0]),
        (normalize_quaternion, [0.0, 0.0, 0.0, 0.0]),
        (normalize_quaternion, [1.0, 0.0, np.nan, 0.0]),
        (rotvec_to_quaternion, [0.0, 0.0]),
        (rotvec_to_quaternion, [0.0, np.inf, 0.0]),
    ],
)
def test_quaternion_shape_and_finite_validation(function, value):
    with pytest.raises(ValueError):
        function(np.asarray(value))


def test_libero_raw_state_to_canonical():
    raw = np.array(
        [
            0.1,
            -0.2,
            0.3,
            0.0,
            0.0,
            np.pi / 2.0,
            0.02,
            -0.02,
        ]
    )
    canonical = libero_state_to_canonical(raw)

    assert canonical.shape == (8,)
    np.testing.assert_allclose(canonical[:3], raw[:3])
    np.testing.assert_allclose(
        canonical[3:7],
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1e-7,
    )
    np.testing.assert_allclose(canonical[7], 0.5)

    clipped_closed = raw.copy()
    clipped_closed[6:8] = [-0.01, 0.01]
    np.testing.assert_allclose(libero_state_to_canonical(clipped_closed)[7], 0.0)


def test_libero_conversion_supports_flattened_bimanual_batches():
    left = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.01, -0.01])
    right = np.array([-1.0, -2.0, -3.0, np.pi, 0.0, 0.0, 0.005, -0.005])
    raw = np.stack([np.concatenate([left, right]), np.concatenate([left, right])])

    canonical = libero_state_to_canonical(raw, arm_count=2)

    assert canonical.shape == (2, 16)
    np.testing.assert_allclose(canonical[:, 7], 0.25)
    np.testing.assert_allclose(canonical[:, 15], 0.125)
    np.testing.assert_allclose(canonical[:, 8:11], np.broadcast_to(right[:3], (2, 3)))


@pytest.mark.parametrize(
    "raw,arm_count",
    [
        (np.zeros(7), None),
        (np.zeros(8), 2),
        (np.full(8, np.nan), None),
    ],
)
def test_libero_conversion_validates_shape_arm_count_and_finiteness(raw, arm_count):
    with pytest.raises(ValueError):
        libero_state_to_canonical(raw, arm_count=arm_count)


def test_absolute_trajectory_10hz_targets_sampled_at_20hz():
    # Policy targets arrive every 0.1 s; execution samples halfway between them.
    times = np.array([0.1, 0.2, 0.3])
    targets = np.stack(
        [
            _pose(xyz=(0.0, 0.0, 0.0), gripper=0.0),
            _pose(
                xyz=(0.1, 0.2, 0.3),
                quaternion=rotvec_to_quaternion([0.0, 0.0, np.pi / 2.0]),
                gripper=0.5,
            ),
            _pose(
                xyz=(0.2, 0.4, 0.6),
                quaternion=rotvec_to_quaternion([0.0, 0.0, np.pi]),
                gripper=1.0,
            ),
        ]
    )
    trajectory = AbsoluteEefTrajectory(
        times,
        targets,
        initial_pose=_pose(xyz=(-0.1, -0.2, -0.3), gripper=0.0),
    )

    sampled = trajectory.sample(0.15)
    np.testing.assert_allclose(sampled[:3], [0.05, 0.1, 0.15], atol=1e-7)
    np.testing.assert_allclose(
        quaternion_to_rotvec(sampled[3:7]),
        [0.0, 0.0, np.pi / 4.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(sampled[7], 0.25)

    sampled = trajectory(0.25)
    np.testing.assert_allclose(sampled[:3], [0.15, 0.3, 0.45], atol=1e-7)
    np.testing.assert_allclose(sampled[7], 0.75)
    np.testing.assert_allclose(
        trajectory.sample(0.05),
        _pose(xyz=(-0.05, -0.1, -0.15), gripper=0.0),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        trajectory.sample(0.0),
        _pose(xyz=(-0.1, -0.2, -0.3), gripper=0.0),
    )
    np.testing.assert_allclose(trajectory.sample(1.0), targets[-1])


def test_absolute_trajectory_normalizes_antipodal_targets():
    targets = np.stack([_pose(), _pose(quaternion=(-1.0, 0.0, 0.0, 0.0))])
    trajectory = AbsoluteEefTrajectory(np.array([0.1, 0.2]), targets)

    sampled = trajectory.sample(0.15)
    np.testing.assert_allclose(sampled[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-7)


@pytest.mark.parametrize(
    "times,targets,arm_count",
    [
        (np.array([0.0, 0.1]), np.zeros((2, 8)), 1),
        (np.array([0.1, 0.1]), np.zeros((2, 8)), 1),
        (np.array([0.2, 0.1]), np.zeros((2, 8)), 1),
        (np.array([0.1, np.nan]), np.zeros((2, 8)), 1),
        (np.array([0.1, 0.2]), np.zeros((2, 7)), 1),
        (np.array([0.1, 0.2]), np.full((2, 8), np.inf), 1),
        (np.array([0.1, 0.2]), np.zeros((2, 8)), 0),
    ],
)
def test_absolute_trajectory_validates_inputs(times, targets, arm_count):
    with pytest.raises(ValueError):
        AbsoluteEefTrajectory(times, targets, arm_count=arm_count)


def test_timed_history_buffer_recovers_real_one_second_history_at_20hz():
    buffer = TimedHistoryBuffer()
    for step in range(21):
        timestamp = step / 20.0
        buffer.append(np.array([step, -step], dtype=np.int64), timestamp)

    values, is_pad = buffer.sample_offsets(
        query_time_s=1.0,
        offsets_s=np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]),
    )

    np.testing.assert_array_equal(values[:, 0], [0, 4, 8, 12, 16, 20])
    np.testing.assert_array_equal(values[:, 1], [0, -4, -8, -12, -16, -20])
    np.testing.assert_array_equal(is_pad, np.zeros(6, dtype=bool))


def test_timed_buffer_causal_sampling_handles_control_period_jitter():
    buffer = TimedValueBuffer(selection="at_or_before")
    timestamps = [0.000, 0.047, 0.103, 0.148, 0.204, 0.251, 0.301]
    for index, timestamp in enumerate(timestamps):
        buffer.append(index, timestamp)

    # Targets are 0.101, 0.201, and 0.301 seconds. Causal sampling must not
    # choose the slightly-late observations at 0.103 or 0.204.
    values, is_pad = buffer.sample_offsets(0.301, [-0.2, -0.1, 0.0])

    np.testing.assert_array_equal(values, [1, 3, 6])
    np.testing.assert_array_equal(is_pad, [False, False, False])


def test_timed_buffer_nearest_sampling_handles_jitter_and_ties_choose_earlier():
    buffer = TimedValueBuffer(selection="nearest")
    for index, timestamp in enumerate([0.000, 0.047, 0.103, 0.148, 0.204, 0.251, 0.301]):
        buffer.append(index, timestamp)

    values, is_pad = buffer.sample_offsets(0.301, [-0.2, -0.1, 0.0])
    np.testing.assert_array_equal(values, [2, 4, 6])
    np.testing.assert_array_equal(is_pad, [False, False, False])

    tie_buffer = TimedValueBuffer(selection="nearest")
    tie_buffer.append("earlier", 1.0)
    tie_buffer.append("later", 1.2)
    tie_values, _ = tie_buffer.sample_offsets(1.1, [0.0])
    np.testing.assert_array_equal(tie_values, ["earlier"])


def test_timed_buffer_episode_start_repeats_earliest_and_marks_padding():
    buffer = TimedHistoryBuffer()
    buffer.append(np.array([10.0, 11.0]), 5.0)
    buffer.append(np.array([20.0, 21.0]), 5.05)

    values, is_pad = buffer.sample_offsets(5.05, [-1.0, -0.05, 0.0])

    np.testing.assert_allclose(values, [[10.0, 11.0], [10.0, 11.0], [20.0, 21.0]])
    np.testing.assert_array_equal(is_pad, [True, False, False])


def test_timed_buffer_copies_appended_values_and_can_clear_for_new_episode():
    buffer = TimedValueBuffer()
    value = np.array([1, 2])
    buffer.append(value, 1.0)
    value[:] = 99

    sampled, _ = buffer.sample_offsets(1.0, [0.0])
    np.testing.assert_array_equal(sampled, [[1, 2]])
    assert len(buffer) == 1

    buffer.clear()
    assert len(buffer) == 0
    with pytest.raises(ValueError, match="empty"):
        buffer.sample_offsets(1.0, [0.0])
    buffer.append(np.zeros((2, 2)), 0.0)


def test_timed_buffer_selection_can_be_overridden_per_query():
    buffer = TimedValueBuffer()
    buffer.append(0, 0.0)
    buffer.append(1, 0.1)

    causal, _ = buffer.sample_offsets(0.08, [0.0])
    nearest, _ = buffer.sample_offsets(0.08, [0.0], selection="nearest")

    np.testing.assert_array_equal(causal, [0])
    np.testing.assert_array_equal(nearest, [1])


@pytest.mark.parametrize(
    "operation",
    [
        lambda buffer: buffer.append(1, np.nan),
        lambda buffer: buffer.sample_offsets(np.inf, [0.0]),
        lambda buffer: buffer.sample_offsets(0.0, []),
        lambda buffer: buffer.sample_offsets(0.0, [[0.0]]),
        lambda buffer: buffer.sample_offsets(0.0, [np.nan]),
        lambda buffer: buffer.sample_offsets(0.0, [0.0], selection="invalid"),
    ],
)
def test_timed_buffer_validates_sampling_inputs(operation):
    buffer = TimedValueBuffer()
    buffer.append(0, 0.0)
    with pytest.raises(ValueError):
        operation(buffer)


def test_timed_buffer_requires_increasing_timestamps_and_stable_value_shape():
    buffer = TimedValueBuffer()
    buffer.append(np.zeros((2, 3)), 1.0)

    with pytest.raises(ValueError, match="strictly greater"):
        buffer.append(np.zeros((2, 3)), 1.0)
    with pytest.raises(ValueError, match="shape"):
        buffer.append(np.zeros((3, 2)), 2.0)
    with pytest.raises(ValueError, match="selection"):
        TimedValueBuffer(selection="future")


def test_robosuite_adapter_uses_live_state_each_call():
    adapter = RobosuiteOSCAdapter()
    target = _pose(
        xyz=(0.05, -0.025, 0.01),
        quaternion=rotvec_to_quaternion([0.0, 0.0, 0.25]),
        gripper=1.0,
    )

    first = adapter.adapt(_pose(), target)
    np.testing.assert_allclose(first, [1.0, -0.5, 0.2, 0.0, 0.0, 0.5, -1.0], atol=1e-7)

    # Recomputing from a newer measured pose must reduce the command rather
    # than integrating from the state supplied on the previous call.
    live_pose = _pose(
        xyz=(0.04, -0.02, 0.0),
        quaternion=rotvec_to_quaternion([0.0, 0.0, 0.2]),
        gripper=0.2,
    )
    second = adapter.adapt(live_pose, target)
    np.testing.assert_allclose(second, [0.2, -0.1, 0.2, 0.0, 0.0, 0.1, -1.0], atol=1e-7)


def test_robosuite_adapter_clips_pose_and_thresholds_gripper():
    adapter = RobosuiteOSCAdapter()
    target = _pose(
        xyz=(1.0, -1.0, 0.2),
        quaternion=rotvec_to_quaternion([1.0, -1.0, 0.75]),
        gripper=0.499,
    )

    command = adapter(_pose(), target)

    np.testing.assert_allclose(command, [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
    assert np.all(command >= -1.0)
    assert np.all(command <= 1.0)
    assert adapter.adapt(_pose(), _pose(gripper=0.5))[-1] == -1.0


def test_robosuite_adapter_supports_flattened_bimanual_io_and_protocol():
    adapter = RobosuiteOSCAdapter(arm_count=2)
    current = np.concatenate([_pose(), _pose(xyz=(1.0, 1.0, 1.0))])
    target = np.concatenate(
        [
            _pose(xyz=(0.025, 0.0, 0.0), gripper=1.0),
            _pose(xyz=(0.95, 1.0, 1.05), gripper=0.0),
        ]
    )

    command = adapter.command(current, target)

    assert isinstance(adapter, ExecutionAdapter)
    assert command.shape == (14,)
    np.testing.assert_allclose(command[:7], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    np.testing.assert_allclose(command[7:], [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "current,target",
    [
        (np.zeros(7), _pose()),
        (_pose(), np.zeros(9)),
        (np.full(8, np.nan), _pose()),
        (_pose(quaternion=(0.0, 0.0, 0.0, 0.0)), _pose()),
    ],
)
def test_robosuite_adapter_validates_canonical_poses(current, target):
    with pytest.raises(ValueError):
        RobosuiteOSCAdapter().adapt(current, target)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arm_count": 0},
        {"position_scale": 0.0},
        {"position_scale": np.inf},
        {"rotation_scale": -1.0},
    ],
)
def test_robosuite_adapter_validates_configuration(kwargs):
    with pytest.raises(ValueError):
        RobosuiteOSCAdapter(**kwargs)
