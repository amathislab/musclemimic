from dataclasses import replace

import mujoco
import numpy as np
import pytest

from loco_mujoco.trajectory.dataclasses import (
    Trajectory,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
)


def _trajectory(*, metadata: dict, value: float = 0.0, frequency: float = 100.0) -> Trajectory:
    model = TrajectoryModel(
        njnt=1,
        jnt_type=np.asarray([int(mujoco.mjtJoint.mjJNT_HINGE)]),
    )
    info = TrajectoryInfo(
        joint_names=["hinge"],
        model=model,
        frequency=frequency,
        metadata=metadata,
    )
    data = TrajectoryData(
        qpos=np.full((2, 1), value, dtype=np.float32),
        qvel=np.zeros((2, 1), dtype=np.float32),
        split_points=np.asarray([0, 2]),
    )
    return Trajectory(info=info, data=data)


def test_concatenate_accepts_motion_specific_metadata():
    first = _trajectory(
        metadata={"joint_range_interpolation_rescue_frames": [1]},
        value=1.0,
    )
    second = _trajectory(
        metadata={"joint_range_interpolation_rescue_frames": []},
        value=2.0,
    )

    combined = Trajectory.concatenate([first, second], backend=np)

    np.testing.assert_array_equal(combined.data.split_points, [0, 2, 4])
    np.testing.assert_array_equal(combined.data.qpos[:, 0], [1.0, 1.0, 2.0, 2.0])
    assert combined.info.metadata is None


def test_concatenate_retains_common_metadata():
    metadata = {"retargeting_method": "terra"}

    combined = Trajectory.concatenate(
        [_trajectory(metadata=metadata), _trajectory(metadata=dict(metadata))],
        backend=np,
    )

    assert combined.info.metadata == metadata


def test_concatenate_still_rejects_different_sampling_frequencies():
    first = _trajectory(metadata={})
    second = replace(_trajectory(metadata={}), info=replace(first.info, frequency=50.0))

    with pytest.raises(AssertionError, match="compatible for concatenation"):
        Trajectory.concatenate([first, second], backend=np)
