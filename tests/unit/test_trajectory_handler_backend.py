"""A numpy trajectory that lacks one of the model's joints, or carries one the model lacks, must keep a single backend.

TrajectoryHandler.filter_and_extend adds a missing joint with add_joint() and drops an extra one with remove_joints(), both
defaulting to jnp, while the rest of the trajectory stays numpy. is_numpy is judged on qpos alone, so the environment's to_jax() is skipped, and a jitted function that
indexes split_points with a traced index then fails.
"""

import dataclasses

import jax
import mujoco
import numpy as np
import pytest

from loco_mujoco.trajectory.dataclasses import Trajectory
from loco_mujoco.trajectory.handler import TrajectoryHandler

MODEL_XML = """<mujoco><worldbody>
<body name="a"><joint name="j1" type="hinge" axis="0 0 1"/><geom size="0.1" mass="1"/>
<body name="b" pos="0.3 0 0"><joint name="j2" type="hinge" axis="0 1 0"/><geom size="0.1" mass="1"/><site name="s1"/></body></body>
</worldbody></mujoco>"""
N_FRAMES = 50


def _numpy_traj(path, n_joints):
    np.savez(
        path,
        qpos=np.linspace(0, 0.5, N_FRAMES)[:, None] * np.ones((1, n_joints)),
        qvel=np.full((N_FRAMES, n_joints), 0.5),
        split_points=np.array([0, N_FRAMES]),
        joint_names=np.array([f"j{i + 1}" for i in range(n_joints)]),
        frequency=np.array(100.0),
        njnt=np.array(n_joints),
        jnt_type=np.array([int(mujoco.mjtJoint.mjJNT_HINGE)] * n_joints),
    )
    return Trajectory.load(str(path), backend=np)


@pytest.fixture
def numpy_traj_missing_a_joint(tmp_path):
    return _numpy_traj(tmp_path / "one_joint_of_two.npz", 1)


@pytest.fixture
def numpy_traj_with_an_extra_joint(tmp_path):
    return _numpy_traj(tmp_path / "three_joints_of_two.npz", 3)


@pytest.mark.parametrize("cache_type", ["full", "sparse"])
@pytest.mark.parametrize("traj_fixture", ["numpy_traj_missing_a_joint", "numpy_traj_with_an_extra_joint"])
def test_joints_added_or_removed_keep_the_trajectory_backend(traj_fixture, cache_type, request):
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    handler = TrajectoryHandler(
        model,
        traj=request.getfixturevalue(traj_fixture),
        control_dt=0.01,
        random_start=True,
        cache_type=cache_type,
        site_names=["s1"] if cache_type == "sparse" else None,
    )
    data = handler.traj.data
    backends = {
        field.name: type(getattr(data, field.name)).__module__.split(".")[0]
        for field in dataclasses.fields(data)
        if isinstance(getattr(data, field.name), np.ndarray | jax.Array) and getattr(data, field.name).size
    }
    assert len(set(backends.values())) == 1, f"trajectory data mixes backends: {backends}"
    assert handler.is_numpy, "a numpy trajectory should stay numpy until the environment converts it"
    handler.to_jax()
    assert int(jax.jit(lambda i: handler.len_trajectory(i))(0)) == N_FRAMES
