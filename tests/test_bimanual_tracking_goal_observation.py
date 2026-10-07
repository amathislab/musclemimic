"""Tests for heading-local fixed-root bimanual tracking observations."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from loco_mujoco.trajectory.dataclasses import SingleData, TrajectoryData
from musclemimic.core.goals.bimanual_tracking import GoalBimanualTracking, GoalBimanualTrackingVisual
from musclemimic.core.goals.tracking_features import fixed_root_tracking_goal_dimension

jax.config.update("jax_platform_name", "cpu")


@dataclass(frozen=True)
class _TrajectoryInfo:
    site_names: list[str]


class _TrajectoryHandler:
    def __init__(self, trajectory_data: TrajectoryData, site_names: list[str]) -> None:
        self.traj = SimpleNamespace(data=trajectory_data, info=_TrajectoryInfo(site_names))

    def len_trajectory(self, traj_no: Any) -> Any:
        split_points = self.traj.data.split_points
        return split_points[traj_no + 1] - split_points[traj_no]

    def get_traj_data_at(
        self,
        traj_no: Any,
        step: Any,
        carry: Any,
        backend: Any,
    ) -> SingleData:
        del carry
        return self.traj.data.get(traj_no, step, backend)


class DummyMyoBimanualArm:
    sites_for_mimic = ("torso_mimic", "hand_mimic")
    dt = 0.01


def _make_model() -> mujoco.MjModel:
    xml = """
    <mujoco model="bimanual_tracking_goal_test">
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body name="torso">
          <geom type="sphere" size="0.05" mass="1"/>
          <site name="torso_mimic" size="0.01"/>
          <body name="hand" pos="0 0 0.5">
            <joint name="shoulder" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size="0.03" mass="0.5"/>
            <site name="hand_mimic" pos="0.4 0 0" size="0.01"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _make_data(model: mujoco.MjModel, joint_position: float) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[0] = joint_position
    data.qvel[0] = 0.2
    mujoco.mj_forward(model, data)
    return data


def _trajectory_data(
    model: mujoco.MjModel,
    frames: list[mujoco.MjData],
    trajectory_site_names: tuple[str, ...],
) -> TrajectoryData:
    site_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in trajectory_site_names])
    parent_body_ids = model.site_bodyid[site_ids]
    root_body_ids = np.unique(model.body_rootid[parent_body_ids])
    assert root_body_ids.size == 1

    return TrajectoryData(
        qpos=jnp.asarray(np.stack([frame.qpos for frame in frames])),
        qvel=jnp.asarray(np.stack([frame.qvel for frame in frames])),
        site_xpos=jnp.asarray(np.stack([frame.site_xpos[site_ids] for frame in frames])),
        site_xmat=jnp.asarray(np.stack([frame.site_xmat[site_ids] for frame in frames])),
        cvel_parent=jnp.asarray(np.stack([frame.cvel[parent_body_ids] for frame in frames])),
        subtree_com_root=jnp.asarray(np.stack([frame.subtree_com[root_body_ids[0]] for frame in frames])),
        split_points=jnp.asarray([0, len(frames)]),
    )


def _jax_data(data: mujoco.MjData) -> SingleData:
    return SingleData(
        qpos=jnp.asarray(data.qpos),
        qvel=jnp.asarray(data.qvel),
        cvel=jnp.asarray(data.cvel),
        subtree_com=jnp.asarray(data.subtree_com),
        site_xpos=jnp.asarray(data.site_xpos),
        site_xmat=jnp.asarray(data.site_xmat),
    )


def _make_goal(
    goal_class: type[GoalBimanualTracking] = GoalBimanualTracking,
) -> tuple[GoalBimanualTracking, DummyMyoBimanualArm, mujoco.MjModel, mujoco.MjData]:
    model = _make_model()
    frames = [_make_data(model, 0.1 * step) for step in range(6)]
    current_data = _make_data(model, 0.1)
    trajectory_site_names = tuple(reversed(DummyMyoBimanualArm.sites_for_mimic))
    trajectory_data = _trajectory_data(model, frames, trajectory_site_names)
    env = DummyMyoBimanualArm()
    env.th = _TrajectoryHandler(trajectory_data, list(trajectory_site_names))
    goal = goal_class(
        {
            "upper_body_xml_name": "torso",
            "sites_for_mimic": DummyMyoBimanualArm.sites_for_mimic,
        },
        lookahead_steps=(1, 2, 4),
        enable_motion_phase=False,
        visualize_goal=False,
    )
    goal._init_from_mj(env, model, current_data, current_obs_size=0)
    goal.init_from_traj(env.th)
    return goal, env, model, current_data


def _observation(
    goal: GoalBimanualTracking,
    env: DummyMyoBimanualArm,
    model: mujoco.MjModel,
    data: Any,
    backend: Any,
) -> Any:
    carry = SimpleNamespace(
        traj_state=SimpleNamespace(
            traj_no=0,
            subtraj_step_no=0,
        )
    )
    return goal.get_obs_and_update_state(env, model, data, carry, backend)[0]


def test_fixed_root_tracking_goal_dimension() -> None:
    assert fixed_root_tracking_goal_dimension(n_relative_sites=6, n_horizons=6) == 172


def test_bimanual_tracking_observation_uses_next_frame_and_future_metadata() -> None:
    goal, env, model, data = _make_goal()

    observation = _observation(goal, env, model, data, np)

    assert observation.shape == (goal.dim,)
    np.testing.assert_allclose(observation[:12], 0.0, atol=1e-6)
    np.testing.assert_allclose(observation[15:17], [0.02, 1.0], atol=1e-6)
    np.testing.assert_allclose(observation[20:22], [0.04, 1.0], atol=1e-6)


def test_bimanual_tracking_observation_matches_jit() -> None:
    goal, env, model, data = _make_goal()
    expected = _observation(goal, env, model, data, np)

    observed = jax.jit(lambda sim_data: _observation(goal, env, model, sim_data, jnp))(_jax_data(data))

    np.testing.assert_allclose(observed, expected, atol=1e-6)


def test_bimanual_tracking_visual_goal_preserves_observation_layout() -> None:
    goal, env, model, data = _make_goal()
    visual_goal, visual_env, visual_model, visual_data = _make_goal(GoalBimanualTrackingVisual)

    expected = _observation(goal, env, model, data, np)
    observed = _observation(visual_goal, visual_env, visual_model, visual_data, np)

    assert visual_goal.dim == goal.dim
    np.testing.assert_allclose(observed, expected, atol=1e-6)
