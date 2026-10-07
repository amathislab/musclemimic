"""Tests for heading-local full-body trajectory tracking observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from loco_mujoco.trajectory.dataclasses import SingleData, TrajectoryData
from musclemimic.core.goals.fullbody_tracking import (
    DEFAULT_LOOKAHEAD_STEPS,
    SUPPORT_SITE_NAMES,
    GoalFullBodyTracking,
    GoalFullBodyTrackingVisual,
    rescale_lookahead_steps,
)
from musclemimic.core.goals.tracking_features import (
    reference_support_features,
    root_tracking_error_features,
    tracking_goal_dimension,
)
from musclemimic.core.observations.egocentric import HeadingFrameFreeJointVelocity, heading_rotation_matrix

jax.config.update("jax_platform_name", "cpu")


@dataclass(frozen=True)
class _TrajectoryFrame:
    qpos: np.ndarray
    qvel: np.ndarray
    site_xpos: np.ndarray
    site_xmat: np.ndarray
    cvel: np.ndarray
    subtree_com: np.ndarray


@dataclass(frozen=True)
class _TrajectoryInfo:
    site_names: list[str]


class _TrajectoryData:
    def __init__(self, frames: Sequence[_TrajectoryFrame]) -> None:
        self._frames = tuple(frames)

    def get(self, traj_no: int, step: int, backend: Any) -> _TrajectoryFrame:
        del traj_no, backend
        return self._frames[int(step)]


class _TrajectoryHandler:
    def __init__(self, frames: Sequence[_TrajectoryFrame], site_names: list[str]) -> None:
        self._frames = tuple(frames)
        self.traj = SimpleNamespace(
            data=_TrajectoryData(frames),
            info=_TrajectoryInfo(site_names=site_names),
        )

    def len_trajectory(self, traj_no: int) -> int:
        del traj_no
        return len(self._frames)

    def get_traj_data_at(
        self,
        traj_no: int,
        step: int,
        carry: Any,
        backend: Any,
    ) -> _TrajectoryFrame:
        del carry
        return self.traj.data.get(traj_no, step, backend)


class _SparseTrajectoryHandler:
    def __init__(self, trajectory_data: TrajectoryData, site_names: list[str]) -> None:
        self.traj = SimpleNamespace(
            data=trajectory_data,
            info=_TrajectoryInfo(site_names=site_names),
        )

    def len_trajectory(self, traj_no: Any) -> Any:
        split_points = self.traj.data.split_points
        return split_points[traj_no + 1] - split_points[traj_no]

    def get_traj_data_at(
        self,
        traj_no: int,
        step: int,
        carry: Any,
        backend: Any,
    ) -> SingleData:
        del carry
        return self.traj.data.get(traj_no, step, backend)


class DummyMyoFullBody:
    root_free_joint_xml_name = "root"
    sites_for_mimic = ("pelvis_mimic", "head_mimic", *SUPPORT_SITE_NAMES)
    dt = 0.01


_TRAJECTORY_SITE_NAMES = tuple(reversed(DummyMyoFullBody.sites_for_mimic))


def _tracking_block_dimensions(*, include_support_intent: bool = False) -> tuple[int, int, int]:
    n_relative_sites = len(DummyMyoFullBody.sites_for_mimic) - 1
    support_dim = 2 * len(SUPPORT_SITE_NAMES) if include_support_intent else 0
    immediate_error_dim = 12 + 12 * n_relative_sites
    immediate_dim = immediate_error_dim + support_dim
    future_dim = 12 + 3 * n_relative_sites + support_dim + 2
    return immediate_error_dim, immediate_dim, future_dim


def _make_model(*, include_floor: bool = True, include_support_sites: bool = True) -> mujoco.MjModel:
    floor_xml = '<geom name="floor" type="plane" size="2 2 0.1" pos="0 0 -0.05"/>' if include_floor else ""
    support_sites_xml = (
        """
          <site name="left_ankle_mimic" pos="-0.1 0 -0.7" size="0.01"/>
          <site name="left_toes_mimic" pos="-0.1 0.2 -0.8" size="0.01"/>
          <site name="right_ankle_mimic" pos="0.1 0 -0.7" size="0.01"/>
          <site name="right_toes_mimic" pos="0.1 0.2 -0.8" size="0.01"/>
        """
        if include_support_sites
        else ""
    )
    xml = f"""
    <mujoco model="tracking_goal_test">
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        {floor_xml}
        <body name="torso">
          <freejoint name="root"/>
          <geom type="sphere" size="0.05" mass="1"/>
          <site name="pelvis_mimic" size="0.01"/>
          {support_sites_xml}
          <body name="head" pos="0 0 1">
            <joint name="neck" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size="0.03" mass="0.5"/>
            <site name="head_mimic" pos="0.2 0 0" size="0.01"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _yaw_rotation(yaw: float) -> np.ndarray:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _make_data(
    model: mujoco.MjModel,
    *,
    root_position: np.ndarray,
    root_yaw: float,
    root_linear_velocity: np.ndarray,
    root_angular_velocity: np.ndarray,
    joint_position: float,
    joint_velocity: float,
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = root_position
    data.qpos[3:7] = _yaw_quaternion(root_yaw)
    data.qpos[7] = joint_position
    data.qvel[:3] = root_linear_velocity
    data.qvel[3:6] = root_angular_velocity
    data.qvel[6] = joint_velocity
    mujoco.mj_forward(model, data)
    return data


def _trajectory_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory_site_names: Sequence[str],
) -> _TrajectoryFrame:
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in trajectory_site_names]
    return _TrajectoryFrame(
        qpos=data.qpos.copy(),
        qvel=data.qvel.copy(),
        site_xpos=data.site_xpos[site_ids].copy(),
        site_xmat=data.site_xmat[site_ids].copy(),
        cvel=data.cvel.copy(),
        subtree_com=data.subtree_com.copy(),
    )


def _sparse_trajectory_data(
    model: mujoco.MjModel,
    frames: Sequence[_TrajectoryFrame],
    trajectory_site_names: Sequence[str],
) -> TrajectoryData:
    site_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in trajectory_site_names])
    parent_body_ids = model.site_bodyid[site_ids]
    root_body_ids = np.unique(model.body_rootid[parent_body_ids])
    assert root_body_ids.size == 1

    return TrajectoryData(
        qpos=jnp.asarray(np.stack([frame.qpos for frame in frames])),
        qvel=jnp.asarray(np.stack([frame.qvel for frame in frames])),
        site_xpos=jnp.asarray(np.stack([frame.site_xpos for frame in frames])),
        site_xmat=jnp.asarray(np.stack([frame.site_xmat for frame in frames])),
        cvel_parent=jnp.asarray(np.stack([frame.cvel[parent_body_ids] for frame in frames])),
        subtree_com_root=jnp.asarray(np.stack([frame.subtree_com[root_body_ids[0]] for frame in frames])),
        split_points=jnp.asarray([0, len(frames)]),
    )


def _jax_simulation_data(data: mujoco.MjData) -> SingleData:
    return SingleData(
        qpos=jnp.asarray(data.qpos),
        qvel=jnp.asarray(data.qvel),
        cvel=jnp.asarray(data.cvel),
        subtree_com=jnp.asarray(data.subtree_com),
        site_xpos=jnp.asarray(data.site_xpos),
        site_xmat=jnp.asarray(data.site_xmat),
    )


def _build_scenario(global_yaw: float = 0.0) -> tuple[mujoco.MjModel, mujoco.MjData, list[_TrajectoryFrame]]:
    model = _make_model()
    global_rotation = _yaw_rotation(global_yaw)
    initial_position = np.asarray([100.0, 200.0, 1.0])
    trajectory_site_names = _TRAJECTORY_SITE_NAMES
    frames = []

    for step in range(6):
        displacement = np.asarray([0.1 * step, 0.04 * step, 0.0])
        position = global_rotation @ (initial_position + displacement)
        linear_velocity = global_rotation @ np.asarray([0.8, 0.2, 0.1])
        frame_data = _make_data(
            model,
            root_position=position,
            root_yaw=0.2 + 0.03 * step + global_yaw,
            root_linear_velocity=linear_velocity,
            root_angular_velocity=np.asarray([0.1, -0.2, 0.3]),
            joint_position=0.05 * step,
            joint_velocity=0.15,
        )
        frames.append(_trajectory_frame(model, frame_data, trajectory_site_names))

    aligned_step_one_position = global_rotation @ np.asarray([0.1, 0.04, 1.0])
    simulation_data = _make_data(
        model,
        root_position=aligned_step_one_position,
        root_yaw=0.23 + global_yaw,
        root_linear_velocity=global_rotation @ np.asarray([0.8, 0.2, 0.1]),
        root_angular_velocity=np.asarray([0.1, -0.2, 0.3]),
        joint_position=0.05,
        joint_velocity=0.15,
    )
    return model, simulation_data, frames


def _perturbed_simulation(model: mujoco.MjModel, global_yaw: float) -> mujoco.MjData:
    global_rotation = _yaw_rotation(global_yaw)
    return _make_data(
        model,
        root_position=global_rotation @ np.asarray([0.13, 0.02, 1.02]),
        root_yaw=0.31 + global_yaw,
        root_linear_velocity=global_rotation @ np.asarray([0.65, 0.32, 0.04]),
        root_angular_velocity=np.asarray([0.2, -0.1, 0.4]),
        joint_position=0.18,
        joint_velocity=-0.05,
    )


def _make_goal(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frames: Sequence[_TrajectoryFrame],
    *,
    goal_class: type[GoalFullBodyTracking] = GoalFullBodyTracking,
    lookahead_steps: tuple[int, ...] = (1, 2, 4),
    include_support_intent: bool = False,
) -> tuple[GoalFullBodyTracking, DummyMyoFullBody]:
    info = {
        "upper_body_xml_name": "torso",
        "sites_for_mimic": DummyMyoFullBody.sites_for_mimic,
    }
    env = DummyMyoFullBody()
    env.th = _TrajectoryHandler(frames, list(_TRAJECTORY_SITE_NAMES))
    env.mjspec = SimpleNamespace(geoms=[])
    goal = goal_class(
        info,
        lookahead_steps=lookahead_steps,
        include_support_intent=include_support_intent,
        enable_motion_phase=False,
        visualize_goal=False,
    )
    goal._init_from_mj(env, model, data, current_obs_size=0)
    goal.init_from_traj(env.th)
    return goal, env


def _goal_observation(
    goal: GoalFullBodyTracking,
    env: DummyMyoFullBody,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    current_step: int = 0,
    initial_step: int = 0,
    backend: Any = np,
) -> np.ndarray | jax.Array:
    carry = SimpleNamespace(
        traj_state=SimpleNamespace(
            traj_no=0,
            subtraj_step_no=current_step,
            subtraj_step_no_init=initial_step,
        )
    )
    observation, _ = goal.get_obs_and_update_state(env, model, data, carry, backend)
    return observation


def test_tracking_goal_dimension_matches_fullbody_layout() -> None:
    assert tracking_goal_dimension(n_relative_sites=16, n_horizons=5, n_support_sites=0) == 452
    assert tracking_goal_dimension(n_relative_sites=16, n_horizons=6, n_support_sites=0) == 514
    assert tracking_goal_dimension(n_relative_sites=16, n_horizons=6, n_support_sites=4) == 562


def test_support_intent_is_explicitly_opt_in() -> None:
    model, data, frames = _build_scenario()
    baseline_goal, baseline_env = _make_goal(model, data, frames)
    support_goal, support_env = _make_goal(model, data, frames, include_support_intent=True)

    baseline = _goal_observation(baseline_goal, baseline_env, model, data)
    supported = _goal_observation(support_goal, support_env, model, data)

    assert baseline_goal.include_support_intent is False
    assert baseline_goal._support_site_indices is None
    assert baseline_goal._floor_height is None
    assert support_goal.include_support_intent is True
    assert supported.size - baseline.size == len(support_goal.lookahead_steps) * 2 * len(SUPPORT_SITE_NAMES)


def test_disabled_support_intent_does_not_require_floor_or_support_sites() -> None:
    model = _make_model(include_floor=False, include_support_sites=False)
    data = _make_data(
        model,
        root_position=np.asarray([0.0, 0.0, 1.0]),
        root_yaw=0.0,
        root_linear_velocity=np.zeros(3),
        root_angular_velocity=np.zeros(3),
        joint_position=0.0,
        joint_velocity=0.0,
    )
    info = {
        "upper_body_xml_name": "torso",
        "sites_for_mimic": ("pelvis_mimic", "head_mimic"),
    }
    env = SimpleNamespace(
        root_free_joint_xml_name="root",
        sites_for_mimic=("pelvis_mimic", "head_mimic"),
    )
    goal = GoalFullBodyTracking(info, enable_motion_phase=False, visualize_goal=False)

    goal._init_from_mj(env, model, data, current_obs_size=0)

    assert goal._support_site_indices is None
    assert goal._floor_height is None


def test_default_lookahead_includes_tenth_second_intent() -> None:
    assert DEFAULT_LOOKAHEAD_STEPS == (1, 5, 10, 20, 40, 80)


def test_reference_support_features_are_continuous_and_heading_invariant() -> None:
    positions = np.asarray(
        [
            [0.2, 0.4, 0.31],
            [-0.3, 0.1, 0.08],
            [0.5, -0.2, 0.19],
            [-0.4, -0.6, 0.12],
        ]
    )
    velocities = np.zeros((4, 6))
    velocities[:, 5] = np.asarray([0.7, -0.2, 0.0, 0.35])
    support_indices = np.asarray([0, 1, 2, 3])
    floor_height = 0.05

    features = reference_support_features(positions, velocities, support_indices, floor_height, np)
    np.testing.assert_allclose(
        features,
        [0.26, 0.7, 0.03, -0.2, 0.14, 0.0, 0.07, 0.35],
    )

    yaw_rotation = _yaw_rotation(1.1)
    transformed_positions = positions @ yaw_rotation.T + np.asarray([3.0, -4.0, 0.0])
    transformed_velocities = velocities.copy()
    transformed_velocities[:, 3:6] = velocities[:, 3:6] @ yaw_rotation.T
    transformed = reference_support_features(
        transformed_positions,
        transformed_velocities,
        support_indices,
        floor_height,
        np,
    )
    np.testing.assert_allclose(transformed, features, atol=1e-12)

    encode_jax = jax.jit(
        lambda site_positions, site_velocities: reference_support_features(
            site_positions,
            site_velocities,
            jnp.asarray(support_indices),
            jnp.asarray(floor_height),
            jnp,
        )
    )
    jax_features = encode_jax(jnp.asarray(positions), jnp.asarray(velocities))
    np.testing.assert_allclose(np.asarray(jax_features), features, atol=1e-7)


@pytest.mark.parametrize(
    ("lookahead_steps", "message"),
    [
        ((), "start with 1"),
        ((2, 5), "start with 1"),
        ((1, 0), "positive"),
        ((1, 5, 5), "strictly increasing"),
        ((1, 5, 4), "strictly increasing"),
    ],
)
def test_tracking_goal_rejects_invalid_lookahead_steps(
    lookahead_steps: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GoalFullBodyTracking({}, lookahead_steps=lookahead_steps)


@pytest.mark.parametrize("lookahead_steps", [(1, 2.5), (1, "5"), (True, 5)])
def test_tracking_goal_rejects_non_integer_lookahead_steps(lookahead_steps: tuple[Any, ...]) -> None:
    with pytest.raises(TypeError, match="contain integers"):
        GoalFullBodyTracking({}, lookahead_steps=lookahead_steps)


def test_lookahead_rescaling_preserves_future_times_and_next_frame() -> None:
    assert rescale_lookahead_steps(DEFAULT_LOOKAHEAD_STEPS, 0.01, 0.02) == (1, 2, 5, 10, 20, 40)
    assert rescale_lookahead_steps(DEFAULT_LOOKAHEAD_STEPS, 0.01, 0.05) == (1, 2, 3, 4, 8, 16)


def test_immediate_tracking_error_uses_next_reference_frame() -> None:
    model, data, frames = _build_scenario()
    goal, env = _make_goal(model, data, frames)

    observation = _goal_observation(goal, env, model, data)
    immediate_error_dim, _, _ = _tracking_block_dimensions()

    assert observation.shape == (goal.dim,)
    np.testing.assert_allclose(observation[:immediate_error_dim], 0.0, atol=1e-6)


def test_tracking_goal_aligns_reference_to_episode_xy_origin() -> None:
    model, data, frames = _build_scenario()
    goal, env = _make_goal(model, data, frames)

    observation = _goal_observation(goal, env, model, data)

    np.testing.assert_allclose(observation[:3], 0.0, atol=1e-6)


def test_tracking_goal_exposes_simulated_root_drift() -> None:
    model, data, frames = _build_scenario()
    goal, env = _make_goal(model, data, frames)
    baseline = _goal_observation(goal, env, model, data)

    drifted_data = _make_data(
        model,
        root_position=data.qpos[:3] + np.asarray([0.3, 0.0, 0.0]),
        root_yaw=0.23,
        root_linear_velocity=data.qvel[:3],
        root_angular_velocity=data.qvel[3:6],
        joint_position=float(data.qpos[7]),
        joint_velocity=float(data.qvel[6]),
    )
    drifted = _goal_observation(goal, env, model, drifted_data)
    immediate_error_dim, _, _ = _tracking_block_dimensions()

    assert not np.allclose(drifted[:3], baseline[:3])
    np.testing.assert_allclose(drifted[12:immediate_error_dim], baseline[12:immediate_error_dim], atol=1e-6)


@pytest.mark.parametrize("include_support_intent", [False, True])
def test_future_intents_include_requested_time_and_validity(include_support_intent: bool) -> None:
    model, data, frames = _build_scenario()
    goal, env = _make_goal(model, data, frames, include_support_intent=include_support_intent)

    observation = _goal_observation(goal, env, model, data, current_step=3)
    _, immediate_dim, future_dim = _tracking_block_dimensions(include_support_intent=include_support_intent)
    first_future_metadata = observation[immediate_dim + future_dim - 2 : immediate_dim + future_dim]
    second_future_metadata = observation[immediate_dim + 2 * future_dim - 2 : immediate_dim + 2 * future_dim]
    first_future_payload = observation[immediate_dim : immediate_dim + future_dim - 2]
    second_future_payload = observation[immediate_dim + future_dim : immediate_dim + 2 * future_dim - 2]

    np.testing.assert_allclose(first_future_metadata, [0.02, 1.0], atol=1e-7)
    np.testing.assert_allclose(second_future_metadata, [0.04, 0.0], atol=1e-7)
    np.testing.assert_allclose(second_future_payload, first_future_payload, atol=1e-6)


@pytest.mark.parametrize("include_support_intent", [False, True])
def test_tracking_goal_is_invariant_to_global_yaw(include_support_intent: bool) -> None:
    model, _, frames = _build_scenario(global_yaw=0.0)
    data = _perturbed_simulation(model, global_yaw=0.0)
    goal, env = _make_goal(model, data, frames, include_support_intent=include_support_intent)
    baseline = _goal_observation(goal, env, model, data)

    rotated_model, _, rotated_frames = _build_scenario(global_yaw=np.pi / 2.0)
    rotated_data = _perturbed_simulation(rotated_model, global_yaw=np.pi / 2.0)
    rotated_goal, rotated_env = _make_goal(
        rotated_model,
        rotated_data,
        rotated_frames,
        include_support_intent=include_support_intent,
    )
    rotated = _goal_observation(rotated_goal, rotated_env, rotated_model, rotated_data)

    immediate_error_dim, _, _ = _tracking_block_dimensions()
    assert np.linalg.norm(baseline[:12]) > 1e-4
    assert np.linalg.norm(baseline[12:immediate_error_dim]) > 1e-4
    np.testing.assert_allclose(rotated, baseline, atol=2e-5)


def test_tracking_goal_is_invariant_to_quaternion_sign() -> None:
    model, data, frames = _build_scenario()
    goal, env = _make_goal(model, data, frames)
    baseline = _goal_observation(goal, env, model, data)

    data.qpos[3:7] *= -1.0
    signed_frames = [replace(frame, qpos=frame.qpos.copy()) for frame in frames]
    for frame in signed_frames:
        frame.qpos[3:7] *= -1.0
    signed_goal, signed_env = _make_goal(model, data, signed_frames)
    signed = _goal_observation(signed_goal, signed_env, model, data)

    np.testing.assert_allclose(signed, baseline, atol=1e-6)


def test_tracking_goal_jax_backend_matches_numpy() -> None:
    model, data, frames = _build_scenario()
    goal, env = _make_goal(model, data, frames)

    numpy_observation = _goal_observation(goal, env, model, data, current_step=3, backend=np)
    jax_observation = _goal_observation(goal, env, model, data, current_step=3, backend=jnp)

    np.testing.assert_allclose(np.asarray(jax_observation), numpy_observation, atol=2e-5)


@pytest.mark.parametrize("include_support_intent", [False, True])
def test_tracking_goal_sparse_trajectory_is_jittable(include_support_intent: bool) -> None:
    model, data, frames = _build_scenario()
    goal, env = _make_goal(model, data, frames, include_support_intent=include_support_intent)
    trajectory_site_names = _TRAJECTORY_SITE_NAMES
    sparse_trajectory = _sparse_trajectory_data(model, frames, trajectory_site_names)
    env.th = _SparseTrajectoryHandler(sparse_trajectory, trajectory_site_names)
    simulation_data = _jax_simulation_data(data)

    def encode(current_step: jax.Array) -> jax.Array:
        carry = SimpleNamespace(
            traj_state=SimpleNamespace(
                traj_no=jnp.asarray(0),
                subtraj_step_no=current_step,
                subtraj_step_no_init=jnp.asarray(0),
            )
        )
        observation, _ = goal.get_obs_and_update_state(
            env,
            model,
            simulation_data,
            carry,
            jnp,
        )
        return observation

    observations = jax.jit(jax.vmap(encode))(jnp.asarray([0, 3]))

    assert sparse_trajectory.cvel.size == 0
    assert sparse_trajectory.subtree_com.size == 0
    assert observations.shape == (2, goal.dim)
    assert jnp.all(jnp.isfinite(observations))


def test_visualization_goal_preserves_tracking_layout() -> None:
    model, data, frames = _build_scenario()
    goal, env = _make_goal(model, data, frames)
    visual_goal, visual_env = _make_goal(model, data, frames, goal_class=GoalFullBodyTrackingVisual)

    observation = _goal_observation(goal, env, model, data)
    visual_observation = _goal_observation(visual_goal, visual_env, model, data)

    assert visual_goal.dim == goal.dim
    np.testing.assert_allclose(visual_observation, observation, atol=1e-6)


def test_heading_frame_free_joint_velocity_handles_mujoco_frames() -> None:
    model = _make_model()
    baseline_data = _make_data(
        model,
        root_position=np.asarray([0.0, 0.0, 1.0]),
        root_yaw=0.0,
        root_linear_velocity=np.asarray([1.0, 0.0, 0.0]),
        root_angular_velocity=np.asarray([1.0, 0.0, 0.0]),
        joint_position=0.0,
        joint_velocity=0.0,
    )
    rotated_data = _make_data(
        model,
        root_position=np.asarray([0.0, 0.0, 1.0]),
        root_yaw=np.pi / 2.0,
        root_linear_velocity=np.asarray([0.0, 1.0, 0.0]),
        root_angular_velocity=np.asarray([1.0, 0.0, 0.0]),
        joint_position=0.0,
        joint_velocity=0.0,
    )
    observation_type = HeadingFrameFreeJointVelocity("root_velocity", "root")
    observation_type._init_from_mj(None, model, baseline_data, current_obs_size=0)

    baseline, _ = observation_type.get_obs_and_update_state(
        None,
        model,
        baseline_data,
        None,
        np,
    )
    rotated, _ = observation_type.get_obs_and_update_state(
        None,
        model,
        rotated_data,
        None,
        np,
    )

    np.testing.assert_allclose(rotated, baseline, atol=1e-6)
    np.testing.assert_allclose(rotated, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-6)

    def encode(qpos: jax.Array, qvel: jax.Array) -> jax.Array:
        observation, _ = observation_type.get_obs_and_update_state(
            None,
            model,
            SingleData(qpos=qpos, qvel=qvel),
            None,
            jnp,
        )
        return observation

    encode_jit = jax.jit(encode)
    baseline_jax = encode_jit(jnp.asarray(baseline_data.qpos), jnp.asarray(baseline_data.qvel))
    rotated_jax = encode_jit(jnp.asarray(rotated_data.qpos), jnp.asarray(rotated_data.qvel))
    signed_qpos = jnp.asarray(rotated_data.qpos).at[3:7].multiply(-1.0)

    np.testing.assert_allclose(rotated_jax, baseline_jax, atol=1e-6)
    np.testing.assert_allclose(encode_jit(signed_qpos, jnp.asarray(rotated_data.qvel)), rotated_jax, atol=1e-6)


def test_heading_rotation_is_defined_when_forward_axis_is_vertical() -> None:
    yaw = 0.7
    pitch_ninety_degrees = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    root_rotation = _yaw_rotation(yaw) @ pitch_ninety_degrees

    numpy_heading = heading_rotation_matrix(root_rotation, np)
    jax_heading = jax.jit(lambda rotation: heading_rotation_matrix(rotation, jnp))(jnp.asarray(root_rotation))

    np.testing.assert_allclose(numpy_heading, _yaw_rotation(yaw), atol=1e-6)
    np.testing.assert_allclose(jax_heading, numpy_heading, atol=1e-6)


def test_root_tracking_features_are_jittable() -> None:
    def encode(
        current_position: jax.Array,
        current_rotation: jax.Array,
        current_qvel: jax.Array,
        target_position: jax.Array,
        target_rotation: jax.Array,
        target_qvel: jax.Array,
    ) -> jax.Array:
        heading_rotation = heading_rotation_matrix(current_rotation, jnp)
        return root_tracking_error_features(
            current_position,
            current_rotation,
            current_qvel,
            target_position,
            target_rotation,
            target_qvel,
            heading_rotation,
            jnp,
        )

    identity = jnp.eye(3)
    features = jax.jit(encode)(
        jnp.zeros(3),
        identity,
        jnp.zeros(6),
        jnp.asarray([0.1, -0.2, 0.3]),
        identity,
        jnp.ones(6),
    )

    assert features.shape == (12,)
    assert jnp.all(jnp.isfinite(features))
