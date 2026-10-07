"""Egocentric full-body trajectory tracking goal."""

from collections.abc import Sequence
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils.mujoco import mj_jntid2qposid
from musclemimic.core.goals.tracking import (
    DEFAULT_LOOKAHEAD_STEPS,
    TrackingGoalMixin,
    rescale_lookahead_steps,
)
from musclemimic.core.goals.tracking_features import (
    future_intent_features,
    reference_support_features,
    root_tracking_error_features,
    site_kinematics,
    site_positions,
    site_tracking_error_features,
    tracking_goal_dimension,
)
from musclemimic.core.goals.trajectory import GoalTrajMimic, GoalTrajMimicv2
from musclemimic.core.observations.egocentric import (
    heading_rotation_matrix,
    rotation_matrix_from_quaternion,
)

SUPPORT_SITE_NAMES = (
    "left_ankle_mimic",
    "left_toes_mimic",
    "right_ankle_mimic",
    "right_toes_mimic",
)
_FLOOR_GEOM_NAME = "floor"


class GoalFullBodyTracking(TrackingGoalMixin, GoalTrajMimic):
    """Full next-frame error followed by compact future intent.

    Support intent can add reference ankle and toe height plus vertical
    velocity at every horizon without introducing a binary contact label.
    """

    def __init__(
        self,
        info_props: dict,
        rel_body_names: list[str] | None = None,
        lookahead_steps: Sequence[int] = DEFAULT_LOOKAHEAD_STEPS,
        include_support_intent: bool = False,
        **kwargs: Any,
    ) -> None:
        self.include_support_intent = include_support_intent
        self._root_qpos_indices = None
        self._support_site_indices = None
        self._floor_height = None
        super().__init__(
            info_props,
            rel_body_names=rel_body_names,
            lookahead_steps=lookahead_steps,
            **kwargs,
        )

    def _set_goal_dimensions(self, n_relative_sites: int) -> None:
        self._dim = tracking_goal_dimension(
            n_relative_sites,
            len(self.lookahead_steps),
            len(SUPPORT_SITE_NAMES) if self.include_support_intent else 0,
        )
        self._size_additional_observation = 0

    def _init_from_mj(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        current_obs_size: int,
    ) -> None:
        super()._init_from_mj(env, model, data, current_obs_size)
        root_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            env.root_free_joint_xml_name,
        )
        self._root_qpos_indices = np.asarray(mj_jntid2qposid(root_joint_id, model))

        if self.include_support_intent:
            mimic_site_names = tuple(self._info_props["sites_for_mimic"])
            missing_support_sites = [name for name in SUPPORT_SITE_NAMES if name not in mimic_site_names]
            if missing_support_sites:
                raise ValueError(f"sites_for_mimic is missing support sites: {missing_support_sites}")
            self._support_site_indices = np.asarray(
                [mimic_site_names.index(name) for name in SUPPORT_SITE_NAMES],
                dtype=np.int32,
            )

            floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _FLOOR_GEOM_NAME)
            if floor_geom_id < 0:
                raise ValueError(f"model is missing floor geometry '{_FLOOR_GEOM_NAME}'")
            if model.geom_type[floor_geom_id] != mujoco.mjtGeom.mjGEOM_PLANE:
                raise ValueError(f"floor geometry '{_FLOOR_GEOM_NAME}' must be a plane")
            self._floor_height = float(model.geom_pos[floor_geom_id, 2])

    def get_obs_and_update_state(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> tuple[np.ndarray | jnp.ndarray, Any]:
        trajectory_site_indices = self._trajectory_site_indices_for_backend(backend)

        observation = self._tracking_observation(
            env,
            data,
            carry,
            backend,
            trajectory_site_indices,
        )
        if self.visualize_goal:
            carry = self.set_visuals(env, model, data, carry, backend)
        return observation, carry

    def _tracking_observation(
        self,
        env: Any,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
        trajectory_site_indices: Any,
    ) -> np.ndarray | jnp.ndarray:
        traj_state = carry.traj_state
        root_qpos_indices = self._root_qpos_indices
        root_qvel_indices = self._root_qvel_ind

        current_root_qpos = data.qpos[root_qpos_indices]
        current_root_position = current_root_qpos[:3]
        current_root_rotation = rotation_matrix_from_quaternion(current_root_qpos[3:7], backend)
        heading_rotation = heading_rotation_matrix(current_root_rotation, backend)
        current_root_qvel = data.qvel[root_qvel_indices]
        observation_dtype = backend.asarray(current_root_position).dtype

        site_ids = self._rel_site_ids
        site_body_ids = self._site_bodyid[site_ids]
        current_site_kinematics = site_kinematics(
            data,
            site_ids,
            site_body_ids,
            self._body_rootid,
            backend,
        )

        initial_reference = env.th.get_traj_data_at(
            traj_state.traj_no,
            traj_state.subtraj_step_no_init,
            carry,
            backend,
        )
        initial_reference_xy = backend.asarray(
            initial_reference.qpos[root_qpos_indices[:2]],
            dtype=observation_dtype,
        )
        position_offset = backend.concatenate([initial_reference_xy, backend.zeros(1, dtype=observation_dtype)])
        trajectory_length = env.th.len_trajectory(traj_state.traj_no)

        components = []
        for horizon_index, step_offset in enumerate(self.lookahead_steps):
            target_step, is_valid = self._bounded_target_step(
                traj_state.subtraj_step_no,
                step_offset,
                trajectory_length,
                backend,
            )

            target_data = env.th.get_traj_data_at(
                traj_state.traj_no,
                target_step,
                carry,
                backend,
            )
            target_root_qpos = target_data.qpos[root_qpos_indices]
            target_root_position = target_root_qpos[:3] - position_offset
            target_root_rotation = rotation_matrix_from_quaternion(target_root_qpos[3:7], backend)
            target_root_qvel = target_data.qvel[root_qvel_indices]
            target_site_kinematics = None
            target_support_features = None
            if horizon_index == 0 or self.include_support_intent:
                target_site_kinematics = site_kinematics(
                    target_data,
                    site_ids,
                    site_body_ids,
                    self._body_rootid,
                    backend,
                    trajectory_site_indices=trajectory_site_indices,
                )
            if self.include_support_intent:
                target_support_features = reference_support_features(
                    target_site_kinematics[0],
                    target_site_kinematics[2],
                    self._support_site_indices,
                    backend.asarray(self._floor_height, dtype=observation_dtype),
                    backend,
                )

            if horizon_index == 0:
                components.extend(
                    [
                        root_tracking_error_features(
                            current_root_position,
                            current_root_rotation,
                            current_root_qvel,
                            target_root_position,
                            target_root_rotation,
                            target_root_qvel,
                            heading_rotation,
                            backend,
                        ),
                        site_tracking_error_features(
                            *current_site_kinematics,
                            *target_site_kinematics,
                            heading_rotation,
                            backend,
                        ),
                    ]
                )
                if target_support_features is not None:
                    components.append(target_support_features)
                continue

            target_site_positions = (
                target_site_kinematics[0]
                if target_site_kinematics is not None
                else site_positions(target_data, site_ids, trajectory_site_indices)
            )
            components.append(
                future_intent_features(
                    current_root_position,
                    current_root_rotation,
                    target_root_position,
                    target_root_rotation,
                    target_root_qvel,
                    target_site_positions,
                    target_support_features,
                    heading_rotation,
                    backend.asarray(step_offset * env.dt, dtype=observation_dtype),
                    backend.asarray(is_valid, dtype=observation_dtype),
                    backend,
                )
            )

        return backend.concatenate(components)


class GoalFullBodyTrackingVisual(GoalFullBodyTracking, GoalTrajMimicv2):
    """Full-body tracking goal with enhanced ghost-robot visualization."""


def fullbody_visual_goal_type(goal_type: str) -> str:
    """Return the matching visualization goal for a full-body goal type."""
    if goal_type in {GoalFullBodyTracking.__name__, GoalFullBodyTrackingVisual.__name__}:
        return GoalFullBodyTrackingVisual.__name__
    if goal_type in {GoalTrajMimic.__name__, GoalTrajMimicv2.__name__}:
        return GoalTrajMimicv2.__name__
    raise ValueError(f"Unsupported full-body goal type: {goal_type!r}")
