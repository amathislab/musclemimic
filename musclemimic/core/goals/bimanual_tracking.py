"""Heading-local trajectory tracking goal for fixed-root bimanual tasks."""

import logging
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from musclemimic.core.goals.bimanual import GoalBimanualTrajMimic, GoalBimanualTrajMimicv2
from musclemimic.core.goals.tracking import DEFAULT_LOOKAHEAD_STEPS, TrackingGoalMixin
from musclemimic.core.goals.tracking_features import (
    fixed_root_tracking_goal_dimension,
    future_site_intent_features,
    site_kinematics,
    site_tracking_error_features,
)
from musclemimic.core.observations.egocentric import heading_rotation_matrix

logger = logging.getLogger(__name__)


class GoalBimanualTracking(TrackingGoalMixin, GoalBimanualTrajMimic):
    """Full next-frame site error followed by compact future site intent.

    The first configured site acts as the fixed-root anchor. Site positions,
    rotations, and velocities use the current anchor heading. Future horizons
    contain reference site positions plus time and validity metadata.
    """

    def __init__(
        self,
        info_props: dict[str, Any],
        lookahead_steps: Sequence[int] = DEFAULT_LOOKAHEAD_STEPS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            info_props,
            lookahead_steps=lookahead_steps,
            **kwargs,
        )

    def _set_goal_dimensions(
        self,
        *,
        n_qpos: int,
        n_qvel: int,
        n_relative_sites: int,
    ) -> None:
        del n_qpos, n_qvel
        self._dim = fixed_root_tracking_goal_dimension(
            n_relative_sites,
            len(self.lookahead_steps),
        )
        self._size_additional_observation = 0

    def get_obs_and_update_state(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ) -> tuple[np.ndarray | jnp.ndarray, Any]:
        if not self._initialized_from_traj:
            if backend is np and not self._warned_missing_traj:
                logger.warning(
                    "Tracking goal observation returning zeros because the trajectory is not initialized (dim: %s)",
                    self._dim,
                )
                self._warned_missing_traj = True
            return backend.zeros(self._dim), carry

        if self._site_mapper.requires_mapping and env.th is not None:
            self._site_mapper.attach_trajectory_sites(env.th.traj.info.site_names)
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
        site_ids = self._rel_site_ids
        current_site_kinematics = site_kinematics(
            data,
            site_ids,
            self._rel_body_ids,
            self._body_rootid,
            backend,
        )
        heading_rotation = heading_rotation_matrix(current_site_kinematics[1][0], backend)
        observation_dtype = backend.asarray(current_site_kinematics[0]).dtype
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
            target_site_kinematics = site_kinematics(
                target_data,
                site_ids,
                self._rel_body_ids,
                self._body_rootid,
                backend,
                trajectory_site_indices=trajectory_site_indices,
            )

            if horizon_index == 0:
                components.append(
                    site_tracking_error_features(
                        *current_site_kinematics,
                        *target_site_kinematics,
                        heading_rotation,
                        backend,
                    )
                )
                continue

            components.append(
                future_site_intent_features(
                    target_site_kinematics[0],
                    heading_rotation,
                    backend.asarray(step_offset * env.dt, dtype=observation_dtype),
                    backend.asarray(is_valid, dtype=observation_dtype),
                    backend,
                )
            )

        return backend.concatenate(components)


class GoalBimanualTrackingVisual(GoalBimanualTracking, GoalBimanualTrajMimicv2):
    """Bimanual tracking goal with enhanced ghost-robot visualization."""


def bimanual_visual_goal_type(goal_type: str) -> str:
    """Return the matching visualization goal for a bimanual goal type."""
    if goal_type in {GoalBimanualTracking.__name__, GoalBimanualTrackingVisual.__name__}:
        return GoalBimanualTrackingVisual.__name__
    if goal_type in {GoalBimanualTrajMimic.__name__, GoalBimanualTrajMimicv2.__name__}:
        return GoalBimanualTrajMimicv2.__name__
    raise ValueError(f"Unsupported bimanual goal type: {goal_type!r}")


__all__ = [
    "DEFAULT_LOOKAHEAD_STEPS",
    "GoalBimanualTracking",
    "GoalBimanualTrackingVisual",
    "bimanual_visual_goal_type",
]
