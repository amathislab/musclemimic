"""
Trajectory mimic goals for imitation learning.
"""

import logging
from copy import deepcopy
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np
from jax.scipy.spatial.transform import Rotation as jnp_R
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.observations.goals import Goal
from loco_mujoco.core.utils.math import calculate_relative_site_quantities
from loco_mujoco.core.utils.mujoco import mj_jntid2qposid, mj_jntid2qvelid
from musclemimic.core.utils.site_mapping import create_site_mapper

logger = logging.getLogger(__name__)


class GoalTrajMimic(Goal):
    """
    A class representing a trajectory goal in keypoint space (defined by sites) and joint properties.
    All entities are relative to the root body. This is the typical goal to be used with a DeepMimic-style reward.

    Args:
        info_props (Dict): Information properties required for initialization.
        rel_body_names (List[str]): List of relevant body names. Defaults to None.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, info_props: dict, rel_body_names: list[str] = None, **kwargs):
        self.n_step_lookahead = int(kwargs.pop("n_step_lookahead", info_props.get("n_step_lookahead", 1)))
        if self.n_step_lookahead < 1:
            raise ValueError(f"n_step_lookahead must be >= 1, got {self.n_step_lookahead}")
        self.n_step_stride = int(kwargs.pop("n_step_stride", info_props.get("n_step_stride", 1)))
        if self.n_step_stride < 1:
            raise ValueError(f"n_step_stride must be >= 1, got {self.n_step_stride}")
        # Motion phase observation: normalized progress through trajectory [0, 1]
        self.enable_motion_phase = bool(
            kwargs.pop("enable_motion_phase", info_props.get("enable_motion_phase", True))
        )
        # Concise lookahead: use root pos/vel deltas + site_rpos only (smaller observation)
        self.use_concise_lookahead = bool(
            kwargs.pop("use_concise_lookahead", info_props.get("use_concise_lookahead", False))
        )
        # Whether to include current-sim mimic site relative positions in the goal observation
        self.enable_mimic_site_rpos_observations = bool(
            kwargs.pop("enable_mimic_site_rpos_observations", info_props.get("enable_mimic_site_rpos_observations", True))
        )
        self.enable_global_root_tracking_observations = bool(
            kwargs.pop(
                "enable_global_root_tracking_observations",
                info_props.get("enable_global_root_tracking_observations", False),
            )
        )
        visualize_goal = bool(kwargs.get("visualize_goal", False))
        n_visual_geoms = int(
            kwargs.pop(
                "n_visual_geoms",
                len(info_props["sites_for_mimic"]) if visualize_goal else 0,
            )
        )
        if n_visual_geoms < 0:
            raise ValueError(f"n_visual_geoms must be >= 0, got {n_visual_geoms}")
        super().__init__(info_props, n_visual_geoms=n_visual_geoms, **kwargs)

        self.main_body_name = self._info_props["upper_body_xml_name"]
        self._qpos_ind = None
        self._qvel_ind = None
        self._root_qpos_ind = None  # Full root qpos indices (xyz + quat)
        self._root_qpos_full_ind = None
        self._root_qvel_ind = None  # Full root qvel indices (linear + angular vel)
        self._size_additional_observation = None

        # To be initialized
        self._relevant_body_names = [] if rel_body_names is None else rel_body_names
        self._relevant_body_ids = []
        self._rel_site_ids = []
        self._body_rootid = None
        self._site_bodyid = None
        self._dim = None

    def _init_from_mj(self, env: Any, model: MjModel | Model, data: MjData | Data, current_obs_size: int):
        """
        Initialize the goal from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """

        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name in self._relevant_body_names and body_name != self.main_body_name and body_name != "world":
                self._relevant_body_ids.append(i)

        for name in self._info_props["sites_for_mimic"]:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            self._rel_site_ids.append(site_id)

        self._rel_site_ids = np.array(self._rel_site_ids)
        self._body_rootid = model.body_rootid
        self._site_bodyid = model.site_bodyid

        root_free_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, env.root_free_joint_xml_name)
        self._qpos_ind = np.concatenate(
            [mj_jntid2qposid(i, model)[2:] for i in range(model.njnt) if i == root_free_joint_id]
            + [mj_jntid2qposid(i, model) for i in range(model.njnt) if i != root_free_joint_id]
        )
        self._qvel_ind = np.concatenate([mj_jntid2qvelid(i, model) for i in range(model.njnt)])

        # Store full root indices for simplified lookahead (root pos xyz + root vel 6D)
        self._root_qpos_full_ind = mj_jntid2qposid(root_free_joint_id, model)
        self._root_qpos_ind = self._root_qpos_full_ind[:3]  # xyz only
        self._root_qvel_ind = mj_jntid2qvelid(root_free_joint_id, model)  # 6D linear + angular vel

        n_relative_sites = len(self._info_props["sites_for_mimic"]) - 1
        size_for_sites_full = (3 + 3 + 6) * n_relative_sites  # rpos + rvel + rangles
        size_for_sites_rpos_only = 3 * n_relative_sites  # rpos only

        if self.use_concise_lookahead:
            # Simplified: root_pos_delta(3) + root_vel_delta(6) + site_rpos only
            # Note: step 0 has no delta (it's the reference), so we have n_step_lookahead-1 delta steps
            # But we still include step 0's site_rpos as the baseline
            traj_step_dim = 3 + 6 + size_for_sites_rpos_only  # per future step: root_pos_delta + root_vel_delta + site_rpos
            # step 0 just has site_rpos (no delta since it's the reference)
            self._dim = size_for_sites_rpos_only + traj_step_dim * (self.n_step_lookahead - 1) if self.n_step_lookahead > 1 else size_for_sites_rpos_only
        else:
            # Original: qpos + qvel + site_rpos + site_rangles + site_rvel per step
            traj_step_dim = len(self._qpos_ind) + len(self._qvel_ind) + size_for_sites_full
            self._dim = traj_step_dim * self.n_step_lookahead

        motion_phase_dim = 1 if self.enable_motion_phase else 0
        global_root_dim = 13 if self.enable_global_root_tracking_observations else 0
        self._dim += motion_phase_dim + global_root_dim
        if self.enable_mimic_site_rpos_observations:
            self._size_additional_observation = size_for_sites_full  # rpos + rvel + rangles
        else:
            self._size_additional_observation = (3 + 6) * n_relative_sites  # rvel + rangles only

        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.data_type_ind = np.array([i for i in range(data.userdata.size)])
        self.obs_ind = np.array([j for j in range(current_obs_size, current_obs_size + self.dim)])

        # Initialize site mapper for trajectory index mapping
        env_sites_for_mimic = getattr(env, "sites_for_mimic", [])
        self._site_mapper = create_site_mapper(model, env.__class__.__name__, env_sites_for_mimic)

        self._initialized_from_mj = True

    def init_from_traj(self, traj_handler: Any):
        """
        Initialize from a trajectory handler.

        Args:
            traj_handler (Any): The trajectory handler.
        """
        assert traj_handler is not None, (
            f"Trajectory handler is None, using {__class__.__name__} requires a trajectory."
        )
        # Attach actual trajectory site order to the site mapper once trajectory is loaded
        self._site_mapper.attach_trajectory_sites(traj_handler.traj.info.site_names)
        self._initialized_from_traj = True

    def get_obs_and_update_state(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[np.ndarray | jnp.ndarray, Any]:
        """
        Get the current goal observation and update the state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Goal observation and updated carry.
        """
        traj_state = carry.traj_state

        rel_site_ids = self._rel_site_ids
        rel_body_ids = self._site_bodyid[rel_site_ids]

        # Collect target trajectory observations for n-step lookahead.
        traj_site_indices = None
        if self._site_mapper.requires_mapping:
            traj_site_indices = self._site_mapper.model_ids_to_traj_indices(rel_site_ids)

        trajectory_length = env.th.len_trajectory(traj_state.traj_no)
        global_root_goal = None
        if self.enable_global_root_tracking_observations:
            current_ref = env.th.get_traj_data_at(
                traj_state.traj_no, traj_state.subtraj_step_no, carry, backend
            )
            root_qpos_full_ind = (
                backend.asarray(self._root_qpos_full_ind)
                if backend == jnp
                else self._root_qpos_full_ind
            )
            root_qvel_ind = (
                backend.asarray(self._root_qvel_ind)
                if backend == jnp
                else self._root_qvel_ind
            )
            global_root_goal = backend.concatenate(
                [current_ref.qpos[root_qpos_full_ind], current_ref.qvel[root_qvel_ind]]
            )

        if self.use_concise_lookahead:
            # Simplified lookahead: root_pos_delta + root_vel_delta + site_rpos only
            # Step 0 is reference (no delta), steps 1+ have deltas relative to step 0
            all_goal_site_rpos = []
            all_root_pos_delta = []
            all_root_vel_delta = []

            # Convert indices for JAX compatibility (JAX requires array indices, not lists)
            root_qpos_ind = backend.asarray(self._root_qpos_ind) if backend == jnp else self._root_qpos_ind
            root_qvel_ind = backend.asarray(self._root_qvel_ind) if backend == jnp else self._root_qvel_ind

            # Get reference trajectory data at current step (step 0)
            ref_traj_data = env.th.get_traj_data_at(
                traj_state.traj_no, traj_state.subtraj_step_no, carry, backend
            )
            ref_root_pos = ref_traj_data.qpos[root_qpos_ind]
            ref_root_vel = ref_traj_data.qvel[root_qvel_ind]

            for step_offset in range(self.n_step_lookahead):
                future_subtraj_step = traj_state.subtraj_step_no + step_offset * self.n_step_stride
                if backend == jnp:
                    future_subtraj_step = jnp.clip(future_subtraj_step, 0, trajectory_length - 1)
                else:
                    future_subtraj_step = max(0, min(future_subtraj_step, trajectory_length - 1))

                traj_data_single = env.th.get_traj_data_at(
                    traj_state.traj_no, future_subtraj_step, carry, backend
                )

                # Site relative positions only (no rangles/rvel)
                site_rpos, _, _ = calculate_relative_site_quantities(
                    traj_data_single,
                    rel_site_ids,
                    rel_body_ids,
                    self._body_rootid,
                    backend,
                    trajectory_site_indices=traj_site_indices if self._site_mapper.requires_mapping else None,
                )
                all_goal_site_rpos.append(site_rpos)

                # Root deltas for future steps (step 0 is reference, no delta)
                if step_offset > 0:
                    root_pos_delta = traj_data_single.qpos[root_qpos_ind] - ref_root_pos
                    root_vel_delta = traj_data_single.qvel[root_qvel_ind] - ref_root_vel
                    all_root_pos_delta.append(root_pos_delta)
                    all_root_vel_delta.append(root_vel_delta)

            # Build simplified trajectory goal observation
            # Structure: [step0_rpos, step1_pos_delta, step1_vel_delta, step1_rpos, ...]
            traj_goal_components = [backend.ravel(all_goal_site_rpos[0])]  # Step 0 site_rpos
            for i in range(len(all_root_pos_delta)):
                traj_goal_components.append(all_root_pos_delta[i])
                traj_goal_components.append(all_root_vel_delta[i])
                traj_goal_components.append(backend.ravel(all_goal_site_rpos[i + 1]))

            traj_goal_obs = backend.concatenate(traj_goal_components) if len(traj_goal_components) > 1 else traj_goal_components[0]

        else:
            # Original full lookahead: qpos + qvel + site_rpos + site_rangles + site_rvel
            all_goal_qpos = []
            all_goal_qvel = []
            all_goal_site_rpos = []
            all_goal_site_rangles = []
            all_goal_site_rvel = []

            for step_offset in range(self.n_step_lookahead):
                future_subtraj_step = traj_state.subtraj_step_no + step_offset * self.n_step_stride
                if backend == jnp:
                    future_subtraj_step = jnp.clip(future_subtraj_step, 0, trajectory_length - 1)
                else:
                    future_subtraj_step = max(0, min(future_subtraj_step, trajectory_length - 1))

                traj_data_single = env.th.get_traj_data_at(
                    traj_state.traj_no, future_subtraj_step, carry, backend
                )

                qpos_traj = traj_data_single.qpos
                qvel_traj = traj_data_single.qvel

                site_rpos, site_rangles, site_rvel = calculate_relative_site_quantities(
                    traj_data_single,
                    rel_site_ids,
                    rel_body_ids,
                    self._body_rootid,
                    backend,
                    trajectory_site_indices=traj_site_indices if self._site_mapper.requires_mapping else None,
                )

                all_goal_qpos.append(qpos_traj[self._qpos_ind])
                all_goal_qvel.append(qvel_traj[self._qvel_ind])
                all_goal_site_rpos.append(site_rpos)
                all_goal_site_rangles.append(site_rangles)
                all_goal_site_rvel.append(site_rvel)

            concatenated_qpos = backend.concatenate(all_goal_qpos)
            concatenated_qvel = backend.concatenate(all_goal_qvel)
            concatenated_site_rpos = backend.concatenate(all_goal_site_rpos)
            concatenated_site_rangles = backend.concatenate(all_goal_site_rangles)
            concatenated_site_rvel = backend.concatenate(all_goal_site_rvel)

            traj_goal_obs = backend.concatenate(
                [
                    concatenated_qpos,
                    concatenated_qvel,
                    backend.ravel(concatenated_site_rpos),
                    backend.ravel(concatenated_site_rangles),
                    backend.ravel(concatenated_site_rvel),
                ]
            )

        if self.visualize_goal:
            carry = self.set_visuals(env, model, data, carry, backend)

        if len(self._rel_site_ids) > 0:
            rel_site_ids = self._rel_site_ids
            rel_body_ids = self._site_bodyid[rel_site_ids]
            site_rpos, site_rangles, site_rvel = calculate_relative_site_quantities(
                data, rel_site_ids, rel_body_ids, self._body_rootid, backend
            )

            goal_components = []
            if self.enable_mimic_site_rpos_observations:
                goal_components.append(backend.ravel(site_rpos))
            goal_components.extend([
                backend.ravel(site_rangles),
                backend.ravel(site_rvel),
            ])
            if global_root_goal is not None:
                goal_components.append(global_root_goal)
            goal_components.append(backend.ravel(traj_goal_obs))

            if self.enable_motion_phase:
                # Motion phase: normalized progress through trajectory [0, 1]
                motion_phase = traj_state.subtraj_step_no / backend.maximum(trajectory_length, 1)
                goal_components.append(backend.atleast_1d(motion_phase))

            goal = backend.concatenate(goal_components)
            return goal, carry
        else:
            goal_components = []
            if global_root_goal is not None:
                goal_components.append(global_root_goal)
            goal_components.append(traj_goal_obs)
            if self.enable_motion_phase:
                motion_phase = traj_state.subtraj_step_no / backend.maximum(trajectory_length, 1)
                goal_components.append(backend.atleast_1d(motion_phase))
            return backend.concatenate(goal_components), carry

    @property
    def has_visual(self) -> bool:
        """Check if the goal supports visualization."""
        return True

    @property
    def requires_trajectory(self) -> bool:
        """Check if the goal requires a trajectory."""
        return True

    def set_visuals(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> Any:
        """
        Set the visualizations for the goal.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Any: Updated carry with visualizations set.
        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        traj_state = carry.traj_state
        user_scene = carry.user_scene
        goal_geoms = user_scene.geoms

        traj_data_single = env.th.get_current_traj_data(carry, backend)
        site_xpos = traj_data_single.site_xpos
        site_xmat = traj_data_single.site_xmat

        # Apply site mapping for visualization (same pattern as goal observations, rewards, etc.)
        if self._site_mapper.requires_mapping:
            traj_indices = self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
            # Convert trajectory indices to backend-compatible array
            if backend == jnp:
                vis_indices = jnp.array(traj_indices)
            else:
                vis_indices = np.array(traj_indices)
        else:
            # Use model site IDs directly when no mapping is required
            if backend == jnp:
                vis_indices = jnp.array(self._rel_site_ids)
            else:
                vis_indices = np.array(self._rel_site_ids)

        traj_data_init = env.th.get_init_traj_data(carry, backend)
        qpos_init = traj_data_init.qpos
        type = backend.full(self.n_visual_geoms, int(mujoco.mjtGeom.mjGEOM_BOX), dtype=backend.int32).reshape((-1, 1))
        size = backend.tile(backend.array([0.075, 0.05, 0.025]), (self.n_visual_geoms, 1))
        color = backend.tile(backend.array([0.0, 1.0, 0.0, 1.0]), (self.n_visual_geoms, 1))
        if backend == jnp:
            geom_pos = user_scene.geoms.pos.at[self.visual_geoms_idx].set(site_xpos[vis_indices])
            geom_mat = user_scene.geoms.mat.at[self.visual_geoms_idx].set(site_xmat[vis_indices])
            geom_type = user_scene.geoms.type.at[self.visual_geoms_idx].set(type)
            geom_size = user_scene.geoms.size.at[self.visual_geoms_idx].set(size)
            geom_rgba = user_scene.geoms.rgba.at[self.visual_geoms_idx].set(color)
            geom_pos = geom_pos.at[self.visual_geoms_idx, :2].add(-qpos_init[:2])

            new_user_scene = user_scene.replace(
                geoms=user_scene.geoms.replace(
                    pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba
                )
            )
            carry = carry.replace(user_scene=new_user_scene)
        else:
            user_scene.geoms.pos[self.visual_geoms_idx] = site_xpos[vis_indices]
            user_scene.geoms.mat[self.visual_geoms_idx] = site_xmat[vis_indices]
            user_scene.geoms.type[self.visual_geoms_idx] = type
            user_scene.geoms.size[self.visual_geoms_idx] = size
            user_scene.geoms.rgba[self.visual_geoms_idx] = color
            user_scene.geoms.pos[self.visual_geoms_idx, :2] -= qpos_init[:2]
            carry = carry.replace(user_scene=user_scene)

        return carry

    @property
    def dim(self) -> int:
        """Get the dimension of the goal."""
        return self._dim + self._size_additional_observation


class GoalTrajMimicv2(GoalTrajMimic):
    """
    Equivalent to GoalTrajMimic but with the ability to visualize the goal with the robot's geoms/body.

    ..note:: This class might slows down the simulation. Use it for visualization purposes only.

    Args:
        info_props (Dict): Information properties required for initialization.
        rel_body_names (List[str]): List of relevant body names. Defaults to None.
        target_geom_rgba (Tuple[float, float, float, float]): RGBA values for the target geom.
        Defaults to (0.471, 0.38, 0.812, 0.5).

    """

    def __init__(
        self,
        info_props: dict,
        rel_body_names: list[str] = None,
        target_geom_rgba: tuple[float, float, float, float] = (0.471, 0.38, 0.812, 0.5),
        **kwargs,
    ):
        self._geom_group_to_include = 0
        self._geom_ids_to_exclude = (0,)  # worldbody
        self._target_geom_rgba = target_geom_rgba
        self._enable_enhanced_vis = kwargs.pop("enable_enhanced_visualization", True)
        if self._enable_enhanced_vis and kwargs.get("visualize_goal", False):
            kwargs["n_visual_geoms"] = 0
        super().__init__(info_props, rel_body_names=rel_body_names, **kwargs)

        self._geom_ids = None
        self._geom_bodyid = None
        self._geom_type = None
        self._geom_size = None
        self._geom_rgba = None
        self._geom_dataid = None
        self._geom_group = None

    def _init_from_mj(self, env: Any, model: MjModel | Model, data: MjData | Data, current_obs_size: int):
        """
        Initialize the goal from Mujoco model and data.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """

        super()._init_from_mj(env, model, data, current_obs_size)

        if self.visualize_goal and self._enable_enhanced_vis:
            geom_ids = []
            geom_bodyid = []
            geom_type = []
            geom_size = []
            geom_rgba = []
            geom_dataid = []
            geom_group = []

            for i in range(model.ngeom):
                if i not in self._geom_ids_to_exclude and model.geom_group[i] == self._geom_group_to_include:
                    geom_ids.append(i)
                    geom_bodyid.append(model.geom_bodyid[i])
                    geom_type.append(model.geom_type[i])
                    geom_size.append(model.geom_size[i])
                    geom_rgba.append(self._target_geom_rgba)
                    geom_dataid.append(model.geom_dataid[i])
                    geom_group.append(model.geom_group[i])

            self._geom_ids = np.array(geom_ids)
            self._geom_bodyid = np.array(geom_bodyid)
            self._geom_type = np.array(geom_type).reshape(-1, 1)
            self._geom_size = np.array(geom_size)
            self._geom_rgba = np.array(geom_rgba)
            self._geom_dataid = np.array(geom_dataid).reshape(-1, 1)
            self._geom_group = np.array(geom_group).reshape(-1, 1)
            self.n_visual_geoms = len(self._geom_ids)

    def set_visuals(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> Any:
        """
        Set the visualizations for the goal.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Any: Updated carry with visualizations set.
        """
        if not self.visualize_goal:
            return carry

        if not self._enable_enhanced_vis:
            return super().set_visuals(env, model, data, carry, backend)

        if self._geom_ids is None or self.visual_geoms_idx is None:
            return carry

        if len(self.visual_geoms_idx) != len(self._geom_ids):
            logger.warning(
                "Ghost robot geom count mismatch (slots=%s, geoms=%s); skipping visualization.",
                len(self.visual_geoms_idx),
                len(self._geom_ids),
            )
            return carry

        user_scene = carry.user_scene

        traj_data_init = env.th.get_init_traj_data(carry, backend)
        traj_data_single = env.th.get_current_traj_data(carry, backend)
        qpos_init = traj_data_init.qpos
        qpos = traj_data_single.qpos
        qvel = traj_data_single.qvel

        if backend == jnp:
            visual_slots = jnp.asarray(self.visual_geoms_idx).reshape(-1)
            qpos = qpos.at[:2].add(-qpos_init[:2])
            data = data.replace(qpos=qpos, qvel=qvel)
            data = mujoco.mjx.kinematics(env.sys, data)
            geom_pos = data.geom_xpos[self._geom_ids]
            geom_mat = data.geom_xmat[self._geom_ids]

            geom_pos = user_scene.geoms.pos.at[visual_slots].set(geom_pos)
            geom_mat = user_scene.geoms.mat.at[visual_slots].set(geom_mat.reshape(-1, 9))
            geom_type = user_scene.geoms.type.at[visual_slots].set(self._geom_type)
            geom_size = user_scene.geoms.size.at[visual_slots].set(self._geom_size)
            geom_rgba = user_scene.geoms.rgba.at[visual_slots].set(self._geom_rgba)
            geom_data = user_scene.geoms.dataid.at[visual_slots].set(self._geom_dataid)
        else:
            visual_slots = np.asarray(self.visual_geoms_idx, dtype=np.intp).reshape(-1)
            data = deepcopy(data)
            data.qpos = qpos
            data.qpos[:2] -= qpos_init[:2]
            data.qvel = qvel
            mujoco.mj_kinematics(model, data)
            geom_pos = data.geom_xpos[self._geom_ids]
            geom_mat = data.geom_xmat[self._geom_ids]

            user_scene.geoms.pos[visual_slots] = geom_pos
            user_scene.geoms.mat[visual_slots] = geom_mat.reshape(-1, 9)
            user_scene.geoms.type[visual_slots] = self._geom_type
            user_scene.geoms.size[visual_slots] = self._geom_size
            user_scene.geoms.rgba[visual_slots] = self._geom_rgba
            user_scene.geoms.dataid[visual_slots] = self._geom_dataid
            carry = carry.replace(user_scene=user_scene)

        if backend == jnp:
            new_user_scene = user_scene.replace(
                geoms=user_scene.geoms.replace(
                    pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba, dataid=geom_data
                )
            )
            carry = carry.replace(user_scene=new_user_scene)

        return carry
