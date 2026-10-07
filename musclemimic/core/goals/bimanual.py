"""
Goal observations for bimanual trajectory imitation tasks.
"""

import logging
from copy import deepcopy
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.observations.goals import Goal
from loco_mujoco.core.utils.math import calculate_relative_site_quantities
from loco_mujoco.core.utils.mujoco import mj_jntid2qposid, mj_jntid2qvelid
from musclemimic.core.utils.site_mapping import create_site_mapper

logger = logging.getLogger(__name__)


class GoalBimanualTrajMimic(Goal):
    """
    Goal observation builder for fixed-base bimanual imitation tasks.

    The observation combines:
    - current relative site features from the simulator, and
    - lookahead target data from the reference trajectory.
    """

    def __init__(self, info_props: dict[str, Any], **kwargs) -> None:
        """
        Initialize the bimanual trajectory mimic goal.

        Args:
            info_props (Dict[str, Any]): Goal metadata such as `sites_for_mimic`.
            **kwargs: Additional keyword arguments for visualization and other options.
        """
        # Extract goal-specific configuration before calling the parent constructor.
        self.n_step_lookahead = int(kwargs.pop("n_step_lookahead", info_props.get("n_step_lookahead", 1)))
        if self.n_step_lookahead < 1:
            raise ValueError(f"n_step_lookahead must be >= 1, got {self.n_step_lookahead}")
        self.n_step_stride = int(kwargs.pop("n_step_stride", info_props.get("n_step_stride", 1)))
        if self.n_step_stride < 1:
            raise ValueError(f"n_step_stride must be >= 1, got {self.n_step_stride}")
        # Normalized progress through the current trajectory.
        self.enable_motion_phase = bool(
            kwargs.pop("enable_motion_phase", info_props.get("enable_motion_phase", True))
        )
        # Optionally keep only qpos/qvel/site_rpos in the lookahead target.
        self.use_concise_lookahead = bool(
            kwargs.pop("use_concise_lookahead", info_props.get("use_concise_lookahead", False))
        )

        # Forward only parameters accepted by the observation base classes.
        valid_observation_params = {}
        if "group" in kwargs:
            valid_observation_params["group"] = kwargs.pop("group")
        if "allow_randomization" in kwargs:
            valid_observation_params["allow_randomization"] = kwargs.pop("allow_randomization")

        valid_goal_params = {}
        if "visualize_goal" in kwargs:
            valid_goal_params["visualize_goal"] = kwargs.pop("visualize_goal")
        if "n_visual_geoms" in kwargs:
            valid_goal_params["n_visual_geoms"] = kwargs.pop("n_visual_geoms")

        # Drop any remaining config-only keys so they do not leak into the base class.
        kwargs.clear()

        # Default visualization uses one marker per tracked site.
        if "n_visual_geoms" in valid_goal_params:
            n_visual_geoms = valid_goal_params["n_visual_geoms"]
        else:
            n_visual_geoms = len(info_props["sites_for_mimic"]) if valid_goal_params.get("visualize_goal", False) else 0
        super().__init__(
            info_props,
            n_visual_geoms=n_visual_geoms,
            visualize_goal=valid_goal_params.get("visualize_goal", False),
            **valid_observation_params,
        )

        self.main_body_name = info_props["upper_body_xml_name"]
        self._relevant_body_names = []
        self._relevant_body_ids = []
        self._rel_site_ids = []
        self._rel_body_ids = []
        self._traj_site_indices = None
        self._initialized_from_mj = False
        self._initialized_from_traj = False
        self._warned_missing_traj = False

    def _init_from_mj(self, env: Any, model: MjModel | Model, data: MjData | Data, current_obs_size: int):
        """
        Initialize model-dependent indices and observation dimensions.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            current_obs_size (int): Current observation size.
        """

        # Resolve mimic site ids in the model.
        for name in self._info_props["sites_for_mimic"]:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id == -1:
                raise ValueError(f"Site '{name}' not found in model")
            self._rel_site_ids.append(site_id)

        n_sites = len(self._info_props["sites_for_mimic"])
        n_relative_sites = n_sites - 1

        self._rel_site_ids = np.array(self._rel_site_ids)
        self._body_rootid = model.body_rootid
        self._site_bodyid = model.site_bodyid
        self._rel_body_ids = self._site_bodyid[self._rel_site_ids]

        # Map model site ids into the trajectory cache layout when needed.
        env_sites_for_mimic = getattr(env, "sites_for_mimic", [])
        self._site_mapper = create_site_mapper(model, env.__class__.__name__, env_sites_for_mimic)

        self._qpos_ind = np.concatenate([mj_jntid2qposid(i, model) for i in range(model.njnt)])
        self._qvel_ind = np.concatenate([mj_jntid2qvelid(i, model) for i in range(model.njnt)])

        self._set_goal_dimensions(
            n_qpos=len(self._qpos_ind),
            n_qvel=len(self._qvel_ind),
            n_relative_sites=n_relative_sites,
        )

        # Sanity-check that the gathered qpos/qvel indices cover the full state.
        if len(self._qpos_ind) != model.nq:
            raise ValueError(f"Joint position indices size mismatch: {len(self._qpos_ind)} != {model.nq}")
        if len(self._qvel_ind) != model.nv:
            raise ValueError(f"Joint velocity indices size mismatch: {len(self._qvel_ind)} != {model.nv}")

        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.data_type_ind = np.arange(data.userdata.size)
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)

        self._initialized_from_mj = True

    def _set_goal_dimensions(
        self,
        *,
        n_qpos: int,
        n_qvel: int,
        n_relative_sites: int,
    ) -> None:
        """Set dimensions for the legacy full or concise trajectory encoding."""

        size_for_joint_pos = n_qpos * self.n_step_lookahead
        size_for_joint_vel = n_qvel * self.n_step_lookahead

        # Target trajectory site features exclude the root/reference site.
        if self.use_concise_lookahead:
            size_for_site_targets = 3 * n_relative_sites * self.n_step_lookahead
        else:
            size_for_site_targets = (3 + 3 + 6) * n_relative_sites * self.n_step_lookahead

        # Current simulator site features are always included in full.
        size_for_current_sites = (3 + 6 + 3) * n_relative_sites

        motion_phase_dim = 1 if self.enable_motion_phase else 0

        self._dim = (
            size_for_joint_pos
            + size_for_joint_vel
            + size_for_site_targets
            + size_for_current_sites
            + motion_phase_dim
        )
        self._size_additional_observation = 0

    def init_from_traj(self, traj_handler: Any):
        """
        Attach trajectory-specific indices after the trajectory handler is ready.

        Args:
            traj_handler: The trajectory handler containing trajectory data.
        """
        if not self._initialized_from_mj:
            raise RuntimeError("Must call _init_from_mj before init_from_traj")

        self._th = traj_handler
        # Trajectory caches may store sites in a different order than the model.
        self._site_mapper.attach_trajectory_sites(traj_handler.traj.info.site_names)
        self._traj_site_indices = (
            self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
            if self._site_mapper.requires_mapping
            else self._rel_site_ids
        )
        self._initialized_from_traj = True

    def _get_obs(self, model: MjModel | Model, data: MjData | Data, carry: Any) -> np.ndarray | jnp.ndarray:
        """
        Build a numpy goal observation for the current state.

        Args:
            model: The Mujoco model.
            data: The Mujoco data.
            carry: Additional carry information.

        Returns:
            Goal observation array.
        """
        if not self._initialized_from_traj:
            raise RuntimeError("Goal not properly initialized from trajectory")

        # Reference data for the current timestep.
        traj_data_single = self._th.get_current_traj_data(carry, np)
        target_qpos = traj_data_single.qpos
        target_qvel = traj_data_single.qvel

        goal_qpos = target_qpos[self._qpos_ind]
        goal_qvel = target_qvel[self._qvel_ind]

        # Resolve trajectory indices once and reuse them for site extraction.
        traj_site_indices = self._traj_site_indices
        if traj_site_indices is None and self._site_mapper.requires_mapping:
            traj_site_indices = self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
            self._traj_site_indices = traj_site_indices

        site_rpos_traj, site_rangles_traj, site_rvel_traj = calculate_relative_site_quantities(
            traj_data_single,
            self._rel_site_ids,
            self._rel_body_ids,
            self._body_rootid,
            np,
            trajectory_site_indices=traj_site_indices if self._site_mapper.requires_mapping else None,
        )

        # Combine current simulator features with target trajectory features.
        if len(self._rel_site_ids) > 0:
            site_rpos, site_rangles, site_rvel = calculate_relative_site_quantities(
                data, self._rel_site_ids, self._rel_body_ids, self._body_rootid, np
            )

            if self.use_concise_lookahead:
                traj_goal_obs = np.concatenate(
                    [
                        goal_qpos,
                        goal_qvel,
                        np.ravel(site_rpos_traj),
                    ]
                )
            else:
                # Full: qpos + qvel + site_rpos + site_rangles + site_rvel
                traj_goal_obs = np.concatenate(
                    [
                        goal_qpos,
                        goal_qvel,
                        np.ravel(site_rpos_traj),
                        np.ravel(site_rangles_traj),
                        np.ravel(site_rvel_traj),
                    ]
                )

            goal_obs = np.concatenate(
                [
                    np.ravel(site_rpos),
                    np.ravel(site_rangles),
                    np.ravel(site_rvel),
                    np.ravel(traj_goal_obs),
                ]
            )
        else:
            raise ValueError("GoalBimanualTrajMimic requires sites_for_mimic to be configured")

        return goal_obs

    def get_observation(self) -> np.ndarray | jnp.ndarray:
        """
        Compatibility helper returning a zero-filled observation buffer.

        Returns:
            Zero-filled goal observation array.
        """
        return np.zeros(self._dim)

    @property
    def dim(self) -> int:
        """Get the dimension of the goal observation."""
        return self._dim

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
        Render marker boxes at the current reference trajectory sites.
        """
        if not self.visualize_goal or self.n_visual_geoms == 0 or self.visual_geoms_idx is None:
            return carry

        user_scene = carry.user_scene
        traj_data_single = env.th.get_current_traj_data(carry, backend)
        site_xpos = traj_data_single.site_xpos
        site_xmat = traj_data_single.site_xmat

        if self._site_mapper.requires_mapping:
            traj_site_indices = self._traj_site_indices
            if traj_site_indices is None:
                traj_site_indices = self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
                self._traj_site_indices = traj_site_indices
            vis_indices = backend.asarray(traj_site_indices) if backend == jnp else np.asarray(traj_site_indices)
        else:
            vis_indices = backend.asarray(self._rel_site_ids) if backend == jnp else np.asarray(self._rel_site_ids)

        geom_type = backend.full(
            self.n_visual_geoms, int(mujoco.mjtGeom.mjGEOM_BOX), dtype=backend.int32
        ).reshape((-1, 1))
        geom_size = backend.tile(backend.array([0.075, 0.05, 0.025]), (self.n_visual_geoms, 1))
        geom_rgba = backend.tile(backend.array([0.0, 1.0, 0.0, 1.0]), (self.n_visual_geoms, 1))

        if backend == jnp:
            geom_pos = user_scene.geoms.pos.at[self.visual_geoms_idx].set(site_xpos[vis_indices])
            geom_mat = user_scene.geoms.mat.at[self.visual_geoms_idx].set(site_xmat[vis_indices])
            geom_type = user_scene.geoms.type.at[self.visual_geoms_idx].set(geom_type)
            geom_size = user_scene.geoms.size.at[self.visual_geoms_idx].set(geom_size)
            geom_rgba = user_scene.geoms.rgba.at[self.visual_geoms_idx].set(geom_rgba)

            new_user_scene = user_scene.replace(
                geoms=user_scene.geoms.replace(
                    pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba
                )
            )
            return carry.replace(user_scene=new_user_scene)

        user_scene.geoms.pos[self.visual_geoms_idx] = site_xpos[vis_indices]
        user_scene.geoms.mat[self.visual_geoms_idx] = site_xmat[vis_indices]
        user_scene.geoms.type[self.visual_geoms_idx] = geom_type
        user_scene.geoms.size[self.visual_geoms_idx] = geom_size
        user_scene.geoms.rgba[self.visual_geoms_idx] = geom_rgba
        return carry.replace(user_scene=user_scene)

    @property
    def initialized(self) -> bool:
        """Check if the goal is initialized.

        This only reflects model-side initialization. Trajectory data may still
        be attached later during environment setup.
        """
        return self._initialized_from_mj

    def get_obs_and_update_state(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> tuple[np.ndarray | jnp.ndarray, Any]:
        """
        Build the goal observation for the current step.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The Mujoco model.
            data (Union[MjData, Data]): The Mujoco data.
            carry (Any): Carry object.
            backend (ModuleType): The backend (numpy or jax).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: Goal observation and updated carry.
        """
        if not self._initialized_from_traj:
            # During environment construction the goal may be queried before the
            # trajectory handler is attached.
            if backend == np and not self._warned_missing_traj:
                logger.warning("Goal observation returning zeros - trajectory not initialized (dim: %s)", self._dim)
                self._warned_missing_traj = True
            return backend.zeros(self._dim), carry

        # Refresh trajectory-site metadata in case the active trajectory changes.
        if self._site_mapper.requires_mapping and hasattr(env, "th") and env.th is not None:
            self._site_mapper.attach_trajectory_sites(env.th.traj.info.site_names)
            if self._traj_site_indices is None:
                self._traj_site_indices = self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)

        rel_site_ids = self._rel_site_ids
        rel_body_ids = self._rel_body_ids

        # Gather multi-step lookahead target features.
        def safe_get_trajectory_data():
            try:
                traj_data = env.th.traj.data
                traj_state = carry.traj_state

                all_goal_qpos = []
                all_goal_qvel = []
                all_goal_site_rpos = []
                all_goal_site_rangles = []
                all_goal_site_rvel = []

                # Reuse cached trajectory indices for the tracked mimic sites.
                traj_site_indices = self._traj_site_indices
                if self._site_mapper.requires_mapping and traj_site_indices is None:
                    traj_site_indices = self._site_mapper.model_ids_to_traj_indices(rel_site_ids)
                    self._traj_site_indices = traj_site_indices

                for step_offset in range(self.n_step_lookahead):
                    future_subtraj_step = traj_state.subtraj_step_no + step_offset * self.n_step_stride
                    future_traj_no = traj_state.traj_no

                    # Clamp lookahead to the active trajectory bounds.
                    trajectory_length = env.th.len_trajectory(traj_state.traj_no)

                    if backend == jnp:
                        future_subtraj_step = jnp.clip(future_subtraj_step, 0, trajectory_length - 1)
                    else:
                        future_subtraj_step = max(0, min(future_subtraj_step, trajectory_length - 1))

                    traj_data_single = traj_data.get(future_traj_no, future_subtraj_step, backend)

                    target_qpos = traj_data_single.qpos
                    target_qvel = traj_data_single.qvel

                    goal_qpos = target_qpos[self._qpos_ind]
                    goal_qvel = target_qvel[self._qvel_ind]

                    site_rpos_traj, site_rangles_traj, site_rvel_traj = calculate_relative_site_quantities(
                        traj_data_single,
                        rel_site_ids,
                        rel_body_ids,
                        self._body_rootid,
                        backend,
                        trajectory_site_indices=traj_site_indices if self._site_mapper.requires_mapping else None,
                    )

                    all_goal_qpos.append(goal_qpos)
                    all_goal_qvel.append(goal_qvel)
                    all_goal_site_rpos.append(site_rpos_traj)
                    if not self.use_concise_lookahead:
                        all_goal_site_rangles.append(site_rangles_traj)
                        all_goal_site_rvel.append(site_rvel_traj)

                concatenated_qpos = backend.concatenate(all_goal_qpos)
                concatenated_qvel = backend.concatenate(all_goal_qvel)
                concatenated_site_rpos = backend.concatenate(all_goal_site_rpos)

                if self.use_concise_lookahead:
                    traj_goal_obs = backend.concatenate(
                        [
                            concatenated_qpos,
                            concatenated_qvel,
                            backend.ravel(concatenated_site_rpos),
                        ]
                    )
                else:
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

                return traj_goal_obs, True

            except Exception as e:
                raise RuntimeError(f"Failed to get trajectory data: {e}")

        traj_goal_obs, _ = safe_get_trajectory_data()

        if self.visualize_goal:
            carry = self.set_visuals(env, model, data, carry, backend)

        if len(self._rel_site_ids) > 0:
            site_rpos, site_rangles, site_rvel = calculate_relative_site_quantities(
                data, rel_site_ids, rel_body_ids, self._body_rootid, backend
            )

            goal_components = [
                backend.ravel(site_rpos),
                backend.ravel(site_rangles),
                backend.ravel(site_rvel),
                backend.ravel(traj_goal_obs),
            ]

            if self.enable_motion_phase:
                traj_state = carry.traj_state
                trajectory_length = env.th.len_trajectory(traj_state.traj_no)
                motion_phase = traj_state.subtraj_step_no / backend.maximum(trajectory_length, 1)
                goal_components.append(backend.atleast_1d(motion_phase))

            goal = backend.concatenate(goal_components)
            return goal, carry
        else:
            goal_components = [traj_goal_obs]
            if self.enable_motion_phase:
                traj_state = carry.traj_state
                trajectory_length = env.th.len_trajectory(traj_state.traj_no)
                motion_phase = traj_state.subtraj_step_no / backend.maximum(trajectory_length, 1)
                goal_components.append(backend.atleast_1d(motion_phase))
            return backend.concatenate(goal_components), carry

    def reset(self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any) -> Any:
        """
        Reset the goal for a new episode.

        Args:
            env: The environment instance.
            model: The Mujoco model.
            data: The Mujoco data.
            carry: Additional carry information.

        Returns:
            Updated carry information.
        """
        return carry


class GoalBimanualTrajMimicv2(GoalBimanualTrajMimic):
    """
    Bimanual trajectory goal with full-robot target visualization.

    Args:
        info_props (Dict[str, Any]): Goal information containing sites_for_mimic and other parameters.
        target_geom_rgba (Tuple[float, float, float, float]): RGBA values for target robot visualization.
            Defaults to a semi-transparent purple highlight.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        info_props: dict[str, Any],
        target_geom_rgba: tuple[float, float, float, float] = (0.471, 0.38, 0.812, 0.5),
        **kwargs,
    ):
        self._geom_group_to_include = 0
        self._geom_ids_to_exclude = (0,)
        self._target_geom_rgba = target_geom_rgba

        self._enable_enhanced_vis = kwargs.pop("enable_enhanced_visualization", True)
        if self._enable_enhanced_vis and kwargs.get("visualize_goal", False):
            kwargs["n_visual_geoms"] = 0
        super().__init__(info_props, **kwargs)

        # Enhanced visualization attributes (initialized in _init_from_mj)
        self._geom_ids = None
        self._geom_bodyid = None
        self._geom_type = None
        self._geom_size = None
        self._geom_rgba = None
        self._geom_dataid = None
        self._geom_group = None

    def _init_from_mj(self, env: Any, model: MjModel | Model, data: MjData | Data, current_obs_size: int):
        """
        Initialize model-dependent goal state and optional enhanced visualization.
        """
        super()._init_from_mj(env, model, data, current_obs_size)

        if self.visualize_goal and self._enable_enhanced_vis:
            self._setup_enhanced_visualization(model)
            self.n_visual_geoms = len(self._geom_ids)

    def _setup_enhanced_visualization(self, model: MjModel | Model):
        """
        Cache the geometry metadata used to render the target robot pose.
        """
        geom_ids = []
        geom_bodyid = []
        geom_type = []
        geom_size = []
        geom_rgba = []
        geom_dataid = []
        geom_group = []

        # Collect the geometries that belong to the rendered robot body.
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

    def set_visuals(
        self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType
    ) -> Any:
        """
        Render the target robot pose into the user scene.
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
        traj_data = env.th.get_current_traj_data(carry, backend)
        qpos = traj_data.qpos
        qvel = traj_data.qvel

        # Use the exact subset of visual slots associated with this goal.
        if backend == jnp:
            visual_slots = jnp.asarray(self.visual_geoms_idx).reshape(-1)
        else:
            visual_slots = np.asarray(self.visual_geoms_idx).reshape(-1)

        if backend == jnp:
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
            data = deepcopy(data)
            data.qpos = qpos
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
            geom_pos = user_scene.geoms.pos[visual_slots]
            geom_mat = user_scene.geoms.mat[visual_slots]
            geom_type = user_scene.geoms.type[visual_slots]
            geom_rgba = user_scene.geoms.rgba[visual_slots]
            geom_size = user_scene.geoms.size[visual_slots]
            geom_data = user_scene.geoms.dataid[visual_slots]

        # Keep the global scene layout unchanged and only update this goal's slots.
        if backend == jnp:
            new_user_scene = user_scene.replace(
                geoms=user_scene.geoms.replace(
                    pos=geom_pos, mat=geom_mat, size=geom_size, type=geom_type, rgba=geom_rgba, dataid=geom_data
                )
            )
            carry = carry.replace(user_scene=new_user_scene)
        else:
            carry = carry.replace(user_scene=user_scene)

        return carry
