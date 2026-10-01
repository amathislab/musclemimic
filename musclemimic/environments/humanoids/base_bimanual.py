import warnings
from copy import deepcopy

import mujoco
import numpy as np
from mujoco import MjSpec

from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from loco_mujoco.smpl.retargeting import extend_motion
from loco_mujoco.trajectory import Trajectory, TrajectoryData, TrajectoryHandler, TrajectoryInfo, TrajectoryModel
from musclemimic.environments import LocoEnv
from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__, identifier="[Bimanual]")


class FixedRootEnv(LocoEnv):
    """
    Base class for bimanual environments without a root joint.
    Excludes specific info properties like root_height_healthy_range.
    """

    def _get_all_info_properties(self):
        """
        Collect all info properties, excluding specific ones for bimanual models.
        """
        excluded_props = {"root_height_healthy_range"}
        info_props = {}

        for attr_name in dir(self):
            if attr_name in excluded_props:
                continue  # Skip excluded properties for bimanual models
            attr_value = getattr(self.__class__, attr_name, None)
            if isinstance(attr_value, property) and getattr(attr_value.fget, "_is_info_property", False):
                info_props[attr_name] = deepcopy(getattr(self, attr_name))

        return info_props

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the root free joint.
        For bimanual models, this is a dummy property since there's no free joint.
        """
        return "none"


class BaseBimanualSkeleton(FixedRootEnv):
    """
    Mujoco environment of a base bimanual skeleton model.
    """

    mjx_enabled = False

    def __init__(
        self,
        use_muscles: bool = False,
        scaling: float = 1.0,
        enable_joint_pos_observations: bool = True,
        enable_joint_vel_observations: bool = True,
        spec: str | MjSpec | None = None,
        observation_spec: list[ObservationType] | None = None,
        actuation_spec: list[str] | None = None,
        goal_params: dict | None = None,
        reward_params: dict | None = None,
        mjx_backend: str = "jax",
        **kwargs,
    ) -> None:
        """
        Initializes the bimanual skeleton environment.

        Args:
            use_muscles (bool): If True, uses muscle actuators. Defaults to False.
            scaling (float): Global scaling factor for the robot. Defaults to 1.0.
            enable_joint_pos_observations (bool): If True (default), include joint positions in observations.
            enable_joint_vel_observations (bool): If True (default), include joint velocities in observations.
            spec (Union[str, MjSpec]): Path to the robot's xml file or MjSpec. If None, uses default.
            observation_spec (List[ObservationType]): Specification of observations. If None, uses default.
            actuation_spec (List[str]): Specification of actuators. If None, uses default.
            goal_params (Dict): Parameters for goal function configuration.
            reward_params (Dict): Parameters for reward function configuration.
            mjx_backend (str): Backend for MJX ('jax' or 'warp'). Defaults to 'jax'.
            **kwargs: Additional arguments passed to parent class.
        """

        # Store observation flags
        self._enable_joint_pos_observations = enable_joint_pos_observations
        self._enable_joint_vel_observations = enable_joint_vel_observations

        # Store goal and reward params for use in sites_for_mimic property
        self._init_goal_params = goal_params or {}
        self._init_reward_params = reward_params or {}

        # Store mjx_backend for use in _modify_spec_for_mjx
        self.mjx_backend = mjx_backend

        # Store nconmax and num_envs from kwargs for automatic scaling
        self.nconmax = kwargs.get("nconmax", None)
        self.num_envs = kwargs.get("num_envs", 1)

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # Apply spec changes to MjSpec
        spec = self._apply_spec_changes(spec)

        self.scaling = scaling
        if scaling != 1.0:
            spec = self.scale_body(spec, use_muscles)

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._use_muscles = use_muscles

        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(
            spec=spec,
            observation_spec=observation_spec,
            actuation_spec=actuation_spec,
            goal_params=goal_params,
            reward_params=reward_params,
            mjx_backend=mjx_backend,
            **kwargs,
        )

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        """
        Apply changes to the MjSpec before environment creation.

        By default, no changes are applied. Subclasses can override this method
        to modify the specification as needed.

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            MjSpec: The (potentially) modified specification
        """
        return spec

    def _get_spec_modifications(self) -> tuple[list[str], list[str], list[str]]:
        """
        Function that specifies which joints, motors, and equality constraints
        should be removed from the Mujoco specification.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
            and names of equality constraints to remove.
        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    def _get_observation_specification(self, spec: MjSpec) -> list[ObservationType]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            A list of observation types.
        """

        # Get all joint names (MyoBimanualArm has no root joint to exclude)
        j_names = [j.name for j in spec.joints]

        observation_spec = []

        # Add all joint positions dynamically (no root joint for MyoBimanualArm)
        if self._enable_joint_pos_observations:
            observation_spec.append(ObservationType.JointPosArray("q_all_pos", j_names))

        # Add all joint velocities dynamically (no root joint for MyoBimanualArm)
        if self._enable_joint_vel_observations:
            observation_spec.append(ObservationType.JointVelArray("dq_all_vel", j_names))

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> list[str]:
        """
        Returns the action specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: A list of actuator names.
        """
        action_spec = []
        for a in spec.actuators:
            action_spec.append(a.name)
        return action_spec

    def scale_body(self, mjspec: MjSpec, use_muscles: bool) -> MjSpec:
        """
        This function scales the kinematics and dynamics of the humanoid model given a Mujoco XML handle.

        Args:
            mjspec (MjSpec): Handle to Mujoco specification.
            use_muscles (bool): If True, muscle actuators will be scaled, else torque actuators will be scaled.

        Returns:
            Modified Mujoco XML handle.
        """

        body_scaling = self.scaling
        head_geoms = ["hat_skull", "hat_jaw", "hat_ribs_cap"]

        # scale meshes
        for mesh in mjspec.meshes:
            if mesh.name not in head_geoms:  # don't scale head
                mesh.scale *= body_scaling

        # change position of head
        for geom in mjspec.geoms:
            if geom.name in head_geoms:
                geom.pos = [0.0, -0.5 * (1 - body_scaling), 0.0]

        # scale bodies
        for body in mjspec.bodies:
            body.pos *= body_scaling
            body.mass *= body_scaling**3
            body.fullinertia *= body_scaling**5
            assert np.array_equal(body.fullinertia[3:], np.zeros(3)), (
                "Some of the diagonal elements of the"
                "inertia matrix are not zero! Scaling is"
                "not done correctly. Double-Check!"
            )

        # scale actuators
        if use_muscles:
            for site in mjspec.sites:
                site.pos *= body_scaling

            for actuator in mjspec.actuators:
                if "mot" not in actuator.name:
                    actuator.force *= body_scaling**2
                else:
                    actuator.gear *= body_scaling**2
        else:
            for actuator in mjspec.actuators:
                actuator.gear *= body_scaling**2

        return mjspec

    def load_trajectory(
        self,
        traj: Trajectory | None = None,
        traj_path: str | None = None,
        warn: bool = True,
        cache_type="full",
        site_names=None,
    ) -> None:
        """
        Loads trajectories. If there were trajectories loaded already, this function overrides the latter.

        Args:
            traj (Trajectory): Datastructure containing all trajectory files. If traj_path is specified, this
                should be None.
            traj_path (string): Path with the trajectory for the model to follow. Should be a numpy zipped file (.npz)
                with a 'traj_data' array and possibly a 'split_points' array inside. The 'traj_data'
                should be in the shape (joints x observations). If traj_files is specified, this should be None.
            warn (bool): If True, a warning will be raised.
        """

        if self.th is not None and warn:
            warnings.warn("New trajectories loaded, which overrides the old ones.", RuntimeWarning, stacklevel=2)

        th_params = self._th_params if self._th_params is not None else {}
        self.th = TrajectoryHandler(
            model=self._model,
            warn=warn,
            traj_path=traj_path,
            traj=traj,
            control_dt=self.dt,
            cache_type=cache_type,
            site_names=site_names,
            **th_params,
        )

        if self.th.traj.obs_container is not None:
            assert self.obs_container == self.th.traj.obs_container, (
                "Observation containers of trajectory and environment do not match. \n"
                "Please, either load a trajectory with the same observation container or "
                "set the observation container of the environment to the one of the trajectory."
            )

        if self.scaling != 1.0:
            # scale trajectory
            traj_info = self.th.traj.info
            traj_data = self.th.traj.data
            free_jnt_pos_id = self.free_jnt_qpos_id[:, :3].reshape(-1)
            free_jnt_lin_vel_id = self.free_jnt_qvel_id[:, :3].reshape(-1)

            # scale trajectory (only qpos and qvel)
            traj_data_new = TrajectoryData(
                qpos=traj_data.qpos.at[:, free_jnt_pos_id].mul(self.scaling),
                qvel=traj_data.qvel.at[:, free_jnt_lin_vel_id].mul(self.scaling),
                split_points=traj_data.split_points,
            )

            # create a new traj info
            traj_model = TrajectoryModel(njnt=traj_info.model.njnt, jnt_type=traj_info.model.jnt_type)
            traj_info = TrajectoryInfo(
                joint_names=traj_info.joint_names, model=traj_model, frequency=traj_info.frequency
            )

            # combine to trajectory
            traj = Trajectory(info=traj_info, data=traj_data_new)

            traj = extend_motion(self.__class__.__name__, {}, traj)

            # update trajectory handler
            self.th = TrajectoryHandler(model=self._model, warn=warn, traj=traj, control_dt=self.dt, **th_params)

        # setup trajectory information in observation_dict, goal and reward if needed
        for obs_entry in self.obs_container.entries():
            obs_entry.init_from_traj(self.th)
        self._goal.init_from_traj(self.th)
        self._terminal_state_handler.init_from_traj(self.th)

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body.
        """
        return "thorax"

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the Mujoco specification:
            1. Disable all contacts except the ones between feet and the floor.
            2. Scale nconmax for Warp backend with multiple parallel environments.

        Args:
            spec (MjSpec): Handle to Mujoco specification.

        Returns:
            Mujoco specification.
        """

        # Set contact budget for Warp backend
        if hasattr(self, "mjx_backend") and self.mjx_backend == "warp":
            # MuJoCo 3.3.7+: nconmax is per-env, naconmax is total across all envs
            per_env_contacts = 10  # Contacts per environment
            num_envs = getattr(self, "num_envs", 1)

            if not hasattr(self, "nconmax") or self.nconmax is None:
                self.nconmax = per_env_contacts  # Per-env contacts
            if not hasattr(self, "naconmax") or getattr(self, "naconmax", None) is None:
                self.naconmax = self.nconmax * num_envs  # Total across all envs
            if not hasattr(self, "njmax") or self.njmax is None:
                self.njmax = 256  # Per-world constraints
            spec.nconmax = self.nconmax
            spec.njmax = self.njmax

        # --- Backend-specific contact handling ---
        if self.mjx_backend == "warp":
            # Warp can handle contacts - keep them enabled
            logger.info("Keeping all contacts enabled for Warp backend")
            pass  # Don't disable contacts
        else:
            # JAX backend - disable complex contacts, keep essential ones
            for g in spec.geoms:
                g.contype = 0
                g.conaffinity = 0
        # Conditionally force dense matrices based on backend
        # JAX backend: Force dense matrices to avoid sparse tendon armature issues
        # This is needed because bimanual model has nv >= 60, triggering sparse mode
        # but MJX 3.3.3 explicitly doesn't support tendon operations with sparse matrices
        # Warp backend: Dense matrices are unsupported for nv > 60, so use sparse (default)
        if hasattr(self, "mjx_backend") and self.mjx_backend == "warp":
            # Warp backend - use default (sparse) matrices for nv > 60
            pass  # Use default sparse matrices
        else:
            # JAX backend (default) - force dense matrices for tendon compatibility
            spec.option.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

        return spec

    @info_property
    def sites_for_mimic(self) -> list[str]:
        """
        Returns the list of sites for mimic using configuration-driven approach like UnitreeH1.

        This can be overridden by configuration parameters passed to goal_params.sites_for_mimic
        or reward_params.sites_for_mimic. If neither is provided, derives sites from robot configuration.
        """
        # Check stored initialization parameters first (for backward compatibility)
        if hasattr(self, "_init_goal_params") and "sites_for_mimic" in self._init_goal_params:
            return self._init_goal_params["sites_for_mimic"]
        if hasattr(self, "_init_reward_params") and "sites_for_mimic" in self._init_reward_params:
            return self._init_reward_params["sites_for_mimic"]

        # Default sites for bimanual manipulation (7 sites, including upper_body)
        sites_for_mimic = [
            "upper_body_mimic",
            "right_shoulder_mimic",
            "right_elbow_mimic",
            "right_hand_mimic",
            "left_shoulder_mimic",
            "left_elbow_mimic",
            "left_hand_mimic",
        ]

        # Canonicalize to model site ID order if model is available
        if hasattr(self, "_model") and self._model is not None:
            return self._canonicalize_sites_by_model_id(sites_for_mimic)
        else:
            return sites_for_mimic

    def _canonicalize_sites_by_model_id(self, sites: list[str]) -> list[str]:
        """
        Sort site names by their model site ID to ensure consistent ordering.

        Args:
            sites: List of site names to sort

        Returns:
            List of site names sorted by ascending model site ID
        """
        # Get site IDs for the provided sites
        site_id_pairs = []
        for site_name in sites:
            site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:  # Valid site ID
                site_id_pairs.append((site_name, site_id))

        # Sort by model site ID (ascending) - this matches trajectory handler logic
        site_id_pairs.sort(key=lambda x: x[1])
        return [site_name for site_name, _ in site_id_pairs]

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body.
        """
        return "root"

    def set_sim_state_from_traj_data(self, data, traj_data, carry):
        """
        Sets the simulation state from the trajectory data for standard MuJoCo backend.

        For bimanual models, we only set joint angles since the model is fixed in place
        and doesn't have a root joint to track global position.

        Args:
            data (MjData): Current Mujoco data.
            traj_data (TrajectoryData): Data from the trajectory.
            carry (AdditionalCarry): Additional carry information.

        Returns:
            MjData: Updated Mujoco data.
        """
        # For bimanual models, directly use the parent Mjx implementation
        # without trying to access root joint positioning
        from musclemimic.core.mujoco_mjx import Mjx

        return Mjx.set_sim_state_from_traj_data(data, traj_data, carry)

    def mjx_set_sim_state_from_traj_data(self, data, traj_data, carry):
        """
        Sets the simulation state from the trajectory data for MJX backend.

        For bimanual models, we only set joint angles since the model is fixed in place
        and doesn't have a root joint to track global position.

        Args:
            data (Data): Current Mujoco data.
            traj_data (TrajectoryData): Data from the trajectory.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Data: Updated Mujoco data.
        """
        # For bimanual models, directly use the parent Mjx implementation
        # without trying to access root joint positioning
        from musclemimic.core.mujoco_mjx import Mjx

        return Mjx.mjx_set_sim_state_from_traj_data(data, traj_data, carry)
