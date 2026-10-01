"""
MyoFullBody environment - A full-body muscle-actuated humanoid model.
"""

import loco_mujoco
import mujoco
from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from musclemimic.environments import LocoEnv
from musclemimic.utils.logging import setup_logger
from musclemimic.utils.model_fixes import apply_myofullbody_model_fixes
from mujoco import MjSpec
import musclemimic_models

logger = setup_logger(__name__, identifier="[MyoFullBody]")


class MyoFullBody(LocoEnv):
    """
    Description
    ------------

    MuJoCo environment of a full-body humanoid model with muscle actuation.
    This model combines the full skeletal structure with comprehensive muscle control
    throughout the entire body, including legs, torso, arms, and optional fingers.

    The model uses muscle actuators (MuJoCo type 4) with Hill-type muscle dynamics,
    providing biomechanically realistic force generation and movement patterns.

    .. note:: Control range for all muscles is modified from default [0,1] to [-1,1]
              to match the MyoBimanualArm convention.

    Default Observation Space
    -----------------

    The observation space consists of joint positions and velocities for the full body,
    plus optional muscle state observations (length, velocity, force, activation).

    Default Action Space
    -------------------

    Control function type: **DefaultControl**

    All muscle actuators use control range [-1.0, 1.0] where:
    - Negative values (-1.0 to 0.0) represent muscle relaxation to baseline activation
    - Positive values (0.0 to 1.0) represent increasing muscle activation levels

    Methods
    ------------

    """

    mjx_enabled = False

    def __init__(
        self,
        timestep: float = 0.002,
        n_substeps: int = 5,
        disable_fingers: bool = True,
        enable_joint_pos_observations: bool = True,
        enable_joint_vel_observations: bool = True,
        enable_global_root_position_observation: bool = False,
        enable_muscle_length_observations: bool = False,
        enable_muscle_velocity_observations: bool = False,
        enable_muscle_force_observations: bool = False,
        enable_muscle_excitation_observations: bool = False,
        enable_muscle_activation_observations: bool = False,
        enable_touch_sensor_observations: bool = True,
        enable_heightmap_observations: bool = False,
        heightmap_grid_rows: int = 11,
        heightmap_grid_cols: int = 11,
        heightmap_grid_resolution: float = 0.1,
        heightmap_grid_forward_offset: float = 0.0,
        heightmap_body_name: str | None = None,
        spec: str | MjSpec = None,
        observation_spec: list[ObservationType] = None,
        actuation_spec: list[str] = None,
        mjx_backend: str = "jax",
        **kwargs,
    ) -> None:
        """
        Constructor for MyoFullBody environment.

        Args:
            timestep (float): Simulation timestep in seconds. Default 0.002 (500Hz physics).
            n_substeps (int): Number of physics substeps per control step. Default 5 (100Hz control).
            disable_fingers (bool): If True (default), finger joints and muscles are disabled
            enable_joint_pos_observations (bool): If True (default), include joint positions in observations
            enable_joint_vel_observations (bool): If True (default), include joint velocities in observations
            enable_global_root_position_observation (bool): Include root x/y instead of removing them.
            enable_muscle_length_observations (bool): If True, include muscle length in observations
            enable_muscle_velocity_observations (bool): If True, include muscle velocity in observations
            enable_muscle_force_observations (bool): If True, include muscle force in observations
            enable_muscle_excitation_observations (bool): If True, include muscle excitation (neural drive from data.ctrl) in observations
            enable_muscle_activation_observations (bool): If True, include muscle activation (actual state from data.act) in observations
            enable_heightmap_observations (bool): If True, include a yaw-aligned terrain height grid.
            heightmap_grid_rows (int): Number of height samples in the forward direction.
            heightmap_grid_cols (int): Number of height samples in the lateral direction.
            heightmap_grid_resolution (float): Spacing between height samples in meters.
            heightmap_grid_forward_offset (float): Forward shift of the yaw-aligned grid in meters.
            heightmap_body_name (str): Body used as the grid origin. Defaults to the root body.
            spec (Union[str, MjSpec]): Path to XML file or MjSpec object. If None, uses default.
            observation_spec (List[ObservationType]): Custom observation specification.
            actuation_spec (List[str]): Custom action specification.
            **kwargs: Additional arguments passed to parent class.
        """

        self._disable_fingers = disable_fingers
        self._enable_joint_pos_observations = enable_joint_pos_observations
        self._enable_joint_vel_observations = enable_joint_vel_observations
        self._enable_global_root_position_observation = enable_global_root_position_observation
        self._enable_muscle_length_observations = enable_muscle_length_observations
        self._enable_muscle_velocity_observations = enable_muscle_velocity_observations
        self._enable_muscle_force_observations = enable_muscle_force_observations
        self._enable_muscle_excitation_observations = enable_muscle_excitation_observations
        self._enable_muscle_activation_observations = enable_muscle_activation_observations
        self._enable_touch_sensor_observations = enable_touch_sensor_observations
        self._enable_heightmap_observations = enable_heightmap_observations
        self._heightmap_grid_rows = int(heightmap_grid_rows)
        self._heightmap_grid_cols = int(heightmap_grid_cols)
        self._heightmap_grid_resolution = float(heightmap_grid_resolution)
        self._heightmap_grid_forward_offset = float(heightmap_grid_forward_offset)
        self._heightmap_body_name = heightmap_body_name
        if self._heightmap_grid_rows < 1 or self._heightmap_grid_cols < 1:
            raise ValueError("Heightmap grid dimensions must be positive.")
        if self._heightmap_grid_resolution <= 0.0:
            raise ValueError("heightmap_grid_resolution must be positive.")

        # Store mjx_backend for use in _modify_spec_for_mjx
        self.mjx_backend = mjx_backend

        # Store nconmax and num_envs from kwargs for automatic scaling
        self.nconmax = kwargs.get("nconmax", None)
        self.njmax = kwargs.get("njmax", None)  # Also handle njmax
        self.num_envs = kwargs.get("num_envs", 1)

        if spec is None:
            spec = self.get_default_xml_file_path()

        # Load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # Apply changes to the MjSpec
        spec = self._apply_spec_changes(spec)

        # Get observation and action specifications
        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        else:
            observation_spec = self.parse_observation_spec(observation_spec)

        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # Modify spec for MJX if enabled
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            spec=spec,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            mjx_backend=mjx_backend,
            **kwargs,
        )

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        """
        Apply changes to the MjSpec including:
        1. Disabling fingers if requested (same as MyoBimanualArm)
        2. Adding mimic sites for trajectory tracking
        3. Modifying muscle control ranges from [0, 1] to [-1, 1] to match MyoBimanualArm convention

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            MjSpec: Modified specification
        """

        # Correct bundled muscle wrap geometry before either the MuJoCo or MJX model is
        # compiled, so online dynamics and offline retargeting metrics see the same paths.
        spec = apply_myofullbody_model_fixes(spec)

        # Handle finger disabling if requested (same logic as MyoBimanualArm)
        if self._disable_fingers:
            # Define specific finger joint names to avoid matching hip joints
            # Use the exact finger joint names from MyoBimanualArm
            finger_joints = [
                # Right hand finger joints (from myoarm_body.xml)
                "cmc_flexion_r",
                "cmc_abduction_r",
                "mp_flexion_r",
                "ip_flexion_r",
                "mcp2_flexion_r",
                "mcp2_abduction_r",
                "mcp3_flexion_r",
                "mcp3_abduction_r",
                "mcp4_flexion_r",
                "mcp4_abduction_r",
                "mcp5_flexion_r",
                "mcp5_abduction_r",
                "md2_flexion_r",
                "md3_flexion_r",
                "md4_flexion_r",
                "md5_flexion_r",
                "pm2_flexion_r",
                "pm3_flexion_r",
                "pm4_flexion_r",
                "pm5_flexion_r",
                # Left hand finger joints (from myoarm_left_body.xml - uses "L" suffix)
                "cmc_flexion_l",
                "cmc_abduction_l",
                "mp_flexion_l",
                "ip_flexion_l",
                "mcp2_flexion_l",
                "mcp2_abduction_l",
                "mcp3_flexion_l",
                "mcp3_abduction_l",
                "mcp4_flexion_l",
                "mcp4_abduction_l",
                "mcp5_flexion_l",
                "mcp5_abduction_l",
                "md2_flexion_l",
                "md3_flexion_l",
                "md4_flexion_l",
                "md5_flexion_l",
                "pm2_flexion_l",
                "pm3_flexion_l",
                "pm4_flexion_l",
                "pm5_flexion_l",
            ]

            finger_muscles = [
                # Right hand muscles
                "FDS2",
                "FDS3",
                "FDS4",
                "FDS5",  # Finger flexors (superficial)
                "FDP2",
                "FDP3",
                "FDP4",
                "FDP5",  # Finger flexors (deep)
                "EDC2",
                "EDC3",
                "EDC4",
                "EDC5",  # Finger extensors
                "EDM",
                "EIP",  # Finger extensors (specific)
                "EPL",
                "EPB",
                "FPL",
                "APL",  # Thumb muscles
                "OP",  # Opponens pollicis
                "RI2",
                "RI3",
                "RI4",
                "RI5",  # Radial interossei
                "LU_RB2",
                "LU_RB3",
                "LU_RB4",
                "LU_RB5",  # Lumbricals
                "UI_UB2",
                "UI_UB3",
                "UI_UB4",
                "UI_UB5",  # Ulnar interossei
                # Left hand muscles (with L suffix)
                "FDS2_left",
                "FDS3_left",
                "FDS4_left",
                "FDS5_left",  # Left finger flexors (superficial)
                "FDP2_left",
                "FDP3_left",
                "FDP4_left",
                "FDP5_left",  # Left finger flexors (deep)
                "EDC2_left",
                "EDC3_left",
                "EDC4_left",
                "EDC5_left",  # Left finger extensors
                "EDM_left",
                "EIP_left",  # Left finger extensors (specific)
                "EPL_left",
                "EPB_left",
                "FPL_left",
                "APL_left",  # Left thumb muscles
                "OP_left",  # Left opponens pollicis
                "RI2_left",
                "RI3_left",
                "RI4_left",
                "RI5_left",  # Left radial interossei
                "LU_RB2_left",
                "LU_RB3_left",
                "LU_RB4_left",
                "LU_RB5_left",  # Left lumbricals
                "UI_UB2_left",
                "UI_UB3_left",
                "UI_UB4_left",
                "UI_UB5_left",  # Left ulnar interossei
            ]

            # Remove finger joints (use exact match to avoid matching hip joints)
            joints_to_remove = []
            for joint in spec.joints:
                if joint.name in finger_joints:
                    joints_to_remove.append(joint)

            for joint in joints_to_remove:
                spec.delete(joint)
            # print(f"[MyoFullBody] Removed {len(joints_to_remove)} finger joints: {[j.name for j in joints_to_remove]}")

            # Remove finger muscles and their tendons (use exact match to avoid matching arm muscles)
            actuators_to_remove = []
            for actuator in spec.actuators:
                if actuator.name in finger_muscles:
                    actuators_to_remove.append(actuator)

            # print(f"[MyoFullBody] Removing {len(actuators_to_remove)} finger muscles: {[a.name for a in actuators_to_remove[:10]]}")
            for actuator in actuators_to_remove:
                spec.delete(actuator)

            # Remove associated tendons (use clean substring matching like MyoBimanualArm)
            tendons_to_remove = []
            for tendon in spec.tendons:
                if any(finger_muscle in tendon.name for finger_muscle in finger_muscles):
                    tendons_to_remove.append(tendon)

            # print(f"[MyoFullBody] Removing {len(tendons_to_remove)} finger tendons: {[t.name for t in tendons_to_remove[:10]]}")
            for tendon in tendons_to_remove:
                spec.delete(tendon)

        # Add mimic sites for trajectory tracking
        for body_name, site_name in self.body2sites_for_mimic.items():
            b = spec.body(body_name)
            pos = [0.0, 0.0, 0.0]
            b.add_site(
                name=site_name,
                group=4,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.075, 0.05, 0.025],
                rgba=[1.0, 0.0, 0.0, 0.5],
                pos=pos,
            )

        
        # Modify control range to -1 to 1
        for actuator in spec.actuators:
            # Only modify muscle actuators (type 4 in MuJoCo), not motor actuators
            if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE:
                actuator.ctrlrange = [-1.0, 1.0]
                actuator.ctrllimited = True
        return spec

    def _get_observation_specification(self, spec: MjSpec) -> list[ObservationType]:
        """
        Get observation specification including joint positions/velocities
        and optional muscle observations.

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            List[ObservationType]: List of observations
        """
        obs_spec = []

        # Get all joint names except the root
        j_names = [j.name for j in spec.joints if j.name != self.root_free_joint_xml_name]

        # Add joint position observations if enabled
        if self._enable_joint_pos_observations:
            root_obs_type = (
                ObservationType.FreeJointPos
                if self._enable_global_root_position_observation
                else ObservationType.FreeJointPosNoXY
            )
            obs_spec.append(root_obs_type("q_free_joint", self.root_free_joint_xml_name))
            # Add all joint positions
            obs_spec.append(ObservationType.JointPosArray("q_all_pos", j_names))

        # Add joint velocity observations if enabled
        if self._enable_joint_vel_observations:
            # Add free joint velocities
            obs_spec.append(ObservationType.FreeJointVel("dq_free_joint", self.root_free_joint_xml_name))
            # Add all joint velocities
            obs_spec.append(ObservationType.JointVelArray("dq_all_vel", j_names))

        # Add muscle observations if enabled
        for actuator in spec.actuators:
            actuator_name = actuator.name

            # Add muscle length observations
            if self._enable_muscle_length_observations:
                obs_name = f"muscle_length_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorLength(obs_name, xml_name=actuator_name))

            # Add muscle velocity observations
            if self._enable_muscle_velocity_observations:
                obs_name = f"muscle_velocity_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorVelocity(obs_name, xml_name=actuator_name))

            # Add muscle force observations
            if self._enable_muscle_force_observations:
                obs_name = f"muscle_force_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorForce(obs_name, xml_name=actuator_name))

            # Add muscle excitation observations (neural drive from data.ctrl)
            if self._enable_muscle_excitation_observations:
                obs_name = f"muscle_excitation_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorExcitation(obs_name, xml_name=actuator_name))

            # Add muscle activation observations (actual state from data.act)
            if self._enable_muscle_activation_observations:
                obs_name = f"muscle_activation_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorActivation(obs_name, xml_name=actuator_name))

        # Add touch sensor observations for foot contact feedback if enabled
        # These sensors are crucial for locomotion balance and gait control
        if self._enable_touch_sensor_observations:
            touch_sensors = ["r_foot", "r_toes", "l_foot", "l_toes"]
            for sensor_name in touch_sensors:
                obs_name = f"touch_{sensor_name}"
                obs_spec.append(ObservationType.TouchSensor(obs_name, xml_name=sensor_name))

        if self._enable_heightmap_observations:
            obs_spec.append(
                ObservationType.HeightMatrix(
                    "terrain_heightmap",
                    grid_rows=self._heightmap_grid_rows,
                    grid_cols=self._heightmap_grid_cols,
                    grid_resolution=self._heightmap_grid_resolution,
                    grid_forward_offset=self._heightmap_grid_forward_offset,
                    body_name=self._heightmap_body_name,
                    allow_randomization=False,
                )
            )

        return obs_spec

    def _get_action_specification(self, spec: MjSpec) -> list[str]:
        """
        Get action specification - returns all actuator names.

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            List[str]: List of actuator names
        """
        action_spec = []
        for actuator in spec.actuators:
            action_spec.append(actuator.name)
        return action_spec

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """
        Modify the model specification for MJX backend compatibility.
        Uses the same contact-simplification idea as the older full-body setup,
        adapted for the maintained MyoFullBody asset.
        """
        if hasattr(self, "mjx_backend") and self.mjx_backend == "warp":
            # MuJoCo 3.3.7+: nconmax is per-env, naconmax is total across all envs
            per_env_contacts = 96  # Contacts per environment
            num_envs = getattr(self, "num_envs", 1)

            if not hasattr(self, "nconmax") or self.nconmax is None:
                self.nconmax = per_env_contacts  # Per-env contacts
            if not hasattr(self, "naconmax") or getattr(self, "naconmax", None) is None:
                self.naconmax = self.nconmax * num_envs  # Total across all envs
            if not hasattr(self, "njmax") or self.njmax is None:
                self.njmax = 768  # Per-world constraints

            # Apply the limits to the spec (spec uses per-env values)
            spec.nconmax = self.nconmax
            spec.njmax = self.njmax

            logger.info(
                "nconmax=%s (per-env), naconmax=%s (total), njmax=%s",
                self.nconmax,
                self.naconmax,
                self.njmax,
            )
            logger.info("Keeping all contacts enabled for Warp backend")
        else:
            for g in spec.geoms:
                # Keep essential ground contacts but disable others
                if g.name and ("floor" in g.name.lower() or "ground" in g.name.lower()):
                    continue  # Keep floor/ground contacts
                g.contype = 0
                g.conaffinity = 0

        return spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.
        """
        return musclemimic_models.get_xml_path("myofullbody").as_posix()

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the root free joint in the Mujoco xml file.
        """
        return "root"

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco xml file.
        """
        return "pelvis"

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml file.
        """
        return "torso"

    @info_property
    def root_height_healthy_range(self) -> tuple[float, float]:
        """
        Returns the healthy range of the root height.
        """
        return (0.6, 1.5)

    @info_property
    def body2sites_for_mimic(self) -> dict[str, str]:
        """
        Returns a dictionary mapping body names to their corresponding mimic site names.
        Tailored for the maintained full-body muscle model.

        Focus on essential tracking points: ankle and toe for proper foot orientation.
        Let forward kinematics handle heel (calcn) positioning.
        """
        body2sitemimic = {
            "pelvis": "pelvis_mimic",
            "lumbar1": "upper_body_mimic",
            "head": "head_mimic",
            # Left arm (note: left uses capital L suffix)
            "humerus_l": "left_shoulder_mimic",
            "ulna_l": "left_elbow_mimic",
            "lunate_l": "left_hand_mimic",
            # Right arm (note: right has no suffix)
            "humerus_r": "right_shoulder_mimic",
            "ulna_r": "right_elbow_mimic",
            "lunate_r": "right_hand_mimic",
            "femur_l": "left_hip_mimic",
            "tibia_l": "left_knee_mimic",
            "talus_l": "left_ankle_mimic",
            "toes_l": "left_toes_mimic",
            "femur_r": "right_hip_mimic",
            "tibia_r": "right_knee_mimic",
            "talus_r": "right_ankle_mimic",
            "toes_r": "right_toes_mimic",
        }
        return body2sitemimic

    @info_property
    def sites_for_mimic(self) -> list[str]:
        """
        Returns a list of all mimic sites.
        """
        return list(self.body2sites_for_mimic.values())

    @info_property
    def goal_visualization_arrow_offset(self) -> list[float]:
        """
        Returns the offset for the goal visualization arrow.
        """
        return [0, 0, 0.4]


class MjxMyoFullBody(MyoFullBody):
    """
    MJX version of MyoFullBody with support for JAX and warp backends.
    """

    mjx_enabled = True

    def __init__(self, timestep: float = 0.002, n_substeps: int = 5, mjx_backend: str = "jax", **kwargs):
        """
        Constructor for MJX version of MyoFullBody.

        Args:
            timestep (float): Timestep of the simulation.
            n_substeps (int): Number of substeps.
            mjx_backend (str): MJX backend to use ('jax' or 'warp'). Default: 'jax'.
            **kwargs: Additional arguments.
        """
        # Extract goal-related parameters to prevent them from being passed to viewer
        goal_related_params = [
            "visualize_goal",
            "enable_enhanced_visualization",
            "target_geom_rgba",
            "n_step_lookahead",
            "goal_type",
            "goal_params",
        ]

        extracted_goal_params = {}
        for param in goal_related_params:
            if param in kwargs:
                extracted_goal_params[param] = kwargs.pop(param)

        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=4, ls_iterations=8, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]

        # Pass goal-related parameters back through kwargs
        kwargs.update(extracted_goal_params)

        # Store mjx_backend for use in parent class
        self.mjx_backend = mjx_backend

        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            model_option_conf=model_option_conf,
            mjx_backend=mjx_backend,
            **kwargs,
        )
