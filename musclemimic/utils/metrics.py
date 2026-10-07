from dataclasses import fields

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from jax.scipy.spatial.transform import Rotation as R  # noqa: N817
from metrx import DistanceMeasures
from omegaconf import DictConfig, OmegaConf

from loco_mujoco.core.utils.math import (
    calc_site_velocities,
    calculate_relative_site_quantities,
    quat_scalarfirst2scalarlast,
)
from loco_mujoco.core.utils.mujoco import mj_jntid2qposid, mj_jntid2qvelid, mj_jntname2qposid
from musclemimic.core.reward.orientation import scalar_first_quaternion_geodesic_distance
from musclemimic.core.utils.site_mapping import create_site_mapper
from musclemimic.core.wrappers import SummaryMetrics

SUPPORTED_QUANTITIES = [
    "JointPosition",
    "JointVelocity",
    "BodyPosition",
    "BodyVelocity",
    "BodyOrientation",
    "SitePosition",
    "SiteVelocity",
    "SiteOrientation",
    "RelSitePosition",
    "RelSiteVelocity",
    "RelSiteOrientation",
]

SUPPORTED_MEASURES = ["EuclideanDistance", "DynamicTimeWarping", "DiscreteFrechetDistance"]

VALIDATION_STEP_METRIC_KEYS = (
    "reward_total",
    "reward_qpos",
    "reward_qvel",
    "reward_root_pos",
    "reward_root_quat",
    "reward_rpos",
    "reward_rquat",
    "reward_rvel_rot",
    "reward_rvel_lin",
    "reward_root_vel",
    "penalty_total",
    "penalty_activation_energy",
    "err_root_xyz",
    "err_root_quat",
    "err_root_yaw",
    "err_joint_pos",
    "err_joint_vel",
    "err_site_abs",
    "err_rpos",
)


def _quat_to_yaw_wxyz(quat):
    """Extract yaw from quaternion in [w, x, y, z] format."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@struct.dataclass
class QuantityContainer:
    qpos: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    qvel: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    xpos: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    xrotvec: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    cvel: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_xpos: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_xrotvec: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_xvel: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_rpos: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_rrotvec: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    site_rvel: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))


@struct.dataclass
class ValidationSummary(SummaryMetrics):
    # Completed episodes behind the episode-level means/rates; cross-replica
    # reductions reweight by this count.
    episode_count: float = 0.0
    # Frame-weighted coverage of the selected trajectory set. Exact online
    # validation can censor trajectories at a fixed horizon.
    frame_coverage: float = 0.0
    total_frame: float = 0.0
    euclidean_distance: QuantityContainer = struct.field(default_factory=QuantityContainer)
    dynamic_time_warping: QuantityContainer = struct.field(default_factory=QuantityContainer)
    discrete_frechet_distance: QuantityContainer = struct.field(default_factory=QuantityContainer)
    # Per-arm metrics for curriculum learning
    left_arm_euclidean_distance: QuantityContainer = struct.field(default_factory=QuantityContainer)
    right_arm_euclidean_distance: QuantityContainer = struct.field(default_factory=QuantityContainer)


class MetricsHandler:
    supported_measures = SUPPORTED_MEASURES
    supported_quantities = SUPPORTED_QUANTITIES

    def __init__(self, config: DictConfig, env):
        self._config = config.experiment

        # Store reference to trajectory handler instead of trajectory data
        self._trajectory_handler = env.th if env.th is not None else None

        self.quantaties = OmegaConf.select(self._config, "validation.quantities")
        self.measures = OmegaConf.select(self._config, "validation.measures")

        rel_joint_names = OmegaConf.select(self._config, "validation.rel_joint_names")
        joints_to_ignore = OmegaConf.select(self._config, "validation.joints_to_ignore")
        rel_body_names = OmegaConf.select(self._config, "validation.rel_body_names")
        rel_site_names = OmegaConf.select(self._config, "validation.rel_site_names")

        if joints_to_ignore is None:
            joints_to_ignore = []

        model = env.get_model()
        if rel_joint_names is not None:
            self.rel_qpos_ids = [
                jnp.array(mj_jntid2qposid(name, model)) for name in rel_joint_names if name not in joints_to_ignore
            ]
            self.rel_qvel_ids = [
                jnp.array(mj_jntid2qvelid(name, model)) for name in rel_joint_names if name not in joints_to_ignore
            ]
        else:
            self.rel_qpos_ids = [
                jnp.array(mj_jntid2qposid(i, model))
                for i in range(model.njnt)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) not in joints_to_ignore
            ]
            self.rel_qvel_ids = [
                jnp.array(mj_jntid2qvelid(i, model))
                for i in range(model.njnt)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) not in joints_to_ignore
            ]

        if rel_body_names is not None:
            self.rel_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in rel_body_names]
            assert -1 not in self.rel_body_ids, f"Body {rel_body_names[self.rel_body_ids.index(-1)]} not found."
        else:
            self.rel_body_ids = list(range(model.nbody))

        if rel_site_names is not None:
            self.rel_site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in rel_site_names]
            assert -1 not in self.rel_site_ids, f"Site {rel_site_names[self.rel_site_ids.index(-1)]} not found."
        else:
            self.rel_site_ids = list(range(model.nsite))

        self._site_bodyid = jnp.array(
            [model.site_bodyid[i] for i in range(model.nsite)]
        )  # get the body id of all sites
        self._body_rootid = jnp.array(model.body_rootid)  # get the root body id for all bodies

        if self.measures is not None:
            assert self._traj_data is not None, "Trajectory data is required for calculating measures."
            for m in self.measures:
                assert m in SUPPORTED_MEASURES, f"{m} is not a supported measure."

            def dummy_func(x, y):
                return 0.0

            self._euclidean_distance = (
                jax.vmap(
                    jax.vmap(DistanceMeasures.create_instance("EuclideanDistance", mean=True), in_axes=(0, 0)),
                    in_axes=(0, 0),
                )
                if "EuclideanDistance" in self.measures
                else dummy_func
            )
            self._dynamic_time_warping = (
                jax.vmap(
                    jax.vmap(DistanceMeasures.create_instance("DynamicTimeWarping"), in_axes=(0, 0)), in_axes=(0, 0)
                )
                if "DynamicTimeWarping" in self.measures
                else dummy_func
            )
            self._discrete_frechet_distance = (
                jax.vmap(
                    jax.vmap(DistanceMeasures.create_instance("DiscreteFrechetDistance"), in_axes=(0, 0)),
                    in_axes=(0, 0),
                )
                if "DiscreteFrechetDistance" in self.measures
                else dummy_func
            )

        if self.quantaties is not None:
            for q in self.quantaties:
                assert q in SUPPORTED_QUANTITIES, f"{q} is not a supported quantity."

                if "Rel" in self.quantaties:
                    assert self.rel_site_ids is not None, (
                        "Relative site quantities requires relative site ids with "
                        "the first site being the site used to calculate the "
                        "relative quantities."
                    )

        self._vec_calc_site_velocities = jax.vmap(
            jax.vmap(calc_site_velocities, in_axes=(None, 0, None, None, None, None, None)),
            in_axes=(None, 0, None, None, None, None, None),
        )
        self._vec_calc_rel_site_quantities = jax.vmap(
            jax.vmap(calculate_relative_site_quantities, in_axes=(0, None, None, None, None, None)),
            in_axes=(0, None, None, None, None, None),
        )

        # Identify which joint DOFs in qpos are quaternions vs scalars so we can
        # convert quats to rotation vectors for metrics.
        self._quat_in_qpos = jnp.concatenate(
            [jnp.array([False] * 3 + [True] * 4) if len(j) == 7 else jnp.array([False]) for j in self.rel_qpos_ids]
        )
        self._not_quat_in_qpos = jnp.invert(self._quat_in_qpos)
        self.rel_qpos_ids = jnp.concatenate(self.rel_qpos_ids)
        self.rel_qvel_ids = jnp.concatenate(self.rel_qvel_ids)
        self.rel_body_ids = jnp.array(self.rel_body_ids)
        self.rel_site_ids = jnp.array(self.rel_site_ids)

        # Per-arm site indices
        self._left_arm_site_indices = None
        self._right_arm_site_indices = None

        if rel_site_names is not None:
            # Focus on manipulation end-effectors: elbow + hand
            left_arm_sites = ["left_elbow_mimic", "left_hand_mimic"]
            right_arm_sites = ["right_elbow_mimic", "right_hand_mimic"]

            # Find indices in the site configuration
            left_indices = [i for i, name in enumerate(rel_site_names) if name in left_arm_sites]
            right_indices = [i for i, name in enumerate(rel_site_names) if name in right_arm_sites]

            # Enable if both arms have at least one site
            if len(left_indices) > 0 and len(right_indices) > 0:
                self._left_arm_site_indices = jnp.array(left_indices)
                self._right_arm_site_indices = jnp.array(right_indices)
                print(f"  Left arm sites: {[rel_site_names[i] for i in left_indices]}")
                print(f"  Right arm sites: {[rel_site_names[i] for i in right_indices]}")

        # Initialize site mapper for trajectory index mapping (use actual trajectory site order if available)
        env_sites_for_mimic = getattr(env, "sites_for_mimic", [])
        traj_site_names = env.th.traj.info.site_names if (hasattr(env, "th") and env.th is not None) else None
        self._site_mapper = create_site_mapper(model, env.__class__.__name__, env_sites_for_mimic, traj_site_names)

        # Store root qpos indices for offset alignment and root error metrics.
        # When episodes start at random XY positions, trajectory data is in world frame
        # while simulation resets to origin. We subtract the init XY offset so positions
        # are compared in local frame.
        self._root_qpos_ids_xy = None
        self._root_qpos_ids = None  # full 7-DOF (xyz + quat) for err_root_xyz/yaw
        info_props = env._get_all_info_properties() if hasattr(env, "_get_all_info_properties") else {}
        root_joint_name = info_props.get("root_free_joint_xml_name")
        if root_joint_name and model is not None:
            try:
                root_qpos_ids = jnp.array(mj_jntname2qposid(root_joint_name, model), dtype=int)
                if root_qpos_ids.size >= 7:
                    self._root_qpos_ids = root_qpos_ids
                if root_qpos_ids.size >= 2:
                    self._root_qpos_ids_xy = root_qpos_ids[:2]
            except (ValueError, TypeError, IndexError) as exc:
                print(f"  Warning: failed to resolve root qpos indices for '{root_joint_name}': {exc}")

    def _get_root_xy_offset(self, env_states):
        """Get XY offset from trajectory init position for local frame alignment.

        Returns (x, y, 0) offset to subtract from world-frame trajectory positions,
        or None if offset correction is not applicable.
        """
        if self._root_qpos_ids_xy is None or self._traj_data is None:
            return None
        traj_states = env_states.additional_carry.traj_state
        if not hasattr(traj_states, "subtraj_step_no_init"):
            return None
        start_idx = self._traj_data.split_points[traj_states.traj_no]
        init_idx = start_idx + traj_states.subtraj_step_no_init
        init_qpos = self._traj_data.qpos[init_idx]
        root_xy = init_qpos[..., self._root_qpos_ids_xy]
        zeros = jnp.zeros((*root_xy.shape[:-1], 1), dtype=root_xy.dtype)
        return jnp.concatenate([root_xy, zeros], axis=-1)

    def _calculate_per_arm_metrics(self, distance_fn, container, container_traj):
        """
        Calculate per-arm metrics by splitting site-based quantities into left/right arms
        (using configured arm site indices) and applying the provided distance function.
        """
        if self._left_arm_site_indices is None or self._right_arm_site_indices is None:
            empty_container = QuantityContainer(
                qpos=jnp.empty(0),
                qvel=jnp.empty(0),
                xpos=jnp.empty(0),
                xrotvec=jnp.empty(0),
                cvel=jnp.empty(0),
                site_xpos=jnp.empty(0),
                site_xrotvec=jnp.empty(0),
                site_xvel=jnp.empty(0),
                site_rpos=jnp.empty(0),
                site_rrotvec=jnp.empty(0),
                site_rvel=jnp.empty(0),
            )
            return empty_container, empty_container

        # Helper function to extract arm-specific sites from quantities
        def extract_arm_sites(quantity_arr, arm_indices):
            if quantity_arr.size == 0:
                return quantity_arr
            # quantity_arr shape should be (N, S, D) after swapaxes in main function
            # where N=sites, S=samples, D=dimensions
            return quantity_arr[arm_indices]

        # If no site-based quantities were requested, return empty containers
        site_arrays_empty = (
            container.site_xpos.size == 0
            and container.site_xrotvec.size == 0
            and container.site_xvel.size == 0
            and container.site_rpos.size == 0
            and container.site_rrotvec.size == 0
            and container.site_rvel.size == 0
        )
        if site_arrays_empty:
            empty_container = QuantityContainer(
                qpos=jnp.empty(0),
                qvel=jnp.empty(0),
                xpos=jnp.empty(0),
                xrotvec=jnp.empty(0),
                cvel=jnp.empty(0),
                site_xpos=jnp.empty(0),
                site_xrotvec=jnp.empty(0),
                site_xvel=jnp.empty(0),
                site_rpos=jnp.empty(0),
                site_rrotvec=jnp.empty(0),
                site_rvel=jnp.empty(0),
            )
            return empty_container, empty_container

        # Extract left arm data
        left_container = QuantityContainer(
            qpos=container.qpos,  # Joint data is not site-specific
            qvel=container.qvel,  # Joint data is not site-specific
            xpos=container.xpos,  # Body data is not site-specific
            xrotvec=container.xrotvec,  # Body data is not site-specific
            cvel=container.cvel,  # Body data is not site-specific
            site_xpos=extract_arm_sites(container.site_xpos, self._left_arm_site_indices),
            site_xrotvec=extract_arm_sites(container.site_xrotvec, self._left_arm_site_indices),
            site_xvel=extract_arm_sites(container.site_xvel, self._left_arm_site_indices),
            site_rpos=extract_arm_sites(container.site_rpos, self._left_arm_site_indices),
            site_rrotvec=extract_arm_sites(container.site_rrotvec, self._left_arm_site_indices),
            site_rvel=extract_arm_sites(container.site_rvel, self._left_arm_site_indices),
        )

        left_container_traj = QuantityContainer(
            qpos=container_traj.qpos,  # Joint data is not site-specific
            qvel=container_traj.qvel,  # Joint data is not site-specific
            xpos=container_traj.xpos,  # Body data is not site-specific
            xrotvec=container_traj.xrotvec,  # Body data is not site-specific
            cvel=container_traj.cvel,  # Body data is not site-specific
            site_xpos=extract_arm_sites(container_traj.site_xpos, self._left_arm_site_indices),
            site_xrotvec=extract_arm_sites(container_traj.site_xrotvec, self._left_arm_site_indices),
            site_xvel=extract_arm_sites(container_traj.site_xvel, self._left_arm_site_indices),
            site_rpos=extract_arm_sites(container_traj.site_rpos, self._left_arm_site_indices),
            site_rrotvec=extract_arm_sites(container_traj.site_rrotvec, self._left_arm_site_indices),
            site_rvel=extract_arm_sites(container_traj.site_rvel, self._left_arm_site_indices),
        )

        # Extract right arm data
        right_container = QuantityContainer(
            qpos=container.qpos,  # Joint data is not site-specific
            qvel=container.qvel,  # Joint data is not site-specific
            xpos=container.xpos,  # Body data is not site-specific
            xrotvec=container.xrotvec,  # Body data is not site-specific
            cvel=container.cvel,  # Body data is not site-specific
            site_xpos=extract_arm_sites(container.site_xpos, self._right_arm_site_indices),
            site_xrotvec=extract_arm_sites(container.site_xrotvec, self._right_arm_site_indices),
            site_xvel=extract_arm_sites(container.site_xvel, self._right_arm_site_indices),
            site_rpos=extract_arm_sites(container.site_rpos, self._right_arm_site_indices),
            site_rrotvec=extract_arm_sites(container.site_rrotvec, self._right_arm_site_indices),
            site_rvel=extract_arm_sites(container.site_rvel, self._right_arm_site_indices),
        )

        right_container_traj = QuantityContainer(
            qpos=container_traj.qpos,  # Joint data is not site-specific
            qvel=container_traj.qvel,  # Joint data is not site-specific
            xpos=container_traj.xpos,  # Body data is not site-specific
            xrotvec=container_traj.xrotvec,  # Body data is not site-specific
            cvel=container_traj.cvel,  # Body data is not site-specific
            site_xpos=extract_arm_sites(container_traj.site_xpos, self._right_arm_site_indices),
            site_xrotvec=extract_arm_sites(container_traj.site_xrotvec, self._right_arm_site_indices),
            site_xvel=extract_arm_sites(container_traj.site_xvel, self._right_arm_site_indices),
            site_rpos=extract_arm_sites(container_traj.site_rpos, self._right_arm_site_indices),
            site_rrotvec=extract_arm_sites(container_traj.site_rrotvec, self._right_arm_site_indices),
            site_rvel=extract_arm_sites(container_traj.site_rvel, self._right_arm_site_indices),
        )

        # Calculate per-arm euclidean distances
        left_arm_metrics = jax.tree.map(
            lambda x, y: jnp.mean(distance_fn(x, y)) if x.size > 0 else x, left_container, left_container_traj
        )

        right_arm_metrics = jax.tree.map(
            lambda x, y: jnp.mean(distance_fn(x, y)) if x.size > 0 else x, right_container, right_container_traj
        )

        return left_arm_metrics, right_arm_metrics

    def __call__(self, env_states, sim_site_idx=None):
        # sim_site_idx: indices for simulation data (0..K-1 if pre-sliced, else global IDs)
        if sim_site_idx is None:
            sim_site_idx = self.rel_site_ids
        # calculate default metrics
        logged_metrics = env_states.metrics
        num_done = jnp.sum(logged_metrics.done.astype(jnp.float32))
        mean_episode_return = jnp.sum(
            jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)
        ) / jnp.maximum(num_done, 1.0)
        mean_episode_length = jnp.sum(
            jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)
        ) / jnp.maximum(num_done, 1.0)
        max_timestep = jnp.max(logged_metrics.timestep * self._config.num_envs)

        # Early termination: episodes that ended due to absorbing state (terminal handler)
        early_term_count = jnp.sum(
            logged_metrics.done.astype(jnp.float32) * logged_metrics.absorbing.astype(jnp.float32)
        )
        early_term_rate = jnp.where(num_done > 0, early_term_count / num_done, 0.0)

        # get all quantities
        if "JointPosition" in self.quantaties:
            qpos, traj_qpos = self.get_joint_positions(env_states)
            # extend last dim
            qpos = jnp.expand_dims(qpos, axis=-1)
            traj_qpos = jnp.expand_dims(traj_qpos, axis=-1)
        else:
            qpos = traj_qpos = jnp.empty(0)
        if "JointVelocity" in self.quantaties:
            qvel, traj_qvel = self.get_joint_velocities(env_states)
            # extend last dim
            qvel = jnp.expand_dims(qvel, axis=-1)
            traj_qvel = jnp.expand_dims(traj_qvel, axis=-1)
        else:
            qvel = traj_qvel = jnp.empty(0)
        if "BodyPosition" in self.quantaties:
            xpos, traj_xpos = self.get_body_positions(env_states)
        else:
            xpos = traj_xpos = jnp.empty(0)
        if "BodyOrientation" in self.quantaties:
            xrotvec, traj_xrotvec = self.get_body_orientations(env_states)
        else:
            xrotvec = traj_xrotvec = jnp.empty(0)
        if "BodyVelocity" in self.quantaties:
            cvel, traj_cvel = self.get_body_velocities(env_states)
        else:
            cvel = traj_cvel = jnp.empty(0)
        if "SitePosition" in self.quantaties:
            site_xpos, traj_site_xpos = self.get_site_positions(env_states, sim_site_idx)
        else:
            site_xpos = traj_site_xpos = jnp.empty(0)
        if "SiteOrientation" in self.quantaties:
            site_xrotvec, traj_site_xrotvec = self.get_site_orientations(env_states, sim_site_idx)
        else:
            site_xrotvec = traj_site_xrotvec = jnp.empty(0)
        if "SiteVelocity" in self.quantaties:
            site_xvel, traj_site_xvel = self.get_site_velocities(env_states, sim_site_idx)
        else:
            site_xvel = traj_site_xvel = jnp.empty(0)
        if (
            "RelSitePosition" in self.quantaties
            or "RelSiteOrientation" in self.quantaties
            or "RelSiteVelocity" in self.quantaties
        ):
            (
                rel_site_pos,
                rel_site_rotvec,
                rel_site_vel,
                traj_rel_site_pos,
                traj_rel_site_rotvec,
                traj_rel_site_vel,
            ) = self.get_relative_site_quantities(env_states, sim_site_idx)
        else:
            rel_site_pos = rel_site_rotvec = rel_site_vel = traj_rel_site_pos = traj_rel_site_rotvec = (
                traj_rel_site_vel
            ) = jnp.empty(0)

        # create containers
        container = QuantityContainer(
            qpos=qpos,
            qvel=qvel,
            xpos=xpos,
            xrotvec=xrotvec,
            cvel=cvel,
            site_xpos=site_xpos,
            site_xrotvec=site_xrotvec,
            site_xvel=site_xvel,
            site_rpos=rel_site_pos,
            site_rrotvec=rel_site_rotvec,
            site_rvel=rel_site_vel,
        )
        container_traj = QuantityContainer(
            qpos=traj_qpos,
            qvel=traj_qvel,
            xpos=traj_xpos,
            xrotvec=traj_xrotvec,
            cvel=traj_cvel,
            site_xpos=traj_site_xpos,
            site_xrotvec=traj_site_xrotvec,
            site_xvel=traj_site_xvel,
            site_rpos=traj_rel_site_pos,
            site_rrotvec=traj_rel_site_rotvec,
            site_rvel=traj_rel_site_vel,
        )

        # the dimensions for each quantity is (S, N, D) where S is the number of samples, N is the number of elements
        # (e.g., joints, bodies, site) and D is the dimension of the quantity (e.g., position, velocity, orientation).
        # We want to switch the dimensions to (N, S, D) to calculate the distance measures on trajectories.

        # err_joint_pos: RMSE of joint positions
        err_joint_pos = jnp.sqrt(jnp.mean(jnp.square(qpos - traj_qpos))) if qpos.size > 0 else 0.0
        # err_joint_vel: RMSE of joint velocities
        err_joint_vel = jnp.sqrt(jnp.mean(jnp.square(qvel - traj_qvel))) if qvel.size > 0 else 0.0
        # err_site_abs: mean L2 norm of absolute site position errors
        err_site_abs = jnp.mean(jnp.linalg.norm(site_xpos - traj_site_xpos, axis=-1)) if site_xpos.size > 0 else 0.0
        # err_rpos: RMSE of relative site positions
        err_rpos = jnp.sqrt(jnp.mean(jnp.square(rel_site_pos - traj_rel_site_pos))) if rel_site_pos.size > 0 else 0.0

        # Root position, full orientation, and yaw-only errors.
        err_root_xyz = 0.0
        err_root_quat = 0.0
        err_root_yaw = 0.0
        if self._root_qpos_ids is not None and self._traj_data is not None:
            sim_qpos = env_states.data.qpos
            traj_indices = self.get_traj_indices(env_states)
            traj_qpos_full = self._traj_data.qpos[traj_indices]
            offset = self._get_root_xy_offset(env_states)
            # Root XYZ error
            root_xyz = sim_qpos[..., self._root_qpos_ids[:3]]
            traj_root_xyz = traj_qpos_full[..., self._root_qpos_ids[:3]]
            if offset is not None:
                traj_root_xyz = traj_root_xyz - offset
            err_root_xyz = jnp.sqrt(jnp.mean(jnp.square(root_xyz - traj_root_xyz)))
            # Full SO(3) and yaw-only root orientation errors.
            root_quat = sim_qpos[..., self._root_qpos_ids[3:7]]
            traj_root_quat = traj_qpos_full[..., self._root_qpos_ids[3:7]]
            err_root_quat = jnp.mean(scalar_first_quaternion_geodesic_distance(root_quat, traj_root_quat, jnp))
            yaw_diff = _quat_to_yaw_wxyz(root_quat) - _quat_to_yaw_wxyz(traj_root_quat)
            err_root_yaw = jnp.mean(jnp.abs(jnp.arctan2(jnp.sin(yaw_diff), jnp.cos(yaw_diff))))

        container = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 2) if x.size > 0 else x, container)
        container_traj = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 2) if x.size > 0 else x, container_traj)

        # Calculate per-arm metrics
        left_arm_euclidean, right_arm_euclidean = self._calculate_per_arm_metrics(
            self._euclidean_distance, container, container_traj
        )

        return ValidationSummary(
            episode_count=num_done,
            mean_episode_return=mean_episode_return,
            mean_episode_length=mean_episode_length,
            max_timestep=max_timestep,
            early_termination_count=early_term_count,
            early_termination_rate=early_term_rate,
            err_root_xyz=err_root_xyz,
            err_root_quat=err_root_quat,
            err_root_yaw=err_root_yaw,
            err_joint_pos=err_joint_pos,
            err_joint_vel=err_joint_vel,
            err_site_abs=err_site_abs,
            err_rpos=err_rpos,
            euclidean_distance=jax.tree.map(
                lambda x, y: jnp.mean(self._euclidean_distance(x, y)) if x.size > 0 else x, container, container_traj
            ),
            dynamic_time_warping=jax.tree.map(
                lambda x, y: jnp.mean(self._dynamic_time_warping(x, y)) if x.size > 0 else x, container, container_traj
            ),
            discrete_frechet_distance=jax.tree.map(
                lambda x, y: jnp.mean(self._discrete_frechet_distance(x, y)) if x.size > 0 else x,
                container,
                container_traj,
            ),
            left_arm_euclidean_distance=left_arm_euclidean,
            right_arm_euclidean_distance=right_arm_euclidean,
        )

    def get_joint_positions(self, env_states):
        # get from data
        qpos = env_states.data.qpos

        # get from trajectory
        traj_qpos = self._traj_data.qpos[self.get_traj_indices(env_states)]
        offset = self._get_root_xy_offset(env_states)
        if offset is not None:
            traj_qpos = traj_qpos.at[..., self._root_qpos_ids_xy].add(-offset[..., :2])
        # filter for relevant joints
        qpos, traj_qpos = qpos[..., self.rel_qpos_ids], traj_qpos[..., self.rel_qpos_ids]

        # there might be quaternions due to free joints, so we need to convert them
        # to rotation vector to use metrics in the Euclidean space
        quat, quat_traj = qpos[..., self._quat_in_qpos], traj_qpos[..., self._quat_in_qpos]

        # Check if there are any quaternion joints
        if quat.size > 0:
            quat, quat_traj = quat.reshape(-1, 4), quat_traj.reshape(-1, 4)
            quat, quat_traj = quat_scalarfirst2scalarlast(quat), quat_scalarfirst2scalarlast(quat_traj)
            rot_vec, rot_vec_traj = R.from_quat(quat).as_rotvec(), R.from_quat(quat_traj).as_rotvec()
            qpos = jnp.concatenate([qpos[..., self._not_quat_in_qpos], rot_vec.reshape((*qpos.shape[:-1], 3))], axis=-1)
            traj_qpos = jnp.concatenate(
                [traj_qpos[..., self._not_quat_in_qpos], rot_vec_traj.reshape((*traj_qpos.shape[:-1], 3))], axis=-1
            )
        else:
            # No quaternion joints, just use non-quaternion positions
            qpos = qpos[..., self._not_quat_in_qpos]
            traj_qpos = traj_qpos[..., self._not_quat_in_qpos]

        return qpos, traj_qpos

    def get_joint_velocities(self, env_states):
        # get from data
        qvel = env_states.data.qvel

        # get from trajectory
        traj_qvel = self._traj_data.qvel[self.get_traj_indices(env_states)]

        return qvel[..., self.rel_qvel_ids], traj_qvel[..., self.rel_qvel_ids]

    def get_body_positions(self, env_states):
        # get from data
        body_pos = env_states.data.xpos

        # get from trajectory
        traj_body_pos = self._traj_data.xpos[self.get_traj_indices(env_states)]
        offset = self._get_root_xy_offset(env_states)
        if offset is not None:
            traj_body_pos = traj_body_pos - offset[..., None, :]

        return body_pos[..., self.rel_body_ids, :], traj_body_pos[..., self.rel_body_ids, :]

    def get_body_orientations(self, env_states):
        # get from data
        xquat_env = quat_scalarfirst2scalarlast(env_states.data.xquat)
        body_rotvec = R.from_quat(xquat_env).as_rotvec()

        # get from trajectory
        xquat_traj = quat_scalarfirst2scalarlast(self._traj_data.xquat[self.get_traj_indices(env_states)])
        traj_body_rotvec = R.from_quat(xquat_traj).as_rotvec()

        return body_rotvec[..., self.rel_body_ids, :], traj_body_rotvec[..., self.rel_body_ids, :]

    def get_body_velocities(self, env_states):
        # get from data
        body_vel = env_states.data.cvel

        # get from trajectory
        traj_body_vel = self._traj_data.cvel[self.get_traj_indices(env_states)]

        return body_vel[..., self.rel_body_ids, :], traj_body_vel[..., self.rel_body_ids, :]

    def get_site_positions(self, env_states, sim_site_idx=None):
        # sim_site_idx: indices for simulation data (0..K-1 if pre-sliced, else global IDs)
        if sim_site_idx is None:
            sim_site_idx = self.rel_site_ids
        site_pos = env_states.data.site_xpos

        # Trajectory always uses global self.rel_site_ids
        traj_indices = self.get_traj_indices(env_states)
        traj_site_pos = self._traj_data.site_xpos[traj_indices]

        if self._site_mapper.requires_mapping:
            traj_site_indices = self._site_mapper.model_ids_to_traj_indices(self.rel_site_ids)
            traj_site_pos = traj_site_pos[..., traj_site_indices, :]
        else:
            traj_site_pos = traj_site_pos[..., self.rel_site_ids, :]

        offset = self._get_root_xy_offset(env_states)
        if offset is not None:
            traj_site_pos = traj_site_pos - offset[..., None, :]

        return site_pos[..., sim_site_idx, :], traj_site_pos

    def get_site_orientations(self, env_states, sim_site_idx=None):
        # sim_site_idx: indices for simulation data (0..K-1 if pre-sliced, else global IDs)
        if sim_site_idx is None:
            sim_site_idx = self.rel_site_ids
        site_rotvec = R.from_matrix(env_states.data.site_xmat).as_rotvec()

        # Trajectory always uses global self.rel_site_ids
        site_xmat = self._traj_data.site_xmat
        assert len(site_xmat.shape) == 3
        site_xmat = site_xmat.reshape(site_xmat.shape[0], site_xmat.shape[1], 3, 3)
        traj_indices = self.get_traj_indices(env_states)
        traj_site_xmat = site_xmat[traj_indices]

        if self._site_mapper.requires_mapping:
            traj_site_indices = self._site_mapper.model_ids_to_traj_indices(self.rel_site_ids)
            traj_site_xmat = traj_site_xmat[..., traj_site_indices, :, :]
        else:
            traj_site_xmat = traj_site_xmat[..., self.rel_site_ids, :, :]

        traj_site_rotvec = R.from_matrix(traj_site_xmat).as_rotvec()

        return site_rotvec[..., sim_site_idx, :], traj_site_rotvec

    def get_site_velocities(self, env_states, sim_site_idx=None):
        # sim_site_idx: indices for simulation data (0..K-1 if pre-sliced, else global IDs)
        # Body lookups always use global self.rel_site_ids
        if sim_site_idx is None:
            sim_site_idx = self.rel_site_ids
        site_xvel = self._vec_calc_site_velocities(
            sim_site_idx,
            env_states.data,
            self._site_bodyid[self.rel_site_ids],
            self._body_rootid[self.rel_site_ids],
            jnp,
            False,
            None,
        )

        # Trajectory data - use trajectory indices for memory-optimized environments
        traj_indices = self.get_traj_indices(env_states)
        traj_data = jax.tree.map(lambda x: x[traj_indices], self._traj_data)

        if self._site_mapper.requires_mapping:
            traj_site_indices = self._site_mapper.model_ids_to_traj_indices(self.rel_site_ids)
        else:
            traj_site_indices = None

        traj_site_xvel = self._vec_calc_site_velocities(
            self.rel_site_ids,
            traj_data,
            self._site_bodyid[self.rel_site_ids],
            self._body_rootid[self.rel_site_ids],
            jnp,
            False,
            traj_site_indices,
        )
        return site_xvel, traj_site_xvel

    def get_relative_site_quantities(self, env_states, sim_site_idx=None):
        # sim_site_idx: indices for simulation data (0..K-1 if pre-sliced, else global IDs)
        # Body lookups always use global self.rel_site_ids
        if sim_site_idx is None:
            sim_site_idx = self.rel_site_ids
        rel_site_pos, rel_site_rotvec, rel_site_vel = self._vec_calc_rel_site_quantities(
            env_states.data,
            sim_site_idx,
            self._site_bodyid[self.rel_site_ids],
            self._body_rootid[self.rel_site_ids],
            jnp,
            None,
        )

        # Trajectory data - use trajectory indices for memory-optimized environments
        traj_states = env_states.additional_carry.traj_state
        traj_data = self._traj_data.get(traj_states.traj_no, traj_states.subtraj_step_no)

        if self._site_mapper.requires_mapping:
            traj_site_indices = self._site_mapper.model_ids_to_traj_indices(self.rel_site_ids)
        else:
            traj_site_indices = None

        traj_rel_site_pos, traj_rel_site_rotvec, traj_rel_site_vel = self._vec_calc_rel_site_quantities(
            traj_data,
            self.rel_site_ids,
            self._site_bodyid[self.rel_site_ids],
            self._body_rootid[self.rel_site_ids],
            jnp,
            traj_site_indices,
        )

        return (rel_site_pos, rel_site_rotvec, rel_site_vel, traj_rel_site_pos, traj_rel_site_rotvec, traj_rel_site_vel)

    def get_traj_indices(self, env_states):
        traj_states = env_states.additional_carry.traj_state
        start_idx = self._traj_data.split_points[traj_states.traj_no]
        return start_idx + traj_states.subtraj_step_no

    @property
    def _traj_data(self):
        """Access trajectory data dynamically to avoid stale references after conversion."""
        return self._trajectory_handler.traj.data if self._trajectory_handler is not None else None

    @property
    def requires_trajectory(self):
        return self._trajectory_handler is not None

    def get_zero_container(self):
        def _zeros_if_exists(quantity_name):
            return jnp.array(0.0) if quantity_name in self.quantaties else jnp.empty(0)

        container = QuantityContainer(
            qpos=_zeros_if_exists("JointPosition"),
            qvel=_zeros_if_exists("JointVelocity"),
            xpos=_zeros_if_exists("BodyPosition"),
            xrotvec=_zeros_if_exists("BodyOrientation"),
            cvel=_zeros_if_exists("BodyVelocity"),
            site_xpos=_zeros_if_exists("SitePosition"),
            site_xrotvec=_zeros_if_exists("SiteOrientation"),
            site_xvel=_zeros_if_exists("SiteVelocity"),
            site_rpos=_zeros_if_exists("RelSitePosition"),
            site_rrotvec=_zeros_if_exists("RelSiteOrientation"),
            site_rvel=_zeros_if_exists("RelSiteVelocity"),
        )

        return ValidationSummary(
            episode_count=jnp.array(0.0),
            frame_coverage=jnp.array(0.0),
            total_frame=jnp.array(0.0),
            mean_episode_return=jnp.array(0.0),
            mean_episode_length=jnp.array(0.0),
            max_timestep=jnp.array(0),
            early_termination_count=jnp.array(0.0),
            early_termination_rate=jnp.array(0.0),
            err_root_xyz=jnp.array(0.0),
            err_root_quat=jnp.array(0.0),
            err_root_yaw=jnp.array(0.0),
            err_joint_pos=jnp.array(0.0),
            err_joint_vel=jnp.array(0.0),
            err_site_abs=jnp.array(0.0),
            err_rpos=jnp.array(0.0),
            euclidean_distance=container,
            dynamic_time_warping=container,
            discrete_frechet_distance=container,
            left_arm_euclidean_distance=container,
            right_arm_euclidean_distance=container,
        )


def flatten_validation_metrics(
    validation_metrics,
    enabled_measures: list[str] | None = None,
    enabled_quantities: list[str] | None = None,
    *,
    include_trajectory_set_metrics: bool = True,
) -> dict[str, float]:
    """Flatten ValidationSummary to dict for logging.

    Args:
        validation_metrics: ValidationSummary instance
        enabled_measures: List of enabled measure names (e.g., ["EuclideanDistance"]).
                         If None, logs all measures.
        enabled_quantities: List of enabled quantity names (e.g., ["JointPosition"]).
                         If None, logs all error metrics for backward compatibility.
        include_trajectory_set_metrics: Whether to include metrics that require
            exact trajectory-set evaluation. Sampled validation leaves these
            fields at their structural defaults, so callers should pass False.
    """
    metrics: dict[str, float] = {
        "val_episode_count": float(validation_metrics.episode_count),
        "val_mean_episode_return": float(validation_metrics.mean_episode_return),
        "val_mean_episode_length": float(validation_metrics.mean_episode_length),
        "val_early_termination_count": float(validation_metrics.early_termination_count),
        "val_early_termination_rate": float(validation_metrics.early_termination_rate),
        # Root error metrics do not depend on configured quantity selection.
        "val_err_root_xyz": float(validation_metrics.err_root_xyz),
        "val_err_root_quat": float(validation_metrics.err_root_quat),
        "val_err_root_yaw": float(validation_metrics.err_root_yaw),
    }
    if include_trajectory_set_metrics:
        metrics.update(
            {
                "val_horizon_timeout_count": float(validation_metrics.horizon_timeout_count),
                "val_horizon_timeout_rate": float(validation_metrics.horizon_timeout_rate),
                "val_frame_coverage": float(validation_metrics.frame_coverage),
                "val_total_frame": float(validation_metrics.total_frame),
            }
        )
    enabled_quantities_set = set(enabled_quantities) if enabled_quantities is not None else None

    error_metric_quantities = {
        "val_err_joint_pos": ("JointPosition", validation_metrics.err_joint_pos),
        "val_err_joint_vel": ("JointVelocity", validation_metrics.err_joint_vel),
        "val_err_site_abs": ("SitePosition", validation_metrics.err_site_abs),
        "val_err_rpos": ("RelSitePosition", validation_metrics.err_rpos),
    }
    for metric_name, (required_quantity, metric_value) in error_metric_quantities.items():
        if enabled_quantities_set is None or required_quantity in enabled_quantities_set:
            metrics[metric_name] = float(metric_value)

    # Map config measure names to field names (e.g., "EuclideanDistance" -> "euclidean_distance")
    enabled_fields = None
    if enabled_measures is not None:
        enabled_fields = {
            m[0].lower() + m[1:].replace("Distance", "_distance").replace("Warping", "_warping")
            for m in enabled_measures
        }

    for field in fields(validation_metrics):
        attr = getattr(validation_metrics, field.name)
        if isinstance(attr, QuantityContainer):
            # Skip disabled measures
            if enabled_fields is not None and field.name not in enabled_fields:
                continue
            for field_attr in fields(attr):
                attr_value = getattr(attr, field_attr.name)
                if hasattr(attr_value, "ndim") and attr_value.size > 0:
                    value = float(attr_value) if attr_value.ndim == 0 else float(attr_value[0])
                    metrics[f"val_{field.name}_{field_attr.name}"] = value
    return metrics
