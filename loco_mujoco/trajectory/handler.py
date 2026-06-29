import logging
from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.trajectory.dataclasses import (
    Trajectory,
    TrajectoryCacheType,
    compute_trajectory_kinematic_caches,
    interpolate_trajectories,
    recompute_trajectory_velocities,
    trajectory_info_with_cache_metadata,
)


@struct.dataclass
class TrajState:
    traj_no: int
    subtraj_step_no: int
    subtraj_step_no_init: int


class TrajectoryHandler(StatefulObject):
    """
    General class to handle Trajectories. It filters and extends the trajectory data to match
    the current model's joints, bodies and sites. The key idea is to ensure that TrajectoryData has the same
    dimensionality and order for all its attributes as in the Mujoco data structure. So TrajectoryData is a
    simplified version of the Mujoco data structure with fewer attributes. This class also automatically
    interpolates the trajectory to the desired control frequency.

    """
    def __init__(self, model, traj_path=None, traj: Trajectory = None, control_dt=0.01, random_start=True,
                 fixed_start_conf=None, start_from_random_step=True, clip_trajectory_to_joint_ranges=False, warn=True,
                 cache_type: TrajectoryCacheType | str = TrajectoryCacheType.FULL, site_names: list[str] | None = None):
        """
        Constructor.

        Args:
            model (mjModel): Current model.
            traj_path (string): path with the trajectory for the model to follow. Should be a numpy zipped file (.npz)
                with a 'traj_data' array and possibly a 'split_points' array inside. The 'traj_data'
                should be in the shape (joints x observations). If traj_files is specified, this should be None.
            traj (Trajectory): Datastructure containing all trajectory files. If traj_path is specified, this
                should be None.
            control_dt (float): Model control frequency used to interpolate the trajectory.
            start_from_random_step (bool): If True, start from a random step in the trajectory.
            warn (bool): If True, a warning will be raised, if some trajectory ranges are violated. todo

        """

        assert (traj_path is not None) != (traj is not None), ("Please specify either traj_path or "
                                                               "trajectory, but not both.")

        # load data
        if traj_path is not None:
            traj = Trajectory.load(traj_path)

        # todo: implement this in observation types in init_from_traj!
        #self.check_if_trajectory_is_in_range(low, high, keys, joint_pos_idx, warn, clip_trajectory_to_joint_ranges)

        assert (fixed_start_conf is not None) != random_start, "Please specify either fixed_start_conf or random_start."
        self.random_start = random_start
        self.fixed_start_conf = fixed_start_conf
        self.use_fixed_start = True if fixed_start_conf is not None else False
        self.start_from_random_step = start_from_random_step

        self.control_dt = control_dt
        self.traj = materialize_trajectory(
            traj,
            model=model,
            control_dt=control_dt,
            cache_type=cache_type,
            site_names=site_names,
        )
        self._is_numpy = True if isinstance(self.traj.data.qpos, np.ndarray) else False

        # Update traj_dt from interpolated trajectory info
        self.traj_dt = 1.0 / self.traj.info.frequency
        assert np.isclose(self.traj_dt, self.control_dt, atol=1e-9), (
            f"traj_dt ({self.traj_dt}) != control_dt ({self.control_dt}) after interpolation"
        )

    def len_trajectory(self, traj_ind):
        return self.traj.data.split_points[traj_ind + 1] - self.traj.data.split_points[traj_ind]

    @property
    def n_trajectories(self):
        return len(self.traj.data.split_points) - 1

    def last_step_idx(self, traj_ind):
        """Return the last valid step index for a trajectory."""
        return self.len_trajectory(traj_ind) - 1

    def reached_trajectory_end(self, traj_state, backend):
        """Check whether a trajectory state is at or past its terminal frame."""
        last_step_idx = self.last_step_idx(traj_state.traj_no)
        if backend == jnp:
            return jnp.greater_equal(traj_state.subtraj_step_no, last_step_idx)
        return traj_state.subtraj_step_no >= last_step_idx

    @staticmethod
    def filter_and_extend(traj_data, traj_info, model):
        """
        To ensure that the data structure of the current model and the trajectory data have the same dimensionality
        and order for all supported attributes, this function filters the elements present in the trajectory but not
        the current model and extends the trajectory data's joints, bodies and sites with elements present in
        the current model but not the trajectory. It is doing so by adding dummy joints, bodies and sites to the
        trajectory data if they are not present in the trajectory data but in the model. It also reorders the
        joints, bodies and sites based on the model.

        Args:
            traj_data (TrajectoryData): Trajectory data to be filtered and extended.
            traj_info (TrajectoryInfo): Trajectory info to be filtered and extended.
            model (mjModel): Current model.

        Returns:
            TrajectoryData, TrajectoryInfo: Filtered and extended trajectory data and trajectory info.

        """

        # Special handling for MyoBimanualArm: detect if this is a MyoBimanualArm trajectory
        # by checking if the trajectory has exactly 7 sites matching MyoBimanualArm's sites_for_mimic
        is_bimanual_muscle_traj = False
        bimanual_sites = ["right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                           "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"]

        if (traj_info.site_names is not None and
            len(traj_info.site_names) == 6 and
            set(traj_info.site_names) == set(bimanual_sites)):
            is_bimanual_muscle_traj = True

        # --- filter the trajectory based on the model and data ---
        # get the joint names from current model
        joint_names = []
        joint_ids = []
        joint_name2id_qpos = dict()
        joint_name2id_qvel = dict()
        j_qpos, j_qvel = 0, 0
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            j_type = model.jnt_type[i]
            joint_names.append(name)

            if j_type == mujoco.mjtJoint.mjJNT_FREE:
                joint_name2id_qpos[name] = jnp.arange(j_qpos, j_qpos + 7)
                joint_name2id_qvel[name] = jnp.arange(j_qvel, j_qvel + 6)
                j_qpos += 7
                j_qvel += 6
            elif j_type == mujoco.mjtJoint.mjJNT_SLIDE or j_type == mujoco.mjtJoint.mjJNT_HINGE:
                joint_name2id_qpos[name] = jnp.array([j_qpos])
                joint_name2id_qvel[name] = jnp.array([j_qvel])
                j_qpos += 1
                j_qvel += 1

            joint_ids.append(i)

        # get the body names from current model
        body_names = set()
        body_name2id = dict()
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_names.add(name)
            body_name2id[name] = i

        # get the site names from current model
        site_names = set()
        site_name2id = dict()
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            site_names.add(name)
            site_name2id[name] = i

        joint_to_be_removed_qpos = dict()
        joint_to_be_removed_qvel = dict()
        for i, j_name in enumerate(traj_info.joint_names):
            if j_name not in joint_names:
                joint_to_be_removed_qpos[j_name] = traj_info.joint_name2ind_qpos[j_name]
                joint_to_be_removed_qvel[j_name] = traj_info.joint_name2ind_qvel[j_name]

        bodies_to_be_removed = dict()
        if traj_info.body_names is not None:
            for i, b_name in enumerate(traj_info.body_names):
                if b_name not in body_names:
                    bodies_to_be_removed[b_name] = i

        site_to_be_removed = dict()
        if traj_info.site_names is not None:
            for i, s_name in enumerate(traj_info.site_names):
                if s_name not in site_names:
                    site_to_be_removed[s_name] = i

        # create new traj_data and traj_info with removed joints, bodies and sites
        if joint_to_be_removed_qpos:
            qpos_ind = jnp.concatenate(list(joint_to_be_removed_qpos.values()))
            qvel_ind = jnp.concatenate(list(joint_to_be_removed_qvel.values()))
            traj_data = traj_data.remove_joints(qpos_ind, qvel_ind)
            traj_info = traj_info.remove_joints(list(joint_to_be_removed_qpos.keys()))
        if bodies_to_be_removed:
            traj_data = traj_data.remove_bodies(jnp.array(list(bodies_to_be_removed.values())))
            traj_info = traj_info.remove_bodies(list(bodies_to_be_removed.keys()))
        if site_to_be_removed:
            traj_data = traj_data.remove_sites(jnp.array(list(site_to_be_removed.values())))
            traj_info = traj_info.remove_sites(list(site_to_be_removed.keys()))

        # --- extend the trajectory data's joints, bodies and sites using the current model and data ---
        for j_name, j_id in zip(joint_names, joint_ids):
            j_type = model.jnt_type[j_id]
            if j_name not in traj_info.joint_names:
                traj_info = traj_info.add_joint(j_name, j_type)
                traj_data = traj_data.add_joint()

        if traj_info.body_names is not None:
            for b_name in body_names:
                if b_name not in traj_info.body_names:
                    b_id = body_name2id[b_name]
                    traj_info = traj_info.add_body(b_name, model.body_rootid[b_id], model.body_weldid[b_id],
                                                   model.body_mocapid[b_id], model.body_pos[b_id],
                                                   model.body_quat[b_id], model.body_ipos[b_id],
                                                   model.body_iquat[b_id])
                    traj_data = traj_data.add_body()

        if traj_info.site_names is not None:
            # For muscle models (MyoBimanualArm, MyoFullBody), preserve filtered site lists
            # and skip adding sites from the model to maintain memory optimization
            has_many_model_sites = len(site_names) > len(traj_info.site_names) * 10  # Model has 10x more sites than trajectory
            is_muscle_model = is_bimanual_muscle_traj or has_many_model_sites

            if not is_muscle_model:
                # Only add sites for non-muscle models
                for s_name in site_names:
                    if s_name not in traj_info.site_names:
                        s_id = site_name2id[s_name]
                        traj_info = traj_info.add_site(s_name, model.site_pos[s_id], model.site_quat[s_id],
                                                       model.site_bodyid[s_id])
                        traj_data = traj_data.add_site()
            else:
                print(f"[TrajectoryHandler] Skipping site addition for muscle model - trajectory has {len(traj_info.site_names)} mimic sites, model has {len(site_names)} total sites")

        # --- reorder the joints and bodies based on the model ---
        new_joint_order_names = []
        new_joint_order_ids_qpos = []
        new_joint_order_ids_qvel = []
        for j_name in joint_names:
            new_joint_order_names.append(traj_info.joint_names.index(j_name))
            new_joint_order_ids_qpos.append(traj_info.joint_name2ind_qpos[j_name])
            new_joint_order_ids_qvel.append(traj_info.joint_name2ind_qvel[j_name])

        if traj_info.body_names is not None:
            new_body_order = []
            for b_name in body_name2id.keys():
                new_body_order.append(traj_info.body_names.index(b_name))

        if traj_info.site_names is not None:
            new_site_order = []
            has_many_model_sites = len(site_names) > len(traj_info.site_names) * 10  # Model has 10x more sites than trajectory
            is_muscle_model = is_bimanual_muscle_traj or has_many_model_sites

            if is_muscle_model:
                # For muscle models, maintain the existing order of mimic sites
                # without trying to reorder to match all model sites
                new_site_order = list(range(len(traj_info.site_names)))
            else:
                # For other models, reorder to match the model's site order
                for s_name in site_name2id.keys():
                    new_site_order.append(traj_info.site_names.index(s_name))

        traj_info = traj_info.reorder_joints(new_joint_order_names)
        traj_info = traj_info.reorder_bodies(new_body_order) if traj_info.body_names is not None else traj_info
        traj_info = traj_info.reorder_sites(new_site_order) if traj_info.site_names is not None else traj_info
        traj_data = traj_data.reorder_joints(jnp.concatenate(new_joint_order_ids_qpos),
                                             jnp.concatenate(new_joint_order_ids_qvel))
        traj_data = traj_data.reorder_bodies(jnp.array(new_body_order)) \
            if traj_info.body_names is not None else traj_data
        traj_data = traj_data.reorder_sites(jnp.array(new_site_order)) \
            if traj_info.site_names is not None else traj_data

        return traj_data, traj_info

    def init_state(self, env, key, model, data, backend):
        return TrajState(0, 0, 0)

    def reset_state(self, env, model, data, carry, backend):

        key = carry.key

        def _selected_idx():
            traj_idx = backend.asarray(carry.selected_traj_idx, dtype=backend.int32)
            return traj_idx, backend.asarray(0, dtype=backend.int32)

        def _random_start_idx():
            if backend == jnp:
                new_key, _k1, _k2 = jax.random.split(key, 3)

                weights = getattr(carry, "sampling_weights", None)
                if weights is not None:
                    traj_idx = jax.random.choice(_k1, self.n_trajectories, p=weights)
                else:
                    traj_idx = jax.random.randint(_k1, shape=(), minval=0, maxval=self.n_trajectories)

                if self.start_from_random_step:
                    subtraj_step_idx = jax.random.randint(_k2, shape=(), minval=0, maxval=self.len_trajectory(traj_idx))
                else:
                    subtraj_step_idx = jnp.int32(0)
                return new_key, traj_idx, subtraj_step_idx

            weights = getattr(carry, "sampling_weights", None)
            if weights is not None:
                traj_idx = np.random.choice(self.n_trajectories, p=np.asarray(weights))
            else:
                traj_idx = np.random.randint(0, self.n_trajectories)
            if self.start_from_random_step:
                subtraj_step_idx = np.random.randint(0, self.len_trajectory(traj_idx))
            else:
                subtraj_step_idx = 0
            return key, traj_idx, subtraj_step_idx

        def _fixed_start_idx():
            traj_idx, start_step = self.fixed_start_conf
            return key, backend.asarray(traj_idx, dtype=backend.int32), backend.asarray(start_step, dtype=backend.int32)

        def _default_idx():
            if self.random_start:
                return _random_start_idx()
            if self.use_fixed_start:
                return _fixed_start_idx()
            return key, backend.asarray(0, dtype=backend.int32), backend.asarray(0, dtype=backend.int32)

        selected_traj_idx = getattr(carry, "selected_traj_idx", None)
        if backend == jnp and selected_traj_idx is not None:
            key, traj_idx, subtraj_step_idx = jax.lax.cond(
                selected_traj_idx >= 0,
                lambda _: (key, *_selected_idx()),
                lambda _: _default_idx(),
                operand=None,
            )
        elif selected_traj_idx is not None and int(selected_traj_idx) >= 0:
            traj_idx, subtraj_step_idx = _selected_idx()
        else:
            key, traj_idx, subtraj_step_idx = _default_idx()

        self.new_traj_no = traj_idx
        new_subtraj_step_no = subtraj_step_idx
        self.new_subtraj_step_no_init = new_subtraj_step_no

        return data, carry.replace(key=key, traj_state=TrajState(self.new_traj_no, new_subtraj_step_no,
                                                                 self.new_subtraj_step_no_init))

    def update_state(self, env, model, data, carry, backend):
        traj_state = carry.traj_state
        last_step_idx = self.last_step_idx(traj_state.traj_no)

        # Episode boundaries are handled by env reset_state(). During a step we only
        # advance within the current trajectory so the terminal frame remains
        # self-consistent for reward, visuals, and done detection.
        if backend == jnp:
            next_subtraj_step_no = jnp.minimum(
                traj_state.subtraj_step_no + 1,
                jnp.asarray(last_step_idx, dtype=traj_state.subtraj_step_no.dtype),
            )
        else:
            next_subtraj_step_no = min(traj_state.subtraj_step_no + 1, last_step_idx)

        traj_state = traj_state.replace(subtraj_step_no=next_subtraj_step_no)
        return carry.replace(traj_state=traj_state)

    def get_current_traj_data(self, carry, backend):
        traj_no = carry.traj_state.traj_no
        subtraj_step_no = carry.traj_state.subtraj_step_no
        return self.traj.data.get(traj_no, subtraj_step_no, backend)

    def get_init_traj_data(self, carry, backend):
        traj_no = carry.traj_state.traj_no
        subtraj_step_no_init = carry.traj_state.subtraj_step_no_init
        return self.traj.data.get(traj_no, subtraj_step_no_init, backend)

    def get_traj_data_at(self, traj_no, subtraj_step_no, carry, backend):
        """Get trajectory data at a specific trajectory index and step."""
        return self.traj.data.get(traj_no, subtraj_step_no, backend)

    def to_numpy(self):
        if not self._is_numpy:
            traj_model = self.traj.info.model.to_numpy()
            traj_info = replace(self.traj.info, model=traj_model)
            self.traj = replace(self.traj, data=self.traj.data.to_numpy(), info=traj_info)
            self._is_numpy = True

    def to_jax(self):
        if self._is_numpy:
            traj_model = self.traj.info.model.to_jax()
            traj_info = replace(self.traj.info, model=traj_model)
            self.traj = replace(self.traj, data=self.traj.data.to_jax(), info=traj_info)
            self._is_numpy = False

    @property
    def is_numpy(self):
        return self._is_numpy


def materialize_trajectory(
    traj: Trajectory,
    model: "mujoco.MjModel",
    control_dt: float,
    cache_type: TrajectoryCacheType | str = TrajectoryCacheType.FULL,
    site_names: list[str] | None = None,
) -> Trajectory:
    cache_type = TrajectoryCacheType(cache_type)
    backend = np if isinstance(traj.data.qpos, np.ndarray) else jnp

    traj_data, traj_info = TrajectoryHandler.filter_and_extend(traj.data, traj.info, model)

    original_frequency = traj_info.frequency
    target_frequency = 1.0 / control_dt
    original_samples = int(np.asarray(traj_data.n_samples))
    traj_dt = 1.0 / traj_info.frequency

    print(
        f"[TrajectoryHandler] traj_dt={traj_dt:.6f} ({traj_info.frequency:.1f}Hz), "
        f"control_dt={control_dt:.6f} ({target_frequency:.1f}Hz), cache={cache_type.value}"
    )

    if not np.isclose(traj_dt, control_dt, atol=1e-9):
        traj_data, traj_info = interpolate_trajectories(traj_data, traj_info, target_frequency, backend=backend)
        print(
            f"[TrajectoryHandler] INFO: Interpolated trajectory from {original_frequency:.1f} Hz "
            f"to {target_frequency:.1f} Hz ({original_samples} -> {int(np.asarray(traj_data.n_samples))} samples)"
        )
        logger = logging.getLogger("trajectory")
        if logger.handlers:
            logger.info(
                "Interpolated trajectory from %.1f Hz to %.1f Hz (%d -> %d samples)",
                original_frequency,
                target_frequency,
                original_samples,
                int(np.asarray(traj_data.n_samples)),
            )

    traj_data = recompute_trajectory_velocities(traj_data, traj_info, model, backend=backend)
    traj_data = compute_trajectory_kinematic_caches(
        traj_data,
        model=model,
        cache_type=cache_type,
        site_names=site_names,
        backend=backend,
    )
    traj_info = trajectory_info_with_cache_metadata(
        traj_info,
        model=model,
        cache_type=cache_type,
        site_names=site_names,
        backend=backend,
    )
    print(f"[TrajectoryHandler] INFO: Materialized {cache_type.value} trajectory cache at {traj_info.frequency:.1f} Hz")

    return replace(traj, data=traj_data, info=traj_info)
