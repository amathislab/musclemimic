from types import ModuleType
from typing import Any, Dict, Tuple, Union

import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model
from flax import struct
import numpy as np
import jax.numpy as jnp
from jax._src.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.reward.base import Reward
from loco_mujoco.core.utils import mj_jntname2qposid, mj_jntname2qvelid, mj_jntid2qposid, mj_jntid2qvelid
from loco_mujoco.core.utils.math import calculate_relative_site_quantities, quaternion_angular_distance
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.reward.utils import out_of_bounds_action_cost
from musclemimic.core.utils.site_mapping import create_site_mapper


def check_traj_provided(method):
    """
    Decorator to check if trajectory handler is None. Raises ValueError if not provided.
    """
    def wrapper(self, *args, **kwargs):
        env = kwargs.get('env', None) if 'env' in kwargs else args[5]  # Assumes 'env' is the 6th positional argument
        if getattr(env, "th") is None:
            raise ValueError("TrajectoryHandler not provided, but required for trajectory-based rewards.")
        return method(self, *args, **kwargs)
    return wrapper


def quat_to_yaw(quat, backend):
    """Extract yaw from quaternion [w,x,y,z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return backend.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class TrajectoryBasedReward(Reward):

    """
    Base class for trajectory-based reward functions. These reward functions require a
    trajectory handler to compute the reward.

    """

    @property
    def requires_trajectory(self) -> bool:
        return True


class TargetVelocityTrajReward(TrajectoryBasedReward):
    """
    Reward function that computes the reward based on the deviation from the trajectory velocity. The trajectory
    velocity is provided as an observation in the environment. The reward is computed as the negative exponential
    of the squared difference between the current velocity and the goal velocity. The reward is computed for the
    x, y, and yaw velocities of the root.

    """

    def __init__(self, env: Any,
                 w_exp=10.0,
                 **kwargs):
        """
        Initialize the reward function.

        Args:
            env (Any): Environment instance.
            w_exp (float, optional): Exponential weight for the reward. Defaults to 10.0.
            **kwargs (Any): Additional keyword arguments.
        """

        super().__init__(env, **kwargs)
        self._free_jnt_name = self._info_props["root_free_joint_xml_name"]
        self._free_joint_qpos_idx = np.array(mj_jntname2qposid(self._free_jnt_name, env._model))
        self._free_joint_qvel_idx = np.array(mj_jntname2qvelid(self._free_jnt_name, env._model))
        self._w_exp = w_exp

    @check_traj_provided
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Computes a tracking reward based on the deviation from the trajectory velocity.
        Tracking is done on the x, y, and yaw velocities of the root.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        if backend == np:
            R = np_R
        else:
            R = jnp_R

        def calc_local_vel(_d):
            _lin_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[:3]
            _ang_vel_global = backend.squeeze(_d.qvel[self._free_joint_qvel_idx])[3:]
            _root_quat = R.from_quat(quat_scalarfirst2scalarlast(backend.squeeze(_d.qpos[self._free_joint_qpos_idx])[3:7]))
            _lin_vel_local = _root_quat.as_matrix().T @ _lin_vel_global
            # construct vel, x, y and yaw
            return backend.concatenate([_lin_vel_local[:2], backend.atleast_1d(_ang_vel_global[2])])

        # get root velocity from data
        vel_local = calc_local_vel(data)

        # calculate the same for the trajectory
        traj_data = env.th.get_current_traj_data(carry, backend)
        traj_vel_local = calc_local_vel(traj_data)

        # calculate tracking reward
        tracking_reward = backend.exp(-self._w_exp*backend.mean(backend.square(vel_local - traj_vel_local)))

        # set nan values to 0
        tracking_reward = backend.nan_to_num(tracking_reward, nan=0.0)

        reward_info = {"reward_total": tracking_reward}
        return tracking_reward, carry, reward_info


@struct.dataclass
class MimicRewardState:
    """
    State of MimicReward.
    """
    last_qvel: Union[np.ndarray, jnp.ndarray]
    last_action: Union[np.ndarray, jnp.ndarray]
    imitation_error_total: float = 0.0  # Raw weighted sum of distances for adaptive sampling


class MimicReward(TrajectoryBasedReward):
    """
    DeepMimic reward function that computes the reward based on the deviation from the trajectory. The reward is
    computed as the negative exponential of the squared difference between the current state and the trajectory state.
    The reward is computed for the joint positions, joint velocities, root position, relative site positions,
    relative site orientations, and relative site velocities. These sites are specified in the environment properties
    and are placed at key points on the body to mimic the motion of the body.

    """

    def __init__(self, env: Any,
                 sites_for_mimic=None,
                 joints_for_mimic=None,
                 **kwargs):
        """
        Initialize the DeepMimic reward function.

        Args:
            env (Any): Environment instance.
            sites_for_mimic (List[str], optional): List of site names to mimic. Defaults to None, taking all.
            joints_for_mimic (List[str], optional): List of joint names to mimic. Defaults to None, taking all.
            **kwargs (Any): Additional keyword arguments.

        """

        super().__init__(env, **kwargs)

        # reward coefficients
        self._qpos_w_exp = kwargs.get("qpos_w_exp", 10.0)
        self._qvel_w_exp = kwargs.get("qvel_w_exp", 2.0)
        self._root_pos_w_exp = kwargs.get("root_pos_w_exp", 10.0)
        self._root_quat_w_exp = kwargs.get("root_quat_w_exp", 10.0)
        self._rpos_w_exp = kwargs.get("rpos_w_exp", 100.0)
        self._rquat_w_exp = kwargs.get("rquat_w_exp", 10.0)
        self._rvel_w_exp = kwargs.get("rvel_w_exp", 0.1)
        self._qpos_w_sum = kwargs.get("qpos_w_sum", 0.0)
        self._qvel_w_sum = kwargs.get("qvel_w_sum", 0.0)
        self._root_pos_w_sum = kwargs.get("root_pos_w_sum", 0.0)
        self._root_quat_w_sum = kwargs.get("root_quat_w_sum", 0.0)
        self._rpos_w_sum = kwargs.get("rpos_w_sum", 0.5)
        self._rquat_w_sum = kwargs.get("rquat_w_sum", 0.3)
        self._rvel_w_sum = kwargs.get("rvel_w_sum", 0.0)
        self._action_out_of_bounds_coeff = kwargs.get("action_out_of_bounds_coeff", 0.01)
        self._joint_acc_coeff = kwargs.get("joint_acc_coeff", 0.0)
        self._joint_torque_coeff = kwargs.get("joint_torque_coeff", 0.0)
        self._action_rate_coeff = kwargs.get("action_rate_coeff", 0.0)
        self._activation_energy_coeff = kwargs.get("activation_energy_coeff", 0.0)
        # Root velocity tracking: [vx_local, vy_local, yaw_rate]
        self._root_vel_w_exp = kwargs.get("root_vel_w_exp", 10.0)
        self._root_vel_w_sum = kwargs.get("root_vel_w_sum", 0.2)
        self._global_root_tracking = bool(kwargs.get("global_root_tracking", False))
        self._joint_only_qpos_qvel = bool(kwargs.get("joint_only_qpos_qvel", False))
        self._root_velocity_frame = kwargs.get("root_velocity_frame", "local")
        if self._root_velocity_frame not in {"local", "global"}:
            raise ValueError("root_velocity_frame must be 'local' or 'global'.")

        # Parallel environment reward calculation mode
        # True: use mean(exp(-beta * dist)) - better for parallel environments  
        # False: use exp(-beta * mean(dist)) - current behavior (backward compatible)
        self._use_mean_exp_reward = kwargs.get("use_mean_exp_reward", False)

        # get main body name of the environment
        self.main_body_name = self._info_props["upper_body_xml_name"]
        model = env._model
        self.main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.main_body_name)
        rel_site_names = self._info_props["sites_for_mimic"] if sites_for_mimic is None else sites_for_mimic
        self._rel_site_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                                       for name in rel_site_names])
        self._rel_body_ids = np.array([model.site_bodyid[site_id] for site_id in self._rel_site_ids])
       
        # determine qpos and qvel indices
        quat_in_qpos = []
        qpos_ind = []
        qvel_ind = []
        for i in range(model.njnt):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joints_for_mimic is None or jnt_name in joints_for_mimic:
                qposid = mj_jntid2qposid(i, model)
                qvelid = mj_jntid2qvelid(i, model)
                qpos_ind.append(qposid)
                qvel_ind.append(qvelid)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    quat_in_qpos.append(qposid[3:])
        self._qpos_ind = np.concatenate(qpos_ind)
        self._qvel_ind = np.concatenate(qvel_ind)
        
        # Handle case where there are no free joints (e.g., bimanual models)
        if len(quat_in_qpos) > 0:
            quat_in_qpos = np.concatenate(quat_in_qpos)
        else:
            quat_in_qpos = np.array([], dtype=int)
        self._quat_in_qpos = np.array([True if q in quat_in_qpos else False for q in self._qpos_ind])

        # calc mask for the root free joint velocities (handle case where it doesn't exist)
        self._free_joint_qpos_ind = None
        self._joint_qpos_mask = np.ones(len(self._qpos_ind), dtype=bool)
        self._joint_qvel_mask = np.ones(len(self._qvel_ind), dtype=bool)
        try:
            self._free_joint_qpos_ind = np.array(mj_jntname2qposid(self._info_props["root_free_joint_xml_name"], model))
            self._free_joint_qvel_ind = np.array(mj_jntname2qvelid(self._info_props["root_free_joint_xml_name"], model))
            self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
            self._free_joint_qvel_mask[self._free_joint_qvel_ind] = True
            # Masks for excluding root from joint errors
            self._joint_qpos_mask = ~np.isin(self._qpos_ind, self._free_joint_qpos_ind)
            self._joint_qvel_mask = ~np.isin(self._qvel_ind, self._free_joint_qvel_ind)
        except (KeyError, ValueError):
            # For bimanual models without a free joint, create empty mask
            self._free_joint_qvel_ind = np.array([], dtype=int)
            self._free_joint_qvel_mask = np.zeros(model.nv, dtype=bool)
        
        # Initialize site mapper for trajectory index mapping
        env_sites_for_mimic = getattr(env, 'sites_for_mimic', [])
        traj_site_names = env.th.traj.info.site_names if (hasattr(env, 'th') and env.th is not None) else None
        self._site_mapper = create_site_mapper(model, env.__class__.__name__, env_sites_for_mimic, traj_site_names)

        # Root XY indices for offset correction. When episodes start at random XY positions,
        # trajectory qpos is in world frame while simulation resets to origin. We subtract
        # the init XY offset so qpos values are compared in local frame.
        self._root_qpos_ids_xy = None
        self._root_xy_in_qpos_ind = None
        root_joint_name = self._info_props.get("root_free_joint_xml_name")
        if root_joint_name:
            try:
                root_qpos_ids = np.array(mj_jntname2qposid(root_joint_name, model))
                if root_qpos_ids.size >= 2:
                    self._root_qpos_ids_xy = root_qpos_ids[:2]
                    xy_in_ind = np.where(np.isin(self._qpos_ind, self._root_qpos_ids_xy))[0]
                    if xy_in_ind.size == 2:
                        self._root_xy_in_qpos_ind = xy_in_ind
            except Exception:
                pass


    def init_state(self, env: Any,
                   key: Any,
                   model: Union[MjModel, Model],
                   data: Union[MjData, Data],
                   backend: ModuleType):
        """
        Initialize the reward state.

        Args:
            env (Any): The environment instance.
            key (Any): Key for the reward state.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            MimicRewardState: The initialized reward state.

        """
        return MimicRewardState(last_qvel=data.qvel, last_action=backend.zeros(env.info.action_space.shape[0]))

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType):
        """
        Reset the reward state.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated data and carry.

        """
        reward_state = self.init_state(env, None, model, data, backend)
        carry = carry.replace(reward_state=reward_state)
        return data, carry

    @check_traj_provided
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        Computes a deep mimic tracking reward based on the deviation from the trajectory. The reward is computed as the
        negative exponential of the squared difference between the current state and the trajectory state. The reward
        is computed for the joint positions, joint velocities, relative site positions, relative site orientations, and
        relative site velocities.

        Args:
            state (Union[np.ndarray, jnp.ndarray]): Last state.
            action (Union[np.ndarray, jnp.ndarray]): Applied action.
            next_state (Union[np.ndarray, jnp.ndarray]): Current state.
            absorbing (bool): Whether the state is absorbing.
            info (Dict[str, Any]): Additional information.
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Additional carry.
            backend (ModuleType): Backend module used for computation (either numpy or jax.numpy).

        Returns:
            Tuple[float, Any]: The reward for the current transition and the updated carry.

        Raises:
            ValueError: If trajectory handler is not provided.

        """
        # Ensure site mapper matches actual trajectory order (handles creation-before-trajectory case)
        if self._site_mapper.requires_mapping and hasattr(env, 'th') and env.th is not None:
            # Always attach (idempotent) to avoid stale mappings
            self._site_mapper.attach_trajectory_sites(env.th.traj.info.site_names)

        # get current reward state
        reward_state = carry.reward_state

        # Get dynamic reward weights from carry (set by reward curriculum)
        # This allows the curriculum to adjust velocity reward weights during training
        qvel_w_sum = carry.qvel_w_sum
        root_vel_w_sum = carry.root_vel_w_sum

        # get trajectory data
        # get all quantities from trajectory
        traj_data_single = env.th.get_current_traj_data(carry, backend)
        qpos_traj_full = traj_data_single.qpos[self._qpos_ind]
        qvel_traj_full = traj_data_single.qvel[self._qvel_ind]

        # Subtract init XY offset from trajectory qpos to compare in local frame
        xy_offset = None
        if (
            not self._global_root_tracking
            and self._root_qpos_ids_xy is not None
            and hasattr(carry.traj_state, "subtraj_step_no_init")
        ):
            init_data = env.th.get_traj_data_at(
                carry.traj_state.traj_no, carry.traj_state.subtraj_step_no_init, carry, backend
            )
            xy_offset = init_data.qpos[self._root_qpos_ids_xy]
            if self._root_xy_in_qpos_ind is not None:
                if backend == np:
                    qpos_traj_full = qpos_traj_full.copy()
                    qpos_traj_full[self._root_xy_in_qpos_ind] -= xy_offset
                else:
                    qpos_traj_full = qpos_traj_full.at[self._root_xy_in_qpos_ind].add(-xy_offset)

        qpos_full = data.qpos[self._qpos_ind]
        qvel_full = data.qvel[self._qvel_ind]
        if self._joint_only_qpos_qvel:
            qpos = qpos_full[self._joint_qpos_mask]
            qpos_traj = qpos_traj_full[self._joint_qpos_mask]
            qvel = qvel_full[self._joint_qvel_mask]
            qvel_traj = qvel_traj_full[self._joint_qvel_mask]
            quat_in_qpos = self._quat_in_qpos[self._joint_qpos_mask]
        else:
            qpos, qpos_traj = qpos_full, qpos_traj_full
            qvel, qvel_traj = qvel_full, qvel_traj_full
            quat_in_qpos = self._quat_in_qpos

        # Handle quaternion joints if they exist
        qpos_quat_traj = qpos_traj[quat_in_qpos]
        if qpos_quat_traj.size > 0:
            qpos_quat_traj = qpos_quat_traj.reshape(-1, 4)
        
        if len(self._rel_site_ids) > 1:
            # For trajectory data access, use trajectory indices to handle memory-optimized environments
            if self._site_mapper.requires_mapping:
                traj_site_indices = self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
            else:
                traj_site_indices = None
            
            site_rpos_traj, site_rangles_traj, site_rvel_traj =\
                calculate_relative_site_quantities(traj_data_single, self._rel_site_ids,
                                                self._rel_body_ids, model.body_rootid, backend,
                                                trajectory_site_indices=traj_site_indices)

        # Handle quaternion joints if they exist
        qpos_quat = qpos[quat_in_qpos]
        if qpos_quat.size > 0:
            qpos_quat = qpos_quat.reshape(-1, 4)
        
        if len(self._rel_site_ids) > 1:
            # For MyoBimanualArm, we use model site IDs for current data (not trajectory indices)
            site_rpos, site_rangles, site_rvel = (
                calculate_relative_site_quantities(data, self._rel_site_ids, self._rel_body_ids,
                                                model.body_rootid, backend))

        # Calculate distances and rewards
        if self._use_mean_exp_reward:
            # Better for parallel environments: mean(exp(-beta * dist))
            # Apply exp first, then mean across joint/site dimensions
            qpos_dists = backend.square(qpos[~quat_in_qpos] - qpos_traj[~quat_in_qpos])
            
            # Add quaternion distance to maintain same structure as original  
            if qpos_quat.size > 0:
                quat_dists = quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend)
                qpos_dists = qpos_dists + backend.mean(quat_dists)  # Add mean quat dist like original
            
            qpos_reward = backend.mean(backend.exp(-self._qpos_w_exp * qpos_dists))
            
            qvel_dists = backend.square(qvel - qvel_traj)
            qvel_reward = backend.mean(backend.exp(-self._qvel_w_exp * qvel_dists))
            
            if len(self._rel_site_ids) > 1:
                rpos_dists = backend.square(site_rpos - site_rpos_traj)
                rpos_reward = backend.mean(backend.exp(-self._rpos_w_exp * rpos_dists))
                
                rangles_dists = backend.square(site_rangles - site_rangles_traj)
                rangles_reward = backend.mean(backend.exp(-self._rquat_w_exp * rangles_dists))
                
                rvel_rot_dists = backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3])
                rvel_rot_reward = backend.mean(backend.exp(-self._rvel_w_exp * rvel_rot_dists))
                
                rvel_lin_dists = backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:])
                rvel_lin_reward = backend.mean(backend.exp(-self._rvel_w_exp * rvel_lin_dists))

            # Compute raw scalar distances for adaptive sampling (mean of per-element squared dists)
            raw_qpos_dist = backend.mean(qpos_dists)
            raw_qvel_dist = backend.mean(qvel_dists)
            if len(self._rel_site_ids) > 1:
                raw_rpos_dist = backend.mean(rpos_dists)
                raw_rangles_dist = backend.mean(rangles_dists)
                raw_rvel_rot_dist = backend.mean(rvel_rot_dists)
                raw_rvel_lin_dist = backend.mean(rvel_lin_dists)
            else:
                raw_rpos_dist = raw_rangles_dist = raw_rvel_rot_dist = raw_rvel_lin_dist = 0.0
        else:
            # Backward compatible: exp(-beta * mean(dist)) - original structure
            qpos_dist = backend.mean(backend.square(qpos[~quat_in_qpos] - qpos_traj[~quat_in_qpos]))
            
            # Add quaternion distance only if quaternion joints exist
            if qpos_quat.size > 0:
                qpos_dist += backend.mean(quaternion_angular_distance(qpos_quat, qpos_quat_traj, backend))
            
            qvel_dist = backend.mean(backend.square(qvel - qvel_traj))
            if len(self._rel_site_ids) > 1:
                rpos_dist = backend.mean(backend.square(site_rpos - site_rpos_traj))
                rangles_dist = backend.mean(backend.square(site_rangles - site_rangles_traj))
                rvel_rot_dist = backend.mean(backend.square(site_rvel[:,:3] - site_rvel_traj[:,:3]))
                rvel_lin_dist = backend.mean(backend.square(site_rvel[:,3:] - site_rvel_traj[:,3:]))

            # calculate rewards
            qpos_reward = backend.exp(-self._qpos_w_exp*qpos_dist)
            qvel_reward = backend.exp(-self._qvel_w_exp*qvel_dist)
            if len(self._rel_site_ids) > 1:
                rpos_reward = backend.exp(-self._rpos_w_exp*rpos_dist)
                rangles_reward = backend.exp(-self._rquat_w_exp*rangles_dist)
                rvel_rot_reward = backend.exp(-self._rvel_w_exp*rvel_rot_dist)
                rvel_lin_reward = backend.exp(-self._rvel_w_exp*rvel_lin_dist)

            # Use existing scalar distances for adaptive sampling
            raw_qpos_dist = qpos_dist
            raw_qvel_dist = qvel_dist
            if len(self._rel_site_ids) > 1:
                raw_rpos_dist = rpos_dist
                raw_rangles_dist = rangles_dist
                raw_rvel_rot_dist = rvel_rot_dist
                raw_rvel_lin_dist = rvel_lin_dist
            else:
                raw_rpos_dist = raw_rangles_dist = raw_rvel_rot_dist = raw_rvel_lin_dist = 0.0

        # Root position tracking reward.
        root_pos_reward = 0.0
        root_quat_reward = 0.0
        raw_root_pos_dist = 0.0
        raw_root_quat_dist = 0.0
        offset_xyz = None
        if self._free_joint_qpos_ind is not None:
            root_xyz = data.qpos[self._free_joint_qpos_ind[:3]]
            traj_root_xyz = traj_data_single.qpos[self._free_joint_qpos_ind[:3]]
            if xy_offset is not None:
                offset_xyz = backend.concatenate([xy_offset, backend.zeros(1, dtype=xy_offset.dtype)])
                traj_root_xyz = traj_root_xyz - offset_xyz
            raw_root_pos_dist = backend.mean(backend.square(root_xyz - traj_root_xyz))
            root_pos_reward = backend.exp(-self._root_pos_w_exp * raw_root_pos_dist)
            root_quat = data.qpos[self._free_joint_qpos_ind[3:7]]
            traj_root_quat = traj_data_single.qpos[self._free_joint_qpos_ind[3:7]]
            root_angle_error = quaternion_angular_distance(
                quat_scalarfirst2scalarlast(root_quat)[None, :],
                quat_scalarfirst2scalarlast(traj_root_quat)[None, :],
                backend,
            )
            raw_root_quat_dist = backend.mean(backend.square(root_angle_error))
            root_quat_reward = backend.exp(-self._root_quat_w_exp * raw_root_quat_dist)

        # Compute total raw imitation error for adaptive sampling (weighted sum of raw distances)
        imitation_error_total = (
            self._qpos_w_sum * raw_qpos_dist +
            qvel_w_sum * raw_qvel_dist +
            self._root_pos_w_sum * raw_root_pos_dist +
            self._root_quat_w_sum * raw_root_quat_dist +
            self._rpos_w_sum * raw_rpos_dist +
            self._rquat_w_sum * raw_rangles_dist +
            self._rvel_w_sum * raw_rvel_rot_dist +
            self._rvel_w_sum * raw_rvel_lin_dist
        )

        # Root velocity tracking reward
        # Always compute if free joint exists (weight from carry handles enable/disable)
        root_vel_reward = 0.0
        if self._free_joint_qpos_ind is not None:
            if backend == np:
                R = np_R
            else:
                R = jnp_R

            def calc_root_vel(_d):
                lin_vel_global = _d.qvel[self._free_joint_qvel_ind][:3]
                ang_vel_global = _d.qvel[self._free_joint_qvel_ind][3:]
                if self._root_velocity_frame == "global":
                    return backend.concatenate([lin_vel_global, ang_vel_global])
                root_quat = R.from_quat(quat_scalarfirst2scalarlast(_d.qpos[self._free_joint_qpos_ind][3:7]))
                lin_vel_local = root_quat.as_matrix().T @ lin_vel_global
                # Include all 6 DOF: XYZ linear velocity + XYZ angular velocity
                return backend.concatenate([lin_vel_local, ang_vel_global])

            root_vel = calc_root_vel(data)
            traj_root_vel = calc_root_vel(traj_data_single)
            root_vel_dist = backend.mean(backend.square(root_vel - traj_root_vel))
            root_vel_reward = backend.exp(-self._root_vel_w_exp * root_vel_dist)

        # calculate costs
        # out of bounds action cost
        if self._action_out_of_bounds_coeff > 0.0:
            out_of_bound_reward = -out_of_bounds_action_cost(action, lower_bound=env.mdp_info.action_space.low,
                                                             upper_bound=env.mdp_info.action_space.high, backend=backend)
        else:
            out_of_bound_reward = 0.0

        # joint acceleration penalty
        if self._joint_acc_coeff > 0.0:
            last_joint_vel = reward_state.last_qvel[~self._free_joint_qvel_mask]
            joint_vel = data.qvel[~self._free_joint_qvel_mask]
            acceleration_norm = backend.sum(backend.square(joint_vel - last_joint_vel) / env.dt)
            acceleration_penalty = -acceleration_norm
        else:
            acceleration_penalty = 0.0

        # joint torque penalty
        if self._joint_torque_coeff > 0.0:
            torque_norm = backend.sum(backend.square(data.qfrc_actuator[~self._free_joint_qvel_mask]))
            torque_penalty = -torque_norm
        else:
            torque_penalty = 0.0

        # action rate penalty
        if self._action_rate_coeff > 0.0:
            action_rate_norm = backend.sum(backend.square(action - reward_state.last_action))
            action_rate_penalty = -action_rate_norm
        else:
            action_rate_penalty = 0.0

        # activation energy penalty
        if self._activation_energy_coeff > 0.0:
            activation_energy = backend.mean(backend.square(data.act))
            activation_energy_penalty = -activation_energy
        else:
            activation_energy_penalty = 0.0

        # total penalties (coefficient applied once here)
        total_penalities = (self._action_out_of_bounds_coeff * out_of_bound_reward
                            + self._joint_acc_coeff * acceleration_penalty
                            + self._joint_torque_coeff * torque_penalty
                            + self._action_rate_coeff * action_rate_penalty
                            + self._activation_energy_coeff * activation_energy_penalty)
        total_penalities = backend.maximum(total_penalities, -1.0)

        # calculate total reward
        total_reward = (self._qpos_w_sum * qpos_reward + qvel_w_sum * qvel_reward
                        + self._root_pos_w_sum * root_pos_reward
                        + self._root_quat_w_sum * root_quat_reward
                        + root_vel_w_sum * root_vel_reward)
        if len(self._rel_site_ids) > 1:
            total_reward = (total_reward
                        + self._rpos_w_sum * rpos_reward + self._rquat_w_sum * rangles_reward
                        + self._rvel_w_sum * rvel_rot_reward + self._rvel_w_sum * rvel_lin_reward)

        total_reward = total_reward + total_penalities

        # clip to positive values
        total_reward = backend.maximum(total_reward, 0.0)

        # set nan values to 0
        total_reward = backend.nan_to_num(total_reward, nan=0.0)

        # update reward state
        reward_state = reward_state.replace(
            last_qvel=data.qvel,
            last_action=action,
            imitation_error_total=imitation_error_total,
        )
        carry = carry.replace(reward_state=reward_state)

        # Diagnostic error metrics (raw errors, not exp-transformed)
        err_root_xyz = err_root_yaw = err_joint_pos = err_joint_vel = err_site_abs = err_rpos = 0.0
        if self._free_joint_qpos_ind is not None:
            # Root XYZ error (world frame, with offset correction)
            err_root_xyz = backend.sqrt(raw_root_pos_dist)
            # Root yaw error
            root_quat = data.qpos[self._free_joint_qpos_ind[3:7]]
            traj_root_quat = traj_data_single.qpos[self._free_joint_qpos_ind[3:7]]
            yaw_diff = quat_to_yaw(root_quat, backend) - quat_to_yaw(traj_root_quat, backend)
            err_root_yaw = backend.abs(backend.arctan2(backend.sin(yaw_diff), backend.cos(yaw_diff)))
        # Joint errors
        if np.any(self._joint_qpos_mask):
            err_joint_pos = backend.sqrt(backend.mean(backend.square(
                qpos_full[self._joint_qpos_mask] - qpos_traj_full[self._joint_qpos_mask])))
        if np.any(self._joint_qvel_mask):
            err_joint_vel = backend.sqrt(backend.mean(backend.square(
                qvel_full[self._joint_qvel_mask] - qvel_traj_full[self._joint_qvel_mask])))
        # Absolute site deviation (like terminal handler)
        if len(self._rel_site_ids) > 1:
            site_mapping = self._rel_site_ids
            cur_sites = data.site_xpos[site_mapping]
            if self._site_mapper.requires_mapping:
                traj_idx = self._site_mapper.model_ids_to_traj_indices(site_mapping)
                ref_sites = traj_data_single.site_xpos[traj_idx]
            else:
                ref_sites = traj_data_single.site_xpos[site_mapping]
            if xy_offset is not None:
                offset_xyz = backend.concatenate([xy_offset, backend.zeros(1, dtype=xy_offset.dtype)])
                ref_sites = ref_sites - offset_xyz
            err_site_abs = backend.mean(backend.linalg.norm(cur_sites - ref_sites, axis=-1))
            # Relative site position error (RMSE of site_rpos)
            err_rpos = backend.sqrt(backend.mean(backend.square(site_rpos - site_rpos_traj)))

        # Build reward_info for logging/diagnostics
        reward_info = {
            "reward_total": total_reward,
            "reward_qpos": qpos_reward,
            "reward_qvel": qvel_reward,
            "reward_root_pos": root_pos_reward,
            "reward_root_quat": root_quat_reward,
            "reward_root_vel": root_vel_reward,
            "penalty_total": total_penalities,
            "penalty_activation_energy": self._activation_energy_coeff * activation_energy_penalty,
            "err_root_xyz": err_root_xyz,
            "err_root_yaw": err_root_yaw,
            "err_joint_pos": err_joint_pos,
            "err_joint_vel": err_joint_vel,
            "err_site_abs": err_site_abs,
            "err_rpos": err_rpos,
        }
        if len(self._rel_site_ids) > 1:
            reward_info["reward_rpos"] = rpos_reward
            reward_info["reward_rquat"] = rangles_reward
            reward_info["reward_rvel_rot"] = rvel_rot_reward
            reward_info["reward_rvel_lin"] = rvel_lin_reward

        return total_reward, carry, reward_info


class TerraMimicReward(MimicReward):
    """TERRA tracking reward with a global root and root-relative body targets."""

    def __init__(self, env: Any, **kwargs):
        kwargs.setdefault("global_root_tracking", True)
        kwargs.setdefault("joint_only_qpos_qvel", True)
        kwargs.setdefault("root_velocity_frame", "global")
        super().__init__(env, **kwargs)

TargetVelocityTrajReward.register()
MimicReward.register()
TerraMimicReward.register()
