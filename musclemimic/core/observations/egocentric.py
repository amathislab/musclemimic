"""Egocentric observation helpers shared by state and goal observations."""

from types import ModuleType
from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np
from jax.scipy.spatial.transform import Rotation as jnp_R
from scipy.spatial.transform import Rotation as np_R

from loco_mujoco.core.observations.base import StatefulObservation
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from loco_mujoco.core.utils.mujoco import mj_jntname2qposid, mj_jntname2qvelid


def rotation_matrix_from_quaternion(quaternion: Any, backend: ModuleType) -> Any:
    """Return the local-to-world rotation matrix for a wxyz quaternion."""
    rotation = np_R if backend is np else jnp_R
    return rotation.from_quat(quat_scalarfirst2scalarlast(quaternion)).as_matrix()


def heading_rotation_matrix(root_rotation: Any, backend: ModuleType) -> Any:
    """Return the closest yaw-only rotation to a root rotation matrix."""
    yaw = backend.arctan2(
        root_rotation[1, 0] - root_rotation[0, 1],
        root_rotation[0, 0] + root_rotation[1, 1],
    )
    cos_yaw = backend.cos(yaw)
    sin_yaw = backend.sin(yaw)
    zero = backend.zeros_like(yaw)
    one = backend.ones_like(yaw)
    return backend.stack(
        [
            backend.stack([cos_yaw, -sin_yaw, zero]),
            backend.stack([sin_yaw, cos_yaw, zero]),
            backend.stack([zero, zero, one]),
        ]
    )


def world_to_heading(vectors: Any, heading_rotation: Any, backend: ModuleType) -> Any:
    """Express world-frame vectors in a heading-aligned frame."""
    return backend.einsum("ij,...j->...i", heading_rotation.T, vectors)


def root_velocity_in_heading_frame(
    qvel: Any,
    root_rotation: Any,
    heading_rotation: Any,
    backend: ModuleType,
) -> Any:
    """Express MuJoCo free-joint velocity in a common heading frame.

    MuJoCo stores free-joint linear velocity in world coordinates and angular
    velocity in root-local coordinates. The output keeps the qvel ordering
    ``[linear, angular]``.
    """
    linear_velocity = world_to_heading(qvel[:3], heading_rotation, backend)
    angular_velocity_world = root_rotation @ qvel[3:6]
    angular_velocity = world_to_heading(angular_velocity_world, heading_rotation, backend)
    return backend.concatenate([linear_velocity, angular_velocity])


def rotation_error_in_frame(
    current_rotation: Any,
    target_rotation: Any,
    frame_rotation: Any,
    backend: ModuleType,
) -> Any:
    """Express the world-frame current-to-target rotation in a chosen frame."""
    rotation = np_R if backend is np else jnp_R
    world_error = backend.einsum("...ij,...kj->...ik", target_rotation, current_rotation)
    frame_error = backend.einsum("ji,...jk,kl->...il", frame_rotation, world_error, frame_rotation)
    return rotation.from_matrix(frame_error).as_rotvec()


class HeadingFrameFreeJointVelocity(StatefulObservation):
    """Free-joint velocity expressed in the joint's yaw-aligned frame."""

    dim = 6

    def __init__(self, obs_name: str, xml_name: str, **kwargs: Any) -> None:
        self.xml_name = xml_name
        self._qpos_indices = None
        self._qvel_indices = None
        super().__init__(obs_name, **kwargs)

    def _init_from_mj(
        self,
        env: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        current_obs_size: int,
    ) -> None:
        del env, data
        self._qpos_indices = np.asarray(mj_jntname2qposid(self.xml_name, model))
        self._qvel_indices = np.asarray(mj_jntname2qvelid(self.xml_name, model))
        self.min = [-np.inf] * self.dim
        self.max = [np.inf] * self.dim
        self.obs_ind = np.arange(current_obs_size, current_obs_size + self.dim)
        self._initialized_from_mj = True

    def get_obs_and_update_state(
        self,
        env: Any,
        model: Any,
        data: Any,
        carry: Any,
        backend: ModuleType,
    ) -> tuple[Any, Any]:
        del env, model
        root_quaternion = data.qpos[self._qpos_indices[3:7]]
        root_rotation = rotation_matrix_from_quaternion(root_quaternion, backend)
        heading_rotation = heading_rotation_matrix(root_rotation, backend)
        root_qvel = data.qvel[self._qvel_indices]
        observation = root_velocity_in_heading_frame(
            root_qvel,
            root_rotation,
            heading_rotation,
            backend,
        )
        return observation, carry
