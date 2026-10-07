"""Backend-independent orientation distances used by imitation rewards."""

from types import ModuleType

import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as JaxRotation
from scipy.spatial.transform import Rotation as NumpyRotation

from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast, quaternion_angular_distance


def rotation_vector_geodesic_distance(
    rotation_a: np.ndarray | jnp.ndarray,
    rotation_b: np.ndarray | jnp.ndarray,
    backend: ModuleType,
) -> np.ndarray | jnp.ndarray:
    """Return the shortest SO(3) angle between axis-angle rotation vectors.

    Euclidean subtraction of rotation vectors becomes discontinuous at the
    positive and negative pi branch cut. Relative rotation magnitude gives the
    shortest angular distance in radians.
    """
    rotation_type = NumpyRotation if backend is np else JaxRotation
    relative_rotation = rotation_type.from_rotvec(rotation_a).inv() * rotation_type.from_rotvec(rotation_b)
    return relative_rotation.magnitude()


def scalar_first_quaternion_geodesic_distance(
    quaternion_a: np.ndarray | jnp.ndarray,
    quaternion_b: np.ndarray | jnp.ndarray,
    backend: ModuleType,
) -> np.ndarray | jnp.ndarray:
    """Return the shortest SO(3) angle between MuJoCo wxyz quaternions."""
    return quaternion_angular_distance(
        quat_scalarfirst2scalarlast(quaternion_a),
        quat_scalarfirst2scalarlast(quaternion_b),
        backend,
    )
