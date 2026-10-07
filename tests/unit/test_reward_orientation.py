"""Tests for the SO(3) distances used by imitation rewards."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from musclemimic.core.reward.orientation import (
    rotation_vector_geodesic_distance,
    scalar_first_quaternion_geodesic_distance,
)


@pytest.mark.parametrize("backend", [np, jnp])
def test_rotation_vector_distance_is_continuous_across_pi(backend):
    epsilon = 1e-3
    rotation_a = backend.asarray([[0.0, 0.0, np.pi - epsilon]])
    rotation_b = backend.asarray([[0.0, 0.0, -np.pi + epsilon]])

    distance = rotation_vector_geodesic_distance(rotation_a, rotation_b, backend)

    np.testing.assert_allclose(distance, [2.0 * epsilon], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("backend", [np, jnp])
def test_quaternion_distance_is_invariant_to_sign(backend):
    quaternion = backend.asarray([0.5, 0.5, -0.5, 0.5])

    distance = scalar_first_quaternion_geodesic_distance(quaternion, -quaternion, backend)

    np.testing.assert_allclose(distance, 0.0, atol=1e-6)


def test_orientation_distances_support_jit():
    rotvec_distance = jax.jit(lambda a, b: rotation_vector_geodesic_distance(a, b, jnp))
    quaternion_distance = jax.jit(lambda a, b: scalar_first_quaternion_geodesic_distance(a, b, jnp))

    angle = np.pi / 3.0
    np.testing.assert_allclose(
        rotvec_distance(jnp.zeros(3), jnp.array([angle, 0.0, 0.0])),
        angle,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        quaternion_distance(
            jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        ),
        angle,
        rtol=1e-6,
    )
