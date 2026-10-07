"""Relative angular velocity regressions using MuJoCo's sensor as an oracle."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from loco_mujoco.core.utils.math import (
    calculate_relative_site_quantities,
    calculate_relative_velocity_in_local_frame,
)

# Float32 JAX calculations are compared with MuJoCo's float64 sensor output.
VELOCITY_ATOL = 2e-6
MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <site name="unused_root"/>
      <site name="reference" pos="0.1 0.02 0.03" euler="20 -35 15"/>
      <body name="child" pos="0 0 0.3">
        <joint type="hinge" axis="1 2 3"/>
        <geom type="sphere" size="0.05" mass="0.5"/>
        <site name="unused_child"/>
        <site name="target" pos="0.02 -0.03 0.1" euler="-10 40 80"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <frameangvel objtype="site" objname="target"
                 reftype="site" refname="reference"/>
  </sensor>
</mujoco>
"""


@pytest.mark.parametrize("backend", [np, jnp])
def test_rigidly_corotating_sites_have_zero_relative_angular_velocity(backend):
    velocity = backend.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    relative_rotation = backend.asarray(Rotation.from_euler("z", 90, degrees=True).as_matrix())

    result = calculate_relative_velocity_in_local_frame(velocity, velocity, backend.eye(3), relative_rotation, backend)

    np.testing.assert_allclose(result[:, :3], 0.0, atol=VELOCITY_ATOL)


def _sample_kinematics(model, state_index, sparse):
    data = mujoco.MjData(model)
    quaternion_xyzw = Rotation.from_rotvec([0.3 + state_index * 0.2, -0.4, 0.7]).as_quat()
    data.qpos[3:7] = np.roll(quaternion_xyzw, 1)
    data.qpos[7] = 0.2 + state_index * 0.3
    data.qvel[:] = [0.3, -0.2, 0.4, 0.6, -0.5, 0.8, state_index * 0.7]
    mujoco.mj_forward(model, data)
    # The zero-hinge-speed case must also work for differently oriented sites.
    expected = data.sensordata.copy()
    if sparse:
        # Cache order differs from model order and from reference-first order.
        cache_sites = np.array([3, 1])
        arrays = {
            "site_xpos": data.site_xpos[cache_sites].copy(),
            "site_xmat": data.site_xmat[cache_sites].copy(),
            "cvel_parent": data.cvel[model.site_bodyid[cache_sites]].copy(),
            "subtree_com_root": data.subtree_com[1].copy(),
        }
    else:
        arrays = {name: getattr(data, name).copy() for name in ("site_xpos", "site_xmat", "cvel", "subtree_com")}
    return arrays, expected


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("execution", ["numpy", "jax", "jit", "jit_vmap"])
def test_relative_site_angular_velocity_matches_mujoco_sensor(sparse, execution):
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    backend = np if execution == "numpy" else jnp
    site_ids = backend.asarray([1, 3])
    body_ids = backend.asarray(model.site_bodyid[[1, 3]])
    root_ids = backend.asarray(model.body_rootid)
    cache_indices = backend.asarray([1, 0]) if sparse else None

    def angular_velocity(arrays):
        return calculate_relative_site_quantities(
            SimpleNamespace(**arrays),
            site_ids,
            body_ids,
            root_ids,
            backend,
            trajectory_site_indices=cache_indices,
        )[2][0, :3]

    samples = [_sample_kinematics(model, index, sparse) for index in range(3)]
    expected = np.stack([sample[1] for sample in samples])
    if execution == "jit_vmap":
        batch = {name: jnp.asarray(np.stack([sample[0][name] for sample in samples])) for name in samples[0][0]}
        result = jax.jit(jax.vmap(angular_velocity))(batch)
    else:
        evaluate = jax.jit(angular_velocity) if execution == "jit" else angular_velocity
        result = np.stack(
            [
                np.asarray(evaluate({name: backend.asarray(value) for name, value in arrays.items()}))
                for arrays, _ in samples
            ]
        )

    np.testing.assert_allclose(result, expected, atol=VELOCITY_ATOL, rtol=VELOCITY_ATOL)
    # A moving reference comparison must retain the oracle's nonzero error.
    np.testing.assert_allclose(
        np.mean((np.asarray(result)[1] - np.asarray(result)[2]) ** 2),
        np.mean((expected[1] - expected[2]) ** 2),
        atol=VELOCITY_ATOL,
        rtol=VELOCITY_ATOL,
    )


def test_relative_angular_velocity_is_invariant_to_world_rotation():
    rotation = Rotation.from_rotvec([0.3, -0.7, 0.5]).as_matrix()
    global_rotation = Rotation.from_rotvec([-0.8, 0.2, 0.4]).as_matrix()
    relative_rotation = Rotation.from_rotvec([0.6, -0.2, 0.1]).as_matrix()
    velocity_a = np.array([0.2, 0.4, -0.6, 0.1, 0.3, 0.5])
    velocity_b = np.array([-0.4, 0.3, 0.7, 0.2, -0.5, 0.6])
    original = calculate_relative_velocity_in_local_frame(velocity_a, velocity_b, rotation, relative_rotation, np)
    rotated = calculate_relative_velocity_in_local_frame(
        (global_rotation @ velocity_a.reshape(2, 3).T).T.reshape(6),
        (global_rotation @ velocity_b.reshape(2, 3).T).T.reshape(6),
        global_rotation @ rotation,
        relative_rotation,
        np,
    )

    np.testing.assert_allclose(original[:, :3], rotated[:, :3], atol=VELOCITY_ATOL)


def test_angular_fix_preserves_existing_linear_feature():
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    result = calculate_relative_velocity_in_local_frame(
        np.array([0, 0, 0, 1, 2, 3]),
        np.array([0, 0, 0, -2, 1, 1]),
        rotation,
        np.eye(3),
        np,
    )

    # Historical linear feature is R_A (v_A - v_B), retained in this fix.
    np.testing.assert_allclose(result[:, 3:], [[-1, 3, 2]], atol=VELOCITY_ATOL)
