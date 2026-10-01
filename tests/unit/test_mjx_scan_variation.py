"""Regression tests for MJX scan carry variation under ``shard_map``."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from musclemimic.core.mujoco_mjx import _preserve_scan_carry_axis_types


@pytest.fixture(scope="module", autouse=True)
def verify_device_simulator() -> None:
    if jax.local_device_count() < 2:
        pytest.skip("The manual-axis regression requires at least two devices.")


def test_opaque_step_output_preserves_scan_carry_variation() -> None:
    devices = jax.local_devices()[:2]
    mesh = jax.make_mesh((2,), ("data",), devices=devices)
    spec = jax.sharding.PartitionSpec("data")

    def shard_step(batch):
        def step_one(value):
            def opaque_step(carry, _):
                # A constant models an FFI output whose data-axis dependency is
                # invisible to JAX's variation analysis.
                next_value = jnp.zeros((), dtype=carry.dtype)
                return _preserve_scan_carry_axis_types(next_value, carry), None

            return jax.lax.scan(opaque_step, value, None, length=1)[0]

        return jax.vmap(step_one)(batch)

    placed = jax.device_put(
        jnp.arange(4, dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, spec),
    )
    compiled = jax.jit(
        jax.shard_map(
            shard_step,
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
            check_vma=True,
        )
    )

    np.testing.assert_array_equal(jax.device_get(compiled(placed)), np.zeros(4))
