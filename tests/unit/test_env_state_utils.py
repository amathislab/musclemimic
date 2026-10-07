"""Tests for environment-state handling at the data-parallel boundary."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import pytest
from flax import struct

from musclemimic.algorithms.common import env_state_utils
from musclemimic.algorithms.common.distributed import (
    DATA_AXIS_NAME,
    make_data_mesh,
    mark_data_varying,
)


@struct.dataclass
class _InnerState:
    data: jax.Array
    info: dict[str, jax.Array]


@struct.dataclass
class _WrapperState:
    env_state: _InnerState
    absorbing: jax.Array
    metrics: dict[str, jax.Array]


def test_preserve_data_axis_variation_covers_wrapper_state():
    if jax.local_device_count() < 4:
        pytest.skip("requires four visible or simulated JAX devices")

    mesh = make_data_mesh(4)
    data = jax.sharding.PartitionSpec(DATA_AXIS_NAME)

    def run_scan(local_values):
        initial = _WrapperState(
            env_state=_InnerState(
                data=local_values,
                info={"value": local_values},
            ),
            absorbing=local_values >= 0,
            metrics={"absorbing": local_values >= 0},
        )
        initial = mark_data_varying(initial, DATA_AXIS_NAME)

        def step(previous, _unused):
            size = previous.absorbing.shape[0]
            next_state = _WrapperState(
                env_state=_InnerState(
                    data=jnp.zeros(size),
                    info={"value": jnp.zeros(size)},
                ),
                absorbing=jnp.zeros(size, dtype=bool),
                metrics={"absorbing": jnp.zeros(size, dtype=bool)},
            )
            return env_state_utils.preserve_data_axis_variation(
                next_state,
                previous,
                DATA_AXIS_NAME,
            ), None

        final, _ = jax.lax.scan(step, initial, None, length=2)
        return final.absorbing, final.metrics["absorbing"]

    compiled = jax.jit(
        jax.shard_map(
            run_scan,
            mesh=mesh,
            in_specs=(data,),
            out_specs=(data, data),
        )
    )

    absorbing, metric_absorbing = compiled(jnp.arange(8, dtype=jnp.float32))

    assert not jnp.any(absorbing)
    assert not jnp.any(metric_absorbing)
