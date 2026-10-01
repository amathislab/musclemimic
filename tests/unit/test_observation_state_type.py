"""Regression tests for stable dynamically generated observation state types."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from loco_mujoco.core.observations.base import ObservationContainer


def test_observation_state_type_is_stable_across_shape_inference_and_shard_map():
    container = ObservationContainer()
    container._stateful_obs = [
        SimpleNamespace(
            name="test_state",
            init_state=lambda _env, _key, _model, data, _backend: data,
        )
    ]

    def initialize(value):
        return container.init_state(None, None, None, value, jnp)

    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    values = jnp.arange(4, dtype=jnp.float32)
    template = jax.eval_shape(initialize, values)
    out_specs = jax.tree.map(lambda _value: P("data"), template)
    compiled_initialize = jax.jit(
        jax.shard_map(
            initialize,
            mesh=mesh,
            in_specs=P("data"),
            out_specs=out_specs,
            check_vma=True,
        )
    )

    state = compiled_initialize(values)

    assert type(state) is type(template)
    np.testing.assert_array_equal(np.asarray(state.test_state), np.asarray(values))
