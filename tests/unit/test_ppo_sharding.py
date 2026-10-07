"""Tests for PPO data-parallel layouts."""

import jax.numpy as jnp
from flax import struct

from musclemimic.runner.ppo_sharding import DATA, REPLICATED, runtime_specs


@struct.dataclass
class _EnvironmentState:
    per_environment: jnp.ndarray
    flattened_contacts: jnp.ndarray
    scalar: jnp.ndarray


@struct.dataclass
class _RuntimeState:
    train_state: jnp.ndarray
    env_state: _EnvironmentState
    last_obs: jnp.ndarray
    rng: jnp.ndarray
    val_rng: jnp.ndarray


def test_runtime_specs_shard_all_non_scalar_environment_state():
    state = _RuntimeState(
        train_state=jnp.ones((3,)),
        env_state=_EnvironmentState(
            per_environment=jnp.ones((4, 3)),
            flattened_contacts=jnp.ones((28, 2)),
            scalar=jnp.asarray(0),
        ),
        last_obs=jnp.ones((4, 5)),
        rng=jnp.ones((1, 2), dtype=jnp.uint32),
        val_rng=jnp.ones((1, 2), dtype=jnp.uint32),
    )

    specs = runtime_specs(state)

    assert specs.train_state == REPLICATED
    assert specs.env_state.per_environment == DATA
    assert specs.env_state.flattened_contacts == DATA
    assert specs.env_state.scalar == REPLICATED
    assert specs.last_obs == DATA
    assert specs.rng == DATA
    assert specs.val_rng == DATA
