"""Device-resident replay storage for FlashSAC."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct


class ReplayBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminated: jax.Array


@struct.dataclass
class ReplayBuffer:
    """Fixed-size circular replay buffer stored as JAX arrays."""

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_observations: jax.Array
    terminated: jax.Array
    position: jax.Array
    size: jax.Array

    @property
    def capacity(self) -> int:
        return self.observations.shape[0]

    def add(
        self,
        observations: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        next_observations: jax.Array,
        terminated: jax.Array,
    ) -> ReplayBuffer:
        """Append one vectorized environment interaction."""
        batch_size = observations.shape[0]
        indices = (self.position + jnp.arange(batch_size, dtype=jnp.int32)) % self.capacity
        return self.replace(
            observations=self.observations.at[indices].set(observations.astype(self.observations.dtype)),
            actions=self.actions.at[indices].set(actions.astype(self.actions.dtype)),
            rewards=self.rewards.at[indices].set(rewards),
            next_observations=self.next_observations.at[indices].set(
                next_observations.astype(self.next_observations.dtype)
            ),
            terminated=self.terminated.at[indices].set(terminated),
            position=(self.position + batch_size) % self.capacity,
            size=jnp.minimum(self.size + batch_size, self.capacity),
        )

    def sample(self, rng: jax.Array, batch_size: int) -> ReplayBatch:
        """Uniformly sample valid transitions as float32 learning inputs."""
        indices = jax.random.randint(rng, (batch_size,), minval=0, maxval=self.size)
        return ReplayBatch(
            observations=self.observations[indices].astype(jnp.float32),
            actions=self.actions[indices].astype(jnp.float32),
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices].astype(jnp.float32),
            terminated=self.terminated[indices],
        )


def create_replay_buffer(
    capacity: int,
    observation_dim: int,
    action_dim: int,
    storage_dtype: str,
) -> ReplayBuffer:
    """Allocate an empty replay buffer."""
    dtypes = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}
    try:
        dtype = dtypes[storage_dtype]
    except KeyError as exc:
        expected = ", ".join(dtypes)
        raise ValueError(f"replay_storage_dtype must be one of: {expected}") from exc

    return ReplayBuffer(
        observations=jnp.zeros((capacity, observation_dim), dtype=dtype),
        actions=jnp.zeros((capacity, action_dim), dtype=dtype),
        rewards=jnp.zeros((capacity,), dtype=jnp.float32),
        next_observations=jnp.zeros((capacity, observation_dim), dtype=dtype),
        terminated=jnp.zeros((capacity,), dtype=jnp.bool_),
        position=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )
