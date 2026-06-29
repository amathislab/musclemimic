"""
RolloutBuffer module for on-policy RL algorithms.

Provides:
- GAE (Generalized Advantage Estimation) computation
- Minibatch generation for PPO-style updates
"""

from __future__ import annotations

from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
from flax import struct

# Type variable for trajectory batch (any pytree with Transition-like structure)
T = TypeVar("T")


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    absorbing: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
    unroll: int = 16,
    bootstrap_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Uses shifted rollout values for V(s_{t+1}) and distinguishes between:
    - bootstrap_mask: whether V(s_{t+1}) should be used in delta
    - done: episode boundary (no GAE propagation)

    Args:
        rewards: (num_steps, num_envs)
        values: (num_steps, num_envs)
        dones: (num_steps, num_envs)
        absorbing: (num_steps, num_envs)
        last_value: (num_envs,)
        gamma: Discount factor
        gae_lambda: GAE decay parameter
        unroll: Scan unroll factor for performance (default 16, matching PPO)
        bootstrap_mask: optional (num_steps, num_envs). Defaults to 1 - absorbing.

    Returns:
        advantages: (num_steps, num_envs)
        returns: (num_steps, num_envs)
    """
    num_steps = rewards.shape[0]
    if bootstrap_mask is None:
        bootstrap_mask = jnp.logical_not(absorbing.astype(bool))

    def _step(carry: tuple[jnp.ndarray, jnp.ndarray], t: int):
        gae, next_val = carry
        bootstrap = bootstrap_mask[t].astype(values.dtype)
        nonterminal = 1.0 - dones[t].astype(values.dtype)
        delta = rewards[t] + gamma * next_val * bootstrap - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        return (gae, values[t]), gae

    _, advs = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_value), last_value),
        jnp.arange(num_steps - 1, -1, -1),
        unroll=unroll,
    )

    advantages = advs[::-1]
    returns = advantages + values
    return advantages, returns


def create_minibatches(
    traj_batch: T,
    advantages: jnp.ndarray,
    targets: jnp.ndarray,
    num_minibatches: int,
    rng: jax.Array,
) -> tuple[T, jnp.ndarray, jnp.ndarray]:
    """
    Shuffle and split trajectory data into minibatches.

    Args:
        traj_batch: Pytree with leaves of shape (num_steps, num_envs, ...)
        advantages: (num_steps, num_envs)
        targets: (num_steps, num_envs)
        num_minibatches: Number of minibatches
        rng: Random key

    Returns:
        batched_traj: Leaves reshaped to (num_minibatches, minibatch_size, ...)
        batched_advantages: (num_minibatches, minibatch_size)
        batched_targets: (num_minibatches, minibatch_size)
    """
    num_steps, num_envs = advantages.shape
    batch_size = num_steps * num_envs
    if batch_size % num_minibatches != 0:
        raise ValueError(f"Batch of size {batch_size} not divisible by num_minibatches={num_minibatches}.")
    mb_size = batch_size // num_minibatches

    # Flatten: (num_steps, num_envs, ...) -> (batch_size, ...)
    def flatten(x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(batch_size) if x.ndim == 2 else x.reshape(batch_size, *x.shape[2:])

    flat_batch = jax.tree.map(flatten, traj_batch)
    flat_adv = advantages.reshape(batch_size)
    flat_tgt = targets.reshape(batch_size)

    # Shuffle
    perm = jax.random.permutation(rng, batch_size)
    shuf_batch = jax.tree.map(lambda x: x[perm], flat_batch)
    shuf_adv = flat_adv[perm]
    shuf_tgt = flat_tgt[perm]

    # Split: (batch_size, ...) -> (num_minibatches, mb_size, ...)
    def split(x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(num_minibatches, mb_size) if x.ndim == 1 else x.reshape(num_minibatches, mb_size, *x.shape[1:])

    return jax.tree.map(split, shuf_batch), split(shuf_adv), split(shuf_tgt)


@struct.dataclass
class RolloutBuffer(Generic[T]):
    """
    Container for trajectory data with GAE and minibatch utilities.

    Attributes:
        traj_batch: Collected transitions, leaves have shape (num_steps, num_envs, ...)
        num_steps: Rollout length
        num_envs: Number of parallel environments
    """

    traj_batch: T
    num_steps: int = struct.field(pytree_node=False)
    num_envs: int = struct.field(pytree_node=False)

    @classmethod
    def from_scan(cls, traj_batch: T) -> RolloutBuffer[T]:
        """Create buffer from jax.lax.scan output."""
        try:
            reward = traj_batch.reward
            num_steps, num_envs = reward.shape[:2]
        except Exception:
            first_leaf = jax.tree.leaves(traj_batch)[0]
            num_steps, num_envs = first_leaf.shape[0], first_leaf.shape[1]
        return cls(traj_batch, num_steps=num_steps, num_envs=num_envs)

    @classmethod
    def allocate(cls, num_steps: int, num_envs: int, transition_template: T) -> RolloutBuffer[T]:
        """
        Allocate an empty buffer with the same per-env shapes/dtypes as transition_template.

        transition_template leaves must have leading shape (num_envs, ...).
        """

        def _zeros_like_leaf(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.zeros((num_steps, num_envs, *x.shape[1:]), dtype=x.dtype)

        traj_batch = jax.tree.map(_zeros_like_leaf, transition_template)
        return cls(traj_batch, num_steps=num_steps, num_envs=num_envs)

    def compute_advantages(
        self,
        last_value: jnp.ndarray,
        gamma: float,
        gae_lambda: float,
        unroll: int = 16,
        bootstrap_mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute GAE advantages and return targets."""
        return compute_gae(
            self.traj_batch.reward,
            self.traj_batch.value,
            self.traj_batch.done,
            self.traj_batch.absorbing,
            last_value,
            gamma,
            gae_lambda,
            unroll=unroll,
            bootstrap_mask=bootstrap_mask,
        )

    def get_minibatches(
        self,
        advantages: jnp.ndarray,
        targets: jnp.ndarray,
        num_minibatches: int,
        rng: jax.Array,
    ) -> tuple[T, jnp.ndarray, jnp.ndarray]:
        """Get shuffled minibatches for PPO update."""
        return create_minibatches(self.traj_batch, advantages, targets, num_minibatches, rng)

    def add_transition(self, transition: T, step: int) -> RolloutBuffer[T]:
        """
        Write a single timestep (batched over envs) into the buffer at index `step`.
        """
        if step < 0 or step >= self.num_steps:
            raise IndexError(f"step {step} out of range for buffer length {self.num_steps}")

        def _write(buf_leaf: jnp.ndarray, trans_leaf: jnp.ndarray) -> jnp.ndarray:
            return buf_leaf.at[step].set(trans_leaf)

        updated = jax.tree.map(_write, self.traj_batch, transition)
        return self.replace(traj_batch=updated)

    @property
    def batch_size(self) -> int:
        return self.num_steps * self.num_envs
