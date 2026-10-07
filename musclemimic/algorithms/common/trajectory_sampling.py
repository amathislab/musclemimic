"""Trajectory-aware environment reset helpers shared by JAX algorithms."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .env_state_utils import unwrap_to_mjx


def _sample_trajectory_starts(
    env: Any,
    trajectory_keys: jax.Array,
    step_keys: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(
        lambda trajectory_key, step_key: env.th.sample_start_indices(
            trajectory_key,
            step_key,
            weights=weights,
            backend=jnp,
        )
    )(trajectory_keys, step_keys)


def _split_reset_keys(reset_keys: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    reset_key_pairs = jax.vmap(jax.random.split)(reset_keys)
    environment_keys = reset_key_pairs[:, 0]
    sampling_keys = reset_key_pairs[:, 1]
    sampling_key_pairs = jax.vmap(jax.random.split)(sampling_keys)
    return environment_keys, sampling_key_pairs[:, 0], sampling_key_pairs[:, 1]


def reset_from_trajectory_weights(
    env: Any,
    reset_keys: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, Any]:
    """Reset from a weighted motion distribution and retain it for autoreset."""
    weights = jnp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError(f"trajectory reset weights must be one-dimensional, got shape {weights.shape}")

    environment_keys, trajectory_keys, step_keys = _split_reset_keys(reset_keys)
    trajectory_ids, start_steps = _sample_trajectory_starts(
        env,
        trajectory_keys,
        step_keys,
        weights,
    )
    observation, env_state = env.reset_to_step(environment_keys, trajectory_ids, start_steps)

    mjx_state, rewrap = unwrap_to_mjx(env_state)
    sampling_weights = jnp.broadcast_to(
        weights,
        (reset_keys.shape[0], weights.shape[0]),
    )
    carry = mjx_state.additional_carry.replace(sampling_weights=sampling_weights)
    return observation, rewrap(mjx_state.replace(additional_carry=carry))
