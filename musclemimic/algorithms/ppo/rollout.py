"""PPO environment rollout collection."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.dataclasses import Transition
from musclemimic.algorithms.common.env_state_utils import unwrap_to_mjx
from musclemimic.core.wrappers import LogEnvState
from musclemimic.utils.debug_tools import DebugFlags, maybe_track_nacon


def _sample_trajectory_starts(
    env: Any,
    trajectory_keys: jax.Array,
    step_keys: jax.Array,
    weights: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample trajectory ids and start steps for a vectorized reset."""
    return jax.vmap(
        lambda trajectory_key, step_key: env.th.sample_start_indices(
            trajectory_key,
            step_key,
            weights=weights,
            backend=jnp,
        )
    )(trajectory_keys, step_keys)


def _split_reset_keys(reset_keys: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create independent environment, trajectory, and step keys per environment."""
    reset_key_pairs = jax.vmap(jax.random.split)(reset_keys)
    environment_keys = reset_key_pairs[:, 0]
    sampling_keys = reset_key_pairs[:, 1]

    sampling_key_pairs = jax.vmap(jax.random.split)(sampling_keys)
    trajectory_keys = sampling_key_pairs[:, 0]
    step_keys = sampling_key_pairs[:, 1]
    return environment_keys, trajectory_keys, step_keys


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
    trajectory_ids, start_steps, _start_modes = _sample_trajectory_starts(
        env,
        trajectory_keys,
        step_keys,
        weights,
    )
    observation, env_state = env.reset_to_step(environment_keys, trajectory_ids, start_steps)

    mjx_state, rewrap = unwrap_to_mjx(env_state)
    num_envs = reset_keys.shape[0]
    sampling_weights = jnp.broadcast_to(weights, (num_envs, weights.shape[0]))
    carry = mjx_state.additional_carry.replace(sampling_weights=sampling_weights)
    mjx_state = mjx_state.replace(additional_carry=carry)
    return observation, rewrap(mjx_state)


def collect_rollout(
    train_state: Any,
    env_state: Any,
    observation: jax.Array,
    rng: jax.Array,
    *,
    network: Any,
    env: Any,
    num_steps: int,
    debug_flags: DebugFlags,
) -> tuple[tuple[Any, Any, jax.Array, jax.Array], Transition]:
    """Collect one fixed-length PPO rollout with ``jax.lax.scan``."""
    trajectory_split_points = jnp.asarray(env.th.traj.data.split_points)

    def _step(carry, _unused):
        train_state, env_state, observation, rng = carry

        rng, action_rng = jax.random.split(rng)
        policy, value = network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            observation,
        )
        action = policy.sample(seed=action_rng)
        log_probability = policy.log_prob(action)

        (
            next_observation,
            reward,
            absorbing,
            done,
            info,
            next_env_state,
            transition_state,
            _transition_observation,
        ) = env.step_with_transition(env_state, action)
        log_env_state = transition_state.find(LogEnvState)
        trajectory_state = transition_state.additional_carry.traj_state
        trajectory_length = (
            trajectory_split_points[trajectory_state.traj_no + 1] - trajectory_split_points[trajectory_state.traj_no]
        )
        trajectory_end = trajectory_state.subtraj_step_no >= trajectory_length - 1

        if debug_flags.track_nacon:
            mjx_state, _ = unwrap_to_mjx(log_env_state.env_state)
            maybe_track_nacon(mjx_state.data._impl, debug_flags)

        transition = Transition(
            done,
            absorbing,
            trajectory_end,
            action,
            value,
            reward,
            log_probability,
            observation,
            info,
            trajectory_state,
            log_env_state.metrics,
        )
        carry = (train_state, next_env_state, next_observation, rng)
        return carry, transition

    initial_carry = (train_state, env_state, observation, rng)
    return jax.lax.scan(_step, initial_carry, None, num_steps)
