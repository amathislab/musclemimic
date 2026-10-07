"""PPO environment rollout collection."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.dataclasses import Transition
from musclemimic.algorithms.common.env_state_utils import (
    preserve_data_axis_variation,
    unwrap_to_mjx,
)
from musclemimic.algorithms.common.trajectory_sampling import (
    reset_from_trajectory_weights,
)
from musclemimic.core.wrappers import LogEnvState
from musclemimic.utils.debug_tools import DebugFlags, maybe_track_nacon


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
    axis_name: str | None = None,
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
        next_env_state = preserve_data_axis_variation(
            next_env_state,
            env_state,
            axis_name,
        )
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
