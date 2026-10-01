"""Cross-device reduction for validation summaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.dataclasses import (
    MetricHandlerTransition,
    ValidationCarry,
    ValidationData,
    ValidationDataFields,
)
from musclemimic.algorithms.common.distributed import mark_data_varying
from musclemimic.algorithms.common.env_state_utils import unwrap_to_mjx
from musclemimic.core.wrappers import LogEnvState


class ValidationRolloutResult(NamedTuple):
    """Result of a single validation rollout function."""

    metrics: Any
    rollout: Any
    rng: jax.Array


def run_validation_rollout(
    policy_action: Callable[[jax.Array, jax.Array], jax.Array],
    rng: jax.Array,
    env: Any,
    config: Any,
    metrics_handler: Any,
    *,
    reset_weights: Any | None = None,
    axis_name: str | None = None,
    preserve_next_env_state: Callable[[Any, Any, str | None], Any] | None = None,
) -> ValidationRolloutResult:
    """Run one vectorized validation rollout for a fixed policy action function."""

    def preserve_state(next_state: Any, previous_state: Any) -> Any:
        if preserve_next_env_state is not None:
            return preserve_next_env_state(next_state, previous_state, axis_name)
        return mark_data_varying(next_state, axis_name, empty_reference=previous_state)

    def step(carry, _unused):
        env_state, observation, step_rng = carry
        step_rng, action_rng = jax.random.split(step_rng)
        action = policy_action(observation, action_rng)
        (
            next_observation,
            _reward,
            _absorbing,
            _done,
            _info,
            next_env_state,
            transition_state,
            _transition_observation,
        ) = env.step_with_transition(env_state, action)
        next_env_state = preserve_state(next_env_state, env_state)
        log_state = transition_state.find(LogEnvState)
        mjx_state, _ = unwrap_to_mjx(log_state.env_state)
        data = mjx_state.data
        site_ids = metrics_handler.rel_site_ids
        validation_data = ValidationData(
            metrics=log_state.metrics,
            data=ValidationDataFields(
                qpos=data.qpos,
                qvel=data.qvel,
                xpos=data.xpos,
                xquat=data.xquat,
                cvel=data.cvel,
                subtree_com=data.subtree_com,
                site_xpos=data.site_xpos[:, site_ids, :],
                site_xmat=data.site_xmat[:, site_ids, :, :],
            ),
            additional_carry=ValidationCarry(
                traj_state=mjx_state.additional_carry.traj_state,
            ),
        )
        transition = MetricHandlerTransition(val_data=validation_data)
        return (next_env_state, next_observation, step_rng), transition

    next_rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_rng, int(config.validation.num_envs))
    if reset_weights is None:
        observation, env_state = env.reset(reset_keys)
    else:
        from musclemimic.algorithms.common.trajectory_sampling import reset_from_trajectory_weights

        observation, env_state = reset_from_trajectory_weights(
            env,
            reset_keys,
            reset_weights,
        )

    env_state = mark_data_varying(env_state, axis_name)
    _, rollout = jax.lax.scan(
        step,
        (env_state, observation, rollout_rng),
        None,
        int(config.validation.num_steps),
    )
    num_sites = metrics_handler.rel_site_ids.shape[0]
    local_site_ids = jnp.arange(num_sites, dtype=metrics_handler.rel_site_ids.dtype)
    metrics = metrics_handler(
        rollout.val_data,
        sim_site_idx=local_site_ids,
    )
    return ValidationRolloutResult(metrics=metrics, rollout=rollout, rng=next_rng)


def reduce_validation_summary(summary: Any, *, axis_name: str) -> Any:
    """Combine shard-local validation summaries on every data replica."""
    rmse_fields = {
        "err_root_xyz",
        "err_joint_pos",
        "err_joint_vel",
        "err_rpos",
    }

    def sum_across_devices(value):
        return jax.lax.psum(value, axis_name=axis_name)

    def reduce(path, value):
        name = path[-1].name
        if name.endswith("_count"):
            return sum_across_devices(value)
        if name == "max_timestep":
            return jax.lax.pmax(value, axis_name=axis_name)
        if name in rmse_fields:
            return jnp.sqrt(jax.lax.pmean(jnp.square(value), axis_name=axis_name))
        return jax.lax.pmean(value, axis_name=axis_name)

    reduced = jax.tree_util.tree_map_with_path(reduce, summary)
    local_count = summary.episode_count
    global_count = jnp.maximum(reduced.episode_count, 1.0)
    return reduced.replace(
        mean_episode_return=sum_across_devices(summary.mean_episode_return * local_count) / global_count,
        mean_episode_length=sum_across_devices(summary.mean_episode_length * local_count) / global_count,
        early_termination_rate=reduced.early_termination_count / global_count,
        horizon_timeout_rate=reduced.horizon_timeout_count / global_count,
        reward_termination_rate=reduced.reward_termination_count / global_count,
    )
