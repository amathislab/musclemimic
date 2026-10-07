"""Environment state traversal utilities for wrapped MJX environments."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from musclemimic.algorithms.common.distributed import mark_data_varying
from musclemimic.core.mujoco_mjx import MjxState


def unwrap_to_mjx(state):
    """Unwrap state chain to find MjxState, returning (mjx_state, rebuild_fn).

    Handles NStepWrapperState which may be nested between LogEnvState and MjxState.
    Mirrors AutoResetWrapper._unwrap_state pattern.

    Args:
        state: Wrapper state (e.g., NormalizeVecRewEnvState, LogEnvState, NStepWrapperState)

    Returns:
        tuple: (mjx_state, rewrap_fn) where rewrap_fn rebuilds the wrapper chain
    """
    chain = []
    inner = state
    while hasattr(inner, "env_state") and not isinstance(inner, MjxState):
        chain.append(inner)
        inner = inner.env_state

    def rewrap(new_mjx):
        cur = new_mjx
        for wrapper in reversed(chain):
            cur = wrapper.replace(env_state=cur)
        return cur

    return inner, rewrap


def preserve_data_axis_variation(
    next_state: Any,
    previous_state: Any,
    axis_name: str | None,
) -> Any:
    """Keep wrapped environment scan carry variation stable after a step."""
    return mark_data_varying(
        next_state,
        axis_name,
        empty_reference=previous_state,
    )


def update_carry_weights_normalized(env_state, new_weights):
    """Update sampling_weights when normalize_env=True.

    State structure: NormalizeVecRewEnvState.env_state -> LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: NormalizeVecRewEnvState
        new_weights: New sampling weights array, shape (num_envs, n_trajectories)

    Returns:
        Updated env_state with new sampling weights
    """
    log_state = env_state.env_state  # LogEnvState
    mjx_state, rebuild = unwrap_to_mjx(log_state.env_state)
    new_carry = mjx_state.additional_carry.replace(sampling_weights=new_weights)
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    new_log = log_state.replace(env_state=new_inner)
    return env_state.replace(env_state=new_log)


def update_carry_weights_unnormalized(env_state, new_weights):
    """Update sampling_weights when normalize_env=False.

    State structure: LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: LogEnvState
        new_weights: New sampling weights array, shape (num_envs, n_trajectories)

    Returns:
        Updated env_state with new sampling weights
    """
    mjx_state, rebuild = unwrap_to_mjx(env_state.env_state)
    new_carry = mjx_state.additional_carry.replace(sampling_weights=new_weights)
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    return env_state.replace(env_state=new_inner)


def update_carry_threshold_normalized(env_state, new_threshold):
    """Update termination_threshold when normalize_env=True.

    State structure: NormalizeVecRewEnvState.env_state -> LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: NormalizeVecRewEnvState
        new_threshold: New termination threshold value

    Returns:
        Updated env_state with new termination threshold
    """
    log_state = env_state.env_state  # LogEnvState
    mjx_state, rebuild = unwrap_to_mjx(log_state.env_state)
    current = mjx_state.additional_carry.termination_threshold
    threshold = jnp.broadcast_to(jnp.asarray(new_threshold, dtype=current.dtype), current.shape)
    new_carry = mjx_state.additional_carry.replace(termination_threshold=threshold)
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    new_log = log_state.replace(env_state=new_inner)
    return env_state.replace(env_state=new_log)


def update_carry_threshold_unnormalized(env_state, new_threshold):
    """Update termination_threshold when normalize_env=False.

    State structure: LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: LogEnvState
        new_threshold: New termination threshold value

    Returns:
        Updated env_state with new termination threshold
    """
    mjx_state, rebuild = unwrap_to_mjx(env_state.env_state)
    current = mjx_state.additional_carry.termination_threshold
    threshold = jnp.broadcast_to(jnp.asarray(new_threshold, dtype=current.dtype), current.shape)
    new_carry = mjx_state.additional_carry.replace(termination_threshold=threshold)
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    return env_state.replace(env_state=new_inner)


def update_carry_ema_normalized(env_state, new_ema_done, new_ema_early):
    """Update EMA counts for adaptive sampling when normalize_env=True.

    State structure: NormalizeVecRewEnvState.env_state -> LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: NormalizeVecRewEnvState
        new_ema_done: New EMA done counts array, shape (num_envs, n_trajectories)
        new_ema_early: New EMA early termination counts array, shape (num_envs, n_trajectories)

    Returns:
        Updated env_state with new EMA counts
    """
    log_state = env_state.env_state  # LogEnvState
    mjx_state, rebuild = unwrap_to_mjx(log_state.env_state)
    new_carry = mjx_state.additional_carry.replace(
        ema_done_counts=new_ema_done,
        ema_early_counts=new_ema_early,
    )
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    new_log = log_state.replace(env_state=new_inner)
    return env_state.replace(env_state=new_log)


def update_carry_ema_unnormalized(env_state, new_ema_done, new_ema_early):
    """Update EMA counts for adaptive sampling when normalize_env=False.

    State structure: LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: LogEnvState
        new_ema_done: New EMA done counts array, shape (num_envs, n_trajectories)
        new_ema_early: New EMA early termination counts array, shape (num_envs, n_trajectories)

    Returns:
        Updated env_state with new EMA counts
    """
    mjx_state, rebuild = unwrap_to_mjx(env_state.env_state)
    new_carry = mjx_state.additional_carry.replace(
        ema_done_counts=new_ema_done,
        ema_early_counts=new_ema_early,
    )
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    return env_state.replace(env_state=new_inner)


def get_carry_normalized(env_state):
    """Get additional_carry when normalize_env=True.

    State structure: NormalizeVecRewEnvState.env_state -> LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: NormalizeVecRewEnvState

    Returns:
        additional_carry from the MjxState
    """
    log_state = env_state.env_state  # LogEnvState
    mjx_state, _ = unwrap_to_mjx(log_state.env_state)
    return mjx_state.additional_carry


def get_carry_unnormalized(env_state):
    """Get additional_carry when normalize_env=False.

    State structure: LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: LogEnvState

    Returns:
        additional_carry from the MjxState
    """
    mjx_state, _ = unwrap_to_mjx(env_state.env_state)
    return mjx_state.additional_carry


def update_carry_qvel_w_sum_normalized(env_state, qvel_w_sum):
    """Update qvel_w_sum (reward curriculum weight) when normalize_env=True.

    State structure: NormalizeVecRewEnvState.env_state -> LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: NormalizeVecRewEnvState
        qvel_w_sum: New qvel reward weight value (scalar)

    Returns:
        Updated env_state with new qvel_w_sum
    """
    log_state = env_state.env_state  # LogEnvState
    mjx_state, rebuild = unwrap_to_mjx(log_state.env_state)
    current = mjx_state.additional_carry.qvel_w_sum
    w_sum = jnp.broadcast_to(jnp.asarray(qvel_w_sum, dtype=current.dtype), current.shape)
    new_carry = mjx_state.additional_carry.replace(qvel_w_sum=w_sum)
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    new_log = log_state.replace(env_state=new_inner)
    return env_state.replace(env_state=new_log)


def update_carry_qvel_w_sum_unnormalized(env_state, qvel_w_sum):
    """Update qvel_w_sum (reward curriculum weight) when normalize_env=False.

    State structure: LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: LogEnvState
        qvel_w_sum: New qvel reward weight value (scalar)

    Returns:
        Updated env_state with new qvel_w_sum
    """
    mjx_state, rebuild = unwrap_to_mjx(env_state.env_state)
    current = mjx_state.additional_carry.qvel_w_sum
    w_sum = jnp.broadcast_to(jnp.asarray(qvel_w_sum, dtype=current.dtype), current.shape)
    new_carry = mjx_state.additional_carry.replace(qvel_w_sum=w_sum)
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    return env_state.replace(env_state=new_inner)


def update_carry_reward_weights_normalized(env_state, qvel_w_sum, root_vel_w_sum):
    """Update both qvel_w_sum and root_vel_w_sum when normalize_env=True.

    State structure: NormalizeVecRewEnvState.env_state -> LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: NormalizeVecRewEnvState
        qvel_w_sum: New qvel reward weight value (scalar)
        root_vel_w_sum: New root_vel reward weight value (scalar)

    Returns:
        Updated env_state with new reward weights
    """
    log_state = env_state.env_state  # LogEnvState
    mjx_state, rebuild = unwrap_to_mjx(log_state.env_state)
    current_qvel = mjx_state.additional_carry.qvel_w_sum
    current_root_vel = mjx_state.additional_carry.root_vel_w_sum
    qvel_w = jnp.broadcast_to(jnp.asarray(qvel_w_sum, dtype=current_qvel.dtype), current_qvel.shape)
    root_vel_w = jnp.broadcast_to(jnp.asarray(root_vel_w_sum, dtype=current_root_vel.dtype), current_root_vel.shape)
    new_carry = mjx_state.additional_carry.replace(qvel_w_sum=qvel_w, root_vel_w_sum=root_vel_w)
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    new_log = log_state.replace(env_state=new_inner)
    return env_state.replace(env_state=new_log)


def update_carry_reward_weights_unnormalized(env_state, qvel_w_sum, root_vel_w_sum):
    """Update both qvel_w_sum and root_vel_w_sum when normalize_env=False.

    State structure: LogEnvState.env_state -> [NStepWrapperState ->] MjxState

    Args:
        env_state: LogEnvState
        qvel_w_sum: New qvel reward weight value (scalar)
        root_vel_w_sum: New root_vel reward weight value (scalar)

    Returns:
        Updated env_state with new reward weights
    """
    mjx_state, rebuild = unwrap_to_mjx(env_state.env_state)
    current_qvel = mjx_state.additional_carry.qvel_w_sum
    current_root_vel = mjx_state.additional_carry.root_vel_w_sum
    qvel_w = jnp.broadcast_to(jnp.asarray(qvel_w_sum, dtype=current_qvel.dtype), current_qvel.shape)
    root_vel_w = jnp.broadcast_to(jnp.asarray(root_vel_w_sum, dtype=current_root_vel.dtype), current_root_vel.shape)
    new_carry = mjx_state.additional_carry.replace(qvel_w_sum=qvel_w, root_vel_w_sum=root_vel_w)
    new_mjx = mjx_state.replace(additional_carry=new_carry)
    new_inner = rebuild(new_mjx)
    return env_state.replace(env_state=new_inner)
