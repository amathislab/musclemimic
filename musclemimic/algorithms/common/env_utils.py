"""
Environment wrapping and observation utilities for RL training.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from musclemimic.core.wrappers import (
    AutoResetWrapper,
    LogWrapper,
    NormalizeVecReward,
    NStepWrapper,
    VecEnv,
)


def expand_obs_indices_for_history(
    obs_ind: jnp.ndarray,
    env: Any,
    config: DictConfig,
) -> jnp.ndarray:
    """
    Expand observation indices for history stacking.

    When split_goal=True, the observation layout is [state_hist, goal],
    where only non-goal indices are stacked and goal indices stay current.

    Args:
        obs_ind: Original observation indices (before history expansion)
        env: Environment instance (needs obs_container with goal group for split_goal)
        config: Experiment config with len_obs_history and split_goal

    Returns:
        Expanded observation indices for the history-stacked observation
    """
    n_steps = config.len_obs_history
    split_goal = config.get("split_goal", False)

    if split_goal:
        if not hasattr(env, "obs_container"):
            raise ValueError("split_goal=True requires env.obs_container with goal group indices")
        goal_indices = env.obs_container.get_obs_ind_by_group("goal")
        if goal_indices.size == 0:
            raise ValueError("split_goal=True requires goal observations grouped as 'goal'")

        raw_obs_dim = env.info.observation_space.shape[0]
        goal_indices = np.asarray(goal_indices, dtype=int)
        raw_indices = np.arange(raw_obs_dim, dtype=int)
        state_mask = np.ones(raw_obs_dim, dtype=bool)
        state_mask[goal_indices] = False
        state_indices = raw_indices[state_mask]

        state_dim = state_indices.size

        state_index_map = np.full(raw_obs_dim, -1, dtype=int)
        state_index_map[state_indices] = np.arange(state_dim, dtype=int)
        goal_index_map = np.full(raw_obs_dim, -1, dtype=int)
        goal_index_map[goal_indices] = np.arange(goal_indices.size, dtype=int)

        obs_ind_np = np.asarray(obs_ind)
        state_positions = state_index_map[obs_ind_np]
        goal_positions = goal_index_map[obs_ind_np]
        state_positions = state_positions[state_positions >= 0]
        goal_positions = goal_positions[goal_positions >= 0]

        dtype = obs_ind_np.dtype
        if state_positions.size:
            state_expanded = np.concatenate([state_positions + i * state_dim for i in range(n_steps)]).astype(dtype)
        else:
            state_expanded = np.array([], dtype=dtype)

        if goal_positions.size:
            goal_expanded = (goal_positions + n_steps * state_dim).astype(dtype)
        else:
            goal_expanded = np.array([], dtype=dtype)

        return jnp.array(np.concatenate([state_expanded, goal_expanded]), dtype=obs_ind_np.dtype)
    else:
        # Original behavior: full obs stacking
        obs_len = env.info.observation_space.shape[0]
        return jnp.concatenate([obs_ind + i * obs_len for i in range(n_steps)])


def wrap_env(
    env: Any,
    config: DictConfig,
    *,
    axis_name: str | None = None,
    normalize_reward: bool | None = None,
) -> Any:
    """
    Apply standard wrappers for RL training.

    Wrapper order for MJX: VecEnv -> LogWrapper -> AutoResetWrapper
    LogWrapper must see real done flags before AutoResetWrapper clears them.

    Args:
        env: base environment
        config: experiment config with normalize_env, gamma, len_obs_history
        axis_name: pmap axis used to aggregate reward statistics globally

    Returns:
        wrapped environment
    """
    if "len_obs_history" in config and config.len_obs_history > 1:
        split_goal = config.get("split_goal", False)
        env = NStepWrapper(env, config.len_obs_history, split_goal=split_goal)

    if hasattr(env, "mjx_env") and bool(env.mjx_env):
        env = VecEnv(env)
        env = LogWrapper(env)
        env = AutoResetWrapper(env)
    else:
        env = LogWrapper(env)
        env = VecEnv(env)

    should_normalize_reward = bool(config.normalize_env) if normalize_reward is None else normalize_reward
    if should_normalize_reward:
        env = NormalizeVecReward(env, config.gamma, axis_name=axis_name)

    return env
