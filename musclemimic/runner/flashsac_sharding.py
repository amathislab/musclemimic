"""FlashSAC array layout at the compiled data-parallel boundary."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.distributed import mark_data_varying

from .data_parallel import DATA, REPLICATED, data_specs, replicated_specs


def train_state_specs(train_state: Any | None) -> Any | None:
    """Replicate learning state while partitioning replay storage."""
    if train_state is None:
        return None
    specs = replicated_specs(train_state)
    if not hasattr(specs, "replay_buffer"):
        return specs
    replay = specs.replay_buffer.replace(
        observations=DATA,
        actions=DATA,
        rewards=DATA,
        next_observations=DATA,
        terminated=DATA,
    )
    return specs.replace(replay_buffer=replay)


def runtime_specs(runtime_state: Any) -> Any:
    """Describe replicated FlashSAC learning state and sharded rollout state."""
    specs = replicated_specs(runtime_state)
    return specs.replace(
        agent=train_state_specs(runtime_state.agent),
        env_state=jax.tree.map(
            lambda value: DATA if value.ndim > 0 else REPLICATED,
            runtime_state.env_state,
        ),
        observation=data_specs(runtime_state.observation),
        exploration_noise=data_specs(runtime_state.exploration_noise),
        noise_steps_remaining=data_specs(runtime_state.noise_steps_remaining),
        rng=DATA,
        validation_rng=DATA,
        discounted_return=data_specs(runtime_state.discounted_return),
    )


def mark_runtime_data_varying(runtime_state: Any, axis_name: str | None) -> Any:
    """Mark zero-initialized local buffers as distinct data-shard values."""
    return runtime_state.replace(
        env_state=mark_data_varying(runtime_state.env_state, axis_name),
        observation=mark_data_varying(runtime_state.observation, axis_name),
        exploration_noise=mark_data_varying(runtime_state.exploration_noise, axis_name),
        noise_steps_remaining=mark_data_varying(runtime_state.noise_steps_remaining, axis_name),
        discounted_return=mark_data_varying(runtime_state.discounted_return, axis_name),
    )


def add_device_key_axis(runtime_state: Any) -> Any:
    """Represent each shard-local key as one global per-device row."""
    return runtime_state.replace(
        rng=jnp.expand_dims(runtime_state.rng, axis=0),
        validation_rng=jnp.expand_dims(runtime_state.validation_rng, axis=0),
    )


def remove_device_key_axis(runtime_state: Any) -> Any:
    """Recover shard-local keys before calling FlashSAC numerical code."""
    return runtime_state.replace(
        rng=jnp.squeeze(runtime_state.rng, axis=0),
        validation_rng=jnp.squeeze(runtime_state.validation_rng, axis=0),
    )
