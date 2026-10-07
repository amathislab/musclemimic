"""PPO data-parallel array layout at the compiled runner boundary."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .data_parallel import (
    DATA,
    REPLICATED,
    data_specs,
    put_tree,
    replicated_specs,
    trajectory_specs,
)


def runtime_specs(runtime_state: Any) -> Any:
    """Describe replicated PPO state and data-sharded environment state."""
    specs = replicated_specs(runtime_state)
    updates = {}
    if hasattr(runtime_state, "env_state"):
        updates["env_state"] = jax.tree.map(
            lambda value: DATA if value.ndim > 0 else REPLICATED,
            runtime_state.env_state,
        )
    if hasattr(runtime_state, "last_obs"):
        updates["last_obs"] = data_specs(runtime_state.last_obs)
    for name in ("rng", "val_rng"):
        if hasattr(runtime_state, name):
            updates[name] = data_specs(getattr(runtime_state, name))
    return specs.replace(**updates) if updates else specs


def add_device_key_axis(runtime_state: Any) -> Any:
    """Represent each shard-local PRNG key as one global per-device row."""
    updates = {
        name: jnp.expand_dims(getattr(runtime_state, name), axis=0)
        for name in ("rng", "val_rng")
        if hasattr(runtime_state, name)
    }
    return runtime_state.replace(**updates) if updates else runtime_state


def remove_device_key_axis(runtime_state: Any) -> Any:
    """Recover shard-local PRNG keys before calling PPO numerical code."""
    updates = {
        name: jnp.squeeze(getattr(runtime_state, name), axis=0)
        for name in ("rng", "val_rng")
        if hasattr(runtime_state, name)
    }
    return runtime_state.replace(**updates) if updates else runtime_state


def localize_resume_state(resume_state: Any | None) -> Any | None:
    """Remove the per-device key row from an exact-resume input."""
    if not isinstance(resume_state, dict) or "runtime_state" not in resume_state:
        return resume_state
    runtime_state = dict(resume_state["runtime_state"])
    for name in ("rng", "val_rng"):
        runtime_state[name] = jnp.squeeze(runtime_state[name], axis=0)
    return {**resume_state, "runtime_state": runtime_state}


def resume_specs(resume_state: Any | None, state_specs: Any) -> Any | None:
    """Build the input layout for a model-only or exact-resume checkpoint."""
    if resume_state is None:
        return None
    specs = replicated_specs(resume_state)
    if isinstance(resume_state, dict) and "runtime_state" in resume_state:
        specs = {
            **specs,
            "runtime_state": state_specs.to_checkpoint_state(),
        }
    return specs


def validation_reset_specs(training: Any) -> jax.sharding.PartitionSpec | None:
    """Describe validation sampling weights returned by each data shard."""
    adaptive = bool(training.config.get("adaptive_sampling", {}).get("enabled", False))
    if training.validation_shares_training_trajectory:
        if adaptive:
            return DATA if training.shard_spec is not None else REPLICATED
        return DATA if training.shard_spec is not None else None
    return DATA if training.validation_shard_spec is not None else None
