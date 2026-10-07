"""Deterministic PRNG streams for scheduled evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def evaluation_stream_keys(
    eval_seed: int,
    validation_number: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return independent rollout and video keys for one validation number."""
    scheduled_key = jax.random.fold_in(
        jax.random.key(int(eval_seed)),
        jnp.asarray(validation_number, dtype=jnp.uint32),
    )
    rollout_key, video_key = jax.random.split(scheduled_key)
    return rollout_key, video_key


def evaluation_rollout_keys(
    eval_seed: int,
    validation_number: int,
    num_devices: int,
) -> jax.Array:
    """Build one deterministic rollout key per data-parallel device."""
    if validation_number < 1:
        raise ValueError("validation_number must be positive")
    if num_devices < 1:
        raise ValueError("num_devices must be positive")
    rollout_key, _ = evaluation_stream_keys(eval_seed, validation_number)
    device_ids = jnp.arange(num_devices, dtype=jnp.uint32)
    return jax.vmap(lambda device_id: jax.random.fold_in(rollout_key, device_id))(device_ids)


def evaluation_video_seed(eval_seed: int, validation_number: int) -> int:
    """Return a stable Python seed for the validation video motion subset."""
    if validation_number < 1:
        raise ValueError("validation_number must be positive")
    _, video_key = evaluation_stream_keys(eval_seed, validation_number)
    key_data = np.asarray(jax.device_get(jax.random.key_data(video_key)), dtype=np.uint32)
    return (int(key_data[0]) << 32) | int(key_data[1])
