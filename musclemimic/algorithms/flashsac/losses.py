"""Probability and distributional Bellman helpers for FlashSAC."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def sample_tanh_normal(
    mean: jax.Array,
    standard_deviation: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Draw a reparameterized tanh-normal sample and corrected log probability."""
    raw_action = mean + standard_deviation * jax.random.normal(rng, mean.shape)
    action = jnp.tanh(raw_action)
    normal_log_probability = -0.5 * (
        jnp.square((raw_action - mean) / standard_deviation)
        + 2.0 * jnp.log(standard_deviation)
        + math.log(2.0 * math.pi)
    )
    # Stable log(1 - tanh(x)^2), the tanh transform Jacobian.
    log_determinant = 2.0 * (math.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action))
    return action, jnp.sum(normal_log_probability - log_determinant, axis=-1)


def select_min_q_distribution(values: jax.Array, log_probabilities: jax.Array) -> jax.Array:
    """Select each sample's categorical distribution from its lower critic."""
    critic_indices = jnp.argmin(values, axis=0)
    batch_indices = jnp.arange(values.shape[1])
    return log_probabilities[critic_indices, batch_indices]


def categorical_td_target(
    target_log_probabilities: jax.Array,
    rewards: jax.Array,
    terminated: jax.Array,
    entropy_term: jax.Array,
    gamma: float,
    min_value: float,
    max_value: float,
) -> jax.Array:
    """Project the soft Bellman target onto the critic's fixed support."""
    batch_size, num_bins = target_log_probabilities.shape
    bins = jnp.linspace(min_value, max_value, num_bins)
    target_bins = rewards[:, None] + gamma * (bins[None, :] - entropy_term[:, None]) * (1.0 - terminated[:, None])
    target_bins = jnp.clip(target_bins, min_value, max_value)

    bin_width = (max_value - min_value) / (num_bins - 1)
    locations = (target_bins - min_value) / bin_width
    lower = jnp.floor(locations).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, num_bins - 1)
    upper_weight = locations - lower

    probabilities = jnp.exp(target_log_probabilities)
    rows = jnp.arange(batch_size)[:, None]
    projected = jnp.zeros_like(probabilities)
    projected = projected.at[rows, lower].add(probabilities * (1.0 - upper_weight))
    projected = projected.at[rows, upper].add(probabilities * upper_weight)
    return projected
