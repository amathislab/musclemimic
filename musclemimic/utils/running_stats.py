"""Numerically stable running-moment utilities for batched JAX data."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def global_batch_moments(values: jax.Array, axis_name: str | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return mean, variance, and count over a local or pmapped batch."""
    local_count = jnp.asarray(values.shape[0], dtype=values.dtype)
    local_sum = jnp.sum(values, axis=0)

    if axis_name is None:
        count = local_count
        total = local_sum
    else:
        count = jax.lax.psum(local_count, axis_name=axis_name)
        total = jax.lax.psum(local_sum, axis_name=axis_name)

    mean = total / count
    local_squared_deviation = jnp.sum(jnp.square(values - mean), axis=0)
    if axis_name is not None:
        local_squared_deviation = jax.lax.psum(local_squared_deviation, axis_name=axis_name)
    variance = local_squared_deviation / count
    return mean, variance, count


def finite_global_batch_moments(
    values: jax.Array,
    axis_name: str | None = None,
    valid: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return moments over finite values, excluding invalid lanes globally.

    ``values`` is expected to have one scalar per environment.  An optional
    ``valid`` mask can exclude otherwise-finite lanes.  A replica with no valid
    lanes contributes a zero count; if every replica is empty, the returned
    mean and variance are finite zeros with a zero count so callers can retain
    their previous running moments.
    """
    values = jnp.asarray(values)
    finite = jnp.isfinite(values)
    if valid is not None:
        finite = jnp.logical_and(finite, jnp.asarray(valid, dtype=bool))

    safe_values = jnp.where(finite, values, 0.0)
    local_count = jnp.sum(finite, dtype=values.dtype)
    local_sum = jnp.sum(safe_values)

    if axis_name is None:
        count = local_count
        total = local_sum
    else:
        count = jax.lax.psum(local_count, axis_name=axis_name)
        total = jax.lax.psum(local_sum, axis_name=axis_name)

    safe_count = jnp.maximum(count, jnp.asarray(1.0, dtype=values.dtype))
    mean = total / safe_count
    local_squared_deviation = jnp.sum(
        jnp.where(finite, jnp.square(values - mean), 0.0)
    )
    if axis_name is not None:
        local_squared_deviation = jax.lax.psum(local_squared_deviation, axis_name=axis_name)
    variance = local_squared_deviation / safe_count
    return mean, variance, count


def merge_running_moments(
    mean: jax.Array,
    variance: jax.Array,
    count: jax.Array,
    batch_mean: jax.Array,
    batch_variance: jax.Array,
    batch_count: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Merge two population-moment summaries with Welford's formula."""
    count = jnp.asarray(count)
    batch_count = batch_count.astype(count.dtype)
    total_count = count + batch_count
    delta = batch_mean - mean
    new_mean = mean + delta * batch_count / total_count

    old_m2 = variance * count
    batch_m2 = batch_variance * batch_count
    correction = jnp.square(delta) * count * batch_count / total_count
    new_variance = (old_m2 + batch_m2 + correction) / total_count
    return new_mean, new_variance, total_count
