"""Discounted-return state used by FlashSAC reward normalization."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def update_discounted_return(
    previous_return: jax.Array,
    rewards: jax.Array,
    done: jax.Array,
    *,
    gamma: float,
) -> tuple[jax.Array, jax.Array]:
    """Return statistic samples and next carry, each shaped ``[num_envs]``.

    The statistic includes the terminal transition in the current episode. The
    carry is then cleared before the next auto-reset episode.
    """
    done = done.astype(jnp.bool_)
    samples = gamma * previous_return + rewards
    next_return = jnp.where(done, jnp.zeros_like(samples), samples)
    return samples, next_return
