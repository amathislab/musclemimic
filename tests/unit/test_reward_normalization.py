"""Reward normalization state and multi-device numerical semantics."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from musclemimic.core.wrappers.mjx import NormalizeVecReward, NormalizeVecRewEnvState


def _state(return_value):
    return NormalizeVecRewEnvState(
        env_state=jnp.asarray(0),
        mean=jnp.asarray(0.0),
        var=jnp.asarray(1.0),
        count=jnp.asarray(1e-4),
        return_val=jnp.asarray(return_value, dtype=jnp.float32),
    )


def test_terminal_reward_is_counted_before_return_carry_is_reset():
    wrapper = NormalizeVecReward(SimpleNamespace(), gamma=0.99)
    reward = jnp.asarray([2.0, 3.0])
    done = jnp.asarray([True, False])

    next_state, _ = wrapper._normalize_step_output(
        _state([5.0, 7.0]),
        reward,
        done,
        jnp.asarray(1),
    )

    discounted_returns = jnp.asarray([5.0 * 0.99 + 2.0, 7.0 * 0.99 + 3.0])
    np.testing.assert_allclose(np.asarray(next_state.return_val), [0.0, discounted_returns[1]])
    expected_mean = jnp.mean(discounted_returns) * 2.0 / 2.0001
    np.testing.assert_allclose(np.asarray(next_state.mean), np.asarray(expected_mean), atol=1e-6)
