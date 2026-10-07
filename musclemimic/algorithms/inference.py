"""Policy calls shared by supported JAX algorithms."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def make_policy_action_fn(agent_conf: Any, *, deterministic: bool) -> Any:
    """Build a jitted action call for PPO or FlashSAC."""
    from musclemimic.algorithms.flashsac.config import FlashSACAgentConf
    from musclemimic.algorithms.ppo.config import PPOAgentConf

    if isinstance(agent_conf, PPOAgentConf):

        def sample_ppo(train_state, observation, rng):
            observation = _as_batch(observation)
            distribution = agent_conf.network.apply(
                {"params": train_state.params, "run_stats": train_state.run_stats},
                observation,
                actor_only=True,
            )
            action = distribution.mode() if deterministic else distribution.sample(seed=rng)
            return _remove_singleton_batch(action), train_state

        return jax.jit(sample_ppo)

    if isinstance(agent_conf, FlashSACAgentConf):

        def sample_flashsac(train_state, observation, rng):
            observation = _as_batch(observation)
            actor_observation = observation[:, agent_conf.actor_observation_indices]
            mean, standard_deviation = agent_conf.actor.apply(
                {
                    "params": train_state.actor_params,
                    "batch_stats": train_state.actor_batch_stats,
                },
                actor_observation,
                training=False,
            )
            noise = jnp.zeros_like(mean) if deterministic else jax.random.normal(rng, mean.shape)
            return _remove_singleton_batch(jnp.tanh(mean + standard_deviation * noise)), train_state

        return jax.jit(sample_flashsac)

    raise TypeError(f"Unsupported agent configuration: {type(agent_conf).__name__}")


def _as_batch(observation: jax.Array) -> jax.Array:
    observation = jnp.asarray(observation)
    return jnp.atleast_2d(observation) if observation.ndim == 1 else observation


def _remove_singleton_batch(action: jax.Array) -> jax.Array:
    return action[0] if action.ndim > 1 and action.shape[0] == 1 else action
