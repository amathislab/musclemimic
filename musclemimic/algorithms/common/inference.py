"""Algorithm-neutral policy calls used by evaluation and visualization tools."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _as_batched_observation(observation: jax.Array) -> jax.Array:
    observation = jnp.asarray(observation)
    return jnp.atleast_2d(observation) if observation.ndim == 1 else observation


def initialize_policy_state(agent_conf: Any, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Initialize the optional recurrent carry behind a common eval interface."""
    observation = _as_batched_observation(observation)
    batch_size = int(observation.shape[0])
    network = getattr(agent_conf, "network", None)
    if network is not None and getattr(network, "is_recurrent", False):
        policy_state = agent_conf.network.initialize_carry(
            batch_size,
            int(agent_conf.network.hidden_size),
            dtype=observation.dtype,
        )
    else:
        policy_state = jnp.zeros((batch_size, 0), dtype=observation.dtype)
    return policy_state, jnp.zeros((batch_size,), dtype=bool)


def make_policy_action_fn(agent_conf: Any, *, deterministic: bool = False):
    """Build a jitted action function for a configured PPO or FlashSAC policy."""
    from musclemimic.algorithms.flashsac.config import FlashSACAgentConf
    from musclemimic.algorithms.ppo.config import PPOAgentConf

    if isinstance(agent_conf, PPOAgentConf) or (
        not isinstance(agent_conf, FlashSACAgentConf) and hasattr(agent_conf, "network")
    ):
        network = agent_conf.network
        recurrent = bool(getattr(network, "is_recurrent", False))

        def sample_ppo(train_state, observation, rng, policy_state, policy_reset):
            observation = _as_batched_observation(observation)
            variables = {"params": train_state.params, "run_stats": train_state.run_stats}
            if recurrent:
                policy_state, distribution, _ = network.apply(
                    variables,
                    policy_state,
                    (observation[None, ...], policy_reset[None, ...]),
                )
            else:
                distribution, _ = network.apply(variables, observation)

            action = distribution.mode() if deterministic else distribution.sample(seed=rng)
            if recurrent:
                action = jnp.squeeze(action, axis=0)
            if action.ndim > 1 and action.shape[0] == 1:
                action = action[0]
            return action, train_state, policy_state

        return jax.jit(sample_ppo)

    if isinstance(agent_conf, FlashSACAgentConf):
        actor = agent_conf.actor
        observation_indices = agent_conf.actor_observation_indices

        def sample_flashsac(train_state, observation, rng, policy_state, _policy_reset):
            observation = _as_batched_observation(observation)
            actor_observation = observation[:, observation_indices]
            mean, standard_deviation = actor.apply(
                {
                    "params": train_state.actor_params,
                    "batch_stats": train_state.actor_batch_stats,
                },
                actor_observation,
                training=False,
            )
            pre_tanh = mean
            if not deterministic:
                pre_tanh = mean + standard_deviation * jax.random.normal(rng, mean.shape)
            action = jnp.tanh(pre_tanh)
            if action.ndim > 1 and action.shape[0] == 1:
                action = action[0]
            return action, train_state, policy_state

        return jax.jit(sample_flashsac)

    raise TypeError(f"Unsupported agent configuration: {type(agent_conf).__name__}")
