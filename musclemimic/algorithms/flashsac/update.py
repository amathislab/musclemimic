"""One FlashSAC gradient update."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from musclemimic.algorithms.common.distributed import axis_size
from musclemimic.algorithms.common.distributed import pmean as _mean

from .config import FlashSACAgentConf
from .losses import categorical_td_target, sample_tanh_normal, select_min_q_distribution
from .networks import normalize_parameters
from .replay_buffer import ReplayBatch
from .state import FlashSACTrainState


class FlashSACUpdateMetrics(NamedTuple):
    actor_loss: jax.Array
    critic_loss: jax.Array
    temperature_loss: jax.Array
    temperature: jax.Array
    entropy: jax.Array
    q_value: jax.Array
    has_actor_update: jax.Array
    has_critic_update: jax.Array


def _update_actor(
    state: FlashSACTrainState,
    batch: ReplayBatch,
    rng: jax.Array,
    agent_conf: FlashSACAgentConf,
    target_entropy: jax.Array,
    axis_name: str | None,
) -> tuple:
    actor_observations = batch.observations[:, agent_conf.actor_observation_indices]
    actor_next_observations = batch.next_observations[:, agent_conf.actor_observation_indices]
    critic_observations = batch.observations[:, agent_conf.critic_observation_indices]
    all_actor_observations = jnp.concatenate((actor_observations, actor_next_observations))

    def actor_loss_fn(actor_params):
        (mean_all, standard_deviation_all), mutable = agent_conf.actor.apply(
            {"params": actor_params, "batch_stats": state.actor_batch_stats},
            all_actor_observations,
            training=True,
            axis_name=axis_name,
            mutable=["batch_stats"],
        )
        batch_size = actor_observations.shape[0]
        actions, log_probabilities = sample_tanh_normal(
            mean_all[:batch_size],
            standard_deviation_all[:batch_size],
            rng,
        )
        q_values, _ = agent_conf.critic.apply(
            {"params": state.critic_params, "batch_stats": state.critic_batch_stats},
            critic_observations,
            actions,
            training=False,
        )
        minimum_q = jnp.minimum(q_values[0], q_values[1])
        temperature = jax.lax.stop_gradient(jnp.exp(state.log_temperature))
        local_loss = jnp.mean(temperature * log_probabilities - minimum_q)
        gradient_loss = local_loss / axis_size(axis_name)
        return gradient_loss, (
            mutable["batch_stats"],
            _mean(-jnp.mean(log_probabilities), axis_name),
            _mean(jnp.mean(minimum_q), axis_name),
            _mean(local_loss, axis_name),
        )

    (_, (actor_batch_stats, entropy, q_value, actor_loss)), actor_gradients = jax.value_and_grad(
        actor_loss_fn,
        has_aux=True,
    )(state.actor_params)
    actor_updates, actor_optimizer_state = agent_conf.actor_optimizer.update(
        actor_gradients,
        state.actor_optimizer_state,
        state.actor_params,
    )
    actor_params = normalize_parameters(optax.apply_updates(state.actor_params, actor_updates))

    def temperature_loss_fn(log_temperature):
        loss = jnp.exp(log_temperature) * (jax.lax.stop_gradient(entropy) - target_entropy)
        return loss / axis_size(axis_name), loss

    (_, temperature_loss), temperature_gradients = jax.value_and_grad(
        temperature_loss_fn,
        has_aux=True,
    )(state.log_temperature)
    temperature_updates, temperature_optimizer_state = agent_conf.temperature_optimizer.update(
        temperature_gradients,
        state.temperature_optimizer_state,
        state.log_temperature,
    )
    log_temperature = jnp.clip(
        optax.apply_updates(state.log_temperature, temperature_updates),
        min=-10.0,
        max=2.0,
    )
    return (
        actor_params,
        actor_batch_stats,
        actor_optimizer_state,
        log_temperature,
        temperature_optimizer_state,
        actor_loss,
        temperature_loss,
        entropy,
        q_value,
    )


def update_from_batch(
    state: FlashSACTrainState,
    batch: ReplayBatch,
    rng: jax.Array,
    agent_conf: FlashSACAgentConf,
    *,
    axis_name: str | None = None,
) -> tuple[FlashSACTrainState, FlashSACUpdateMetrics]:
    """Update actor periodically, then critic and target critic."""
    config = agent_conf.config.experiment
    if bool(config.normalize_reward):
        return_scale = jnp.maximum(
            jnp.sqrt(state.reward_return_var + 1e-8),
            state.reward_return_max / float(config.normalized_return_max),
        )
        batch = batch._replace(rewards=batch.rewards / return_scale)

    actor_rng, critic_rng = jax.random.split(rng)
    action_dim = state.replay_buffer.actions.shape[-1]
    target_entropy = 0.5 * action_dim * jnp.log(2.0 * jnp.pi * jnp.e * float(config.target_sigma) ** 2)

    def update_actor(_unused):
        return _update_actor(
            state,
            batch,
            actor_rng,
            agent_conf,
            target_entropy,
            axis_name,
        )

    def keep_actor(_unused):
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return (
            state.actor_params,
            state.actor_batch_stats,
            state.actor_optimizer_state,
            state.log_temperature,
            state.temperature_optimizer_state,
            zero,
            zero,
            zero,
            zero,
        )

    has_actor_update = state.update_step % int(config.actor_update_period) == 0
    (
        actor_params,
        actor_batch_stats,
        actor_optimizer_state,
        log_temperature,
        temperature_optimizer_state,
        actor_loss,
        temperature_loss,
        entropy,
        q_value,
    ) = jax.lax.cond(
        has_actor_update,
        update_actor,
        keep_actor,
        operand=None,
    )

    actor_next_observations = batch.next_observations[:, agent_conf.actor_observation_indices]
    observations = batch.observations[:, agent_conf.critic_observation_indices]
    next_observations = batch.next_observations[:, agent_conf.critic_observation_indices]
    next_mean, next_standard_deviation = agent_conf.actor.apply(
        {"params": actor_params, "batch_stats": actor_batch_stats},
        actor_next_observations,
        training=False,
    )
    next_actions, next_log_probabilities = sample_tanh_normal(
        next_mean,
        next_standard_deviation,
        critic_rng,
    )
    next_actions = jax.lax.stop_gradient(next_actions)
    next_log_probabilities = jax.lax.stop_gradient(next_log_probabilities)

    all_observations = jnp.concatenate((observations, next_observations))
    all_actions = jnp.concatenate((batch.actions, next_actions))
    (target_values_all, target_log_probabilities_all), target_mutable = agent_conf.critic.apply(
        {
            "params": state.target_critic_params,
            "batch_stats": state.target_critic_batch_stats,
        },
        all_observations,
        all_actions,
        training=True,
        axis_name=axis_name,
        mutable=["batch_stats"],
    )
    batch_size = observations.shape[0]
    selected_target_log_probabilities = select_min_q_distribution(
        target_values_all[:, batch_size:],
        target_log_probabilities_all[:, batch_size:],
    )
    target_probabilities = categorical_td_target(
        target_log_probabilities=selected_target_log_probabilities,
        rewards=batch.rewards,
        terminated=batch.terminated.astype(jnp.float32),
        entropy_term=jnp.exp(log_temperature) * next_log_probabilities,
        gamma=float(config.gamma),
        min_value=float(config.critic_min_value),
        max_value=float(config.critic_max_value),
    )
    target_probabilities = jax.lax.stop_gradient(target_probabilities)

    def critic_loss_fn(critic_params):
        (_values, predicted_log_probabilities_all), mutable = agent_conf.critic.apply(
            {"params": critic_params, "batch_stats": state.critic_batch_stats},
            all_observations,
            all_actions,
            training=True,
            axis_name=axis_name,
            mutable=["batch_stats"],
        )
        predicted_log_probabilities = predicted_log_probabilities_all[:, :batch_size]
        cross_entropy = -jnp.sum(
            target_probabilities[None, ...] * predicted_log_probabilities,
            axis=-1,
        )
        local_loss = jnp.mean(cross_entropy)
        return local_loss / axis_size(axis_name), (
            mutable["batch_stats"],
            _mean(local_loss, axis_name),
        )

    (_, (critic_batch_stats, critic_loss)), critic_gradients = jax.value_and_grad(
        critic_loss_fn,
        has_aux=True,
    )(state.critic_params)
    critic_updates, critic_optimizer_state = agent_conf.critic_optimizer.update(
        critic_gradients,
        state.critic_optimizer_state,
        state.critic_params,
    )
    critic_params = normalize_parameters(optax.apply_updates(state.critic_params, critic_updates))
    tau = float(config.target_update_tau)
    target_critic_params = jax.tree_util.tree_map(
        lambda target, source: (1.0 - tau) * target + tau * source,
        state.target_critic_params,
        critic_params,
    )

    next_state = state.replace(
        actor_params=actor_params,
        actor_batch_stats=actor_batch_stats,
        actor_optimizer_state=actor_optimizer_state,
        critic_params=critic_params,
        critic_batch_stats=critic_batch_stats,
        critic_optimizer_state=critic_optimizer_state,
        target_critic_params=target_critic_params,
        target_critic_batch_stats=target_mutable["batch_stats"],
        log_temperature=log_temperature,
        temperature_optimizer_state=temperature_optimizer_state,
        update_step=state.update_step + 1,
    )
    metrics = FlashSACUpdateMetrics(
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        temperature_loss=temperature_loss,
        temperature=jnp.exp(log_temperature),
        entropy=entropy,
        q_value=q_value,
        has_actor_update=has_actor_update,
        has_critic_update=jnp.asarray(True, dtype=jnp.bool_),
    )
    return next_state, metrics
