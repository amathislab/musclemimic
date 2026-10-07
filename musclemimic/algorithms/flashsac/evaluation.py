"""Compiled validation rollouts for FlashSAC."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.env_state_utils import (
    preserve_data_axis_variation,
)
from musclemimic.algorithms.common.trajectory_evaluation import (
    evaluate_exact_trajectory_set_local,
)
from musclemimic.algorithms.common.validation import (
    run_validation_rollout,
)

from .config import FlashSACAgentConf
from .state import FlashSACPolicyState, FlashSACTrainState


class FlashSACValidationResult(NamedTuple):
    metrics: Any
    rng: jax.Array


def _actor_actions(
    agent_conf: FlashSACAgentConf,
    policy_state: FlashSACPolicyState | FlashSACTrainState,
    observations: jax.Array,
    rng: jax.Array,
    *,
    deterministic: bool,
) -> jax.Array:
    actor_observations = observations[:, agent_conf.actor_observation_indices]
    mean, standard_deviation = agent_conf.actor.apply(
        {
            "params": policy_state.actor_params,
            "batch_stats": policy_state.actor_batch_stats,
        },
        actor_observations,
        training=False,
    )
    noise = jnp.zeros_like(mean) if deterministic else jax.random.normal(rng, mean.shape)
    return jnp.tanh(mean + standard_deviation * noise)


def run_validation_step(
    agent_conf: FlashSACAgentConf,
    policy_state: FlashSACPolicyState | FlashSACTrainState,
    rng: jax.Array,
    env: Any,
    config: Any,
    metrics_handler: Any,
    *,
    reset_weights: jax.Array | None = None,
    trajectory_ids: jax.Array | None = None,
    trajectory_lengths: jax.Array | None = None,
    trajectory_mask: jax.Array | None = None,
    axis_name: str | None = None,
) -> FlashSACValidationResult:
    """Evaluate the current actor for one configured validation rollout."""

    def policy_action(observations: jax.Array, action_rng: jax.Array) -> jax.Array:
        return _actor_actions(
            agent_conf,
            policy_state,
            observations,
            action_rng,
            deterministic=bool(config.validation.get("deterministic", False)),
        )

    if bool(config.validation.get("evaluate_all", False)):
        if trajectory_ids is None or trajectory_lengths is None or trajectory_mask is None:
            raise ValueError("Exact validation requires trajectory ids, lengths, and a padding mask.")
        exact_result = evaluate_exact_trajectory_set_local(
            policy_action,
            rng,
            env,
            trajectory_ids=trajectory_ids,
            trajectory_lengths=trajectory_lengths,
            trajectory_mask=trajectory_mask,
            num_steps=int(config.validation.num_steps),
            axis_name=axis_name,
        )
        return FlashSACValidationResult(
            metrics=exact_result.metrics,
            rng=exact_result.rng,
        )

    rollout_result = run_validation_rollout(
        policy_action=policy_action,
        rng=rng,
        env=env,
        config=config,
        metrics_handler=metrics_handler,
        reset_weights=reset_weights,
        axis_name=axis_name,
        preserve_next_env_state=preserve_data_axis_variation,
    )
    return FlashSACValidationResult(
        metrics=rollout_result.metrics,
        rng=rollout_result.rng,
    )
