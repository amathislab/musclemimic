"""PPO parameter initialization and learning updates."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.checkpoint_utils import reset_lr_schedule_count
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.algorithms.ppo.loss import (
    approx_kl,
    normalize_advantages_global,
    ppo_actor_loss,
    ppo_value_loss,
)
from musclemimic.algorithms.ppo.moe import aggregate_moe_metrics, zero_moe_metrics
from musclemimic.rl_core import create_minibatches


class PPOUpdateMetrics(NamedTuple):
    """Scalar metrics produced by one PPO update."""

    total_loss: jax.Array
    value_loss: jax.Array
    policy_loss: jax.Array
    entropy: jax.Array
    approximate_kl: jax.Array
    moe_loss: jax.Array
    gate_entropy: jax.Array
    expert_utilization_variance: jax.Array
    top2_expert_usage: jax.Array
    gate_weight_mean: jax.Array
    gate_weight_std: jax.Array
    ratio_mean: jax.Array
    ratio_std: jax.Array
    ratio_min: jax.Array
    ratio_max: jax.Array
    clipped_ratio_fraction: jax.Array


def initialize_train_state(
    rng: jax.Array,
    env: Any,
    network: Any,
    optimizer: Any,
    agent_state: Any | None,
    config: Any,
    *,
    apply_resume_resets: bool = True,
) -> TrainState:
    """Create a fresh train state or restore one from an agent checkpoint."""
    if agent_state is None:
        _, init_rng, _ = jax.random.split(rng, 3)
        initial_observation = jnp.zeros(env.info.observation_space.shape)
        variables = network.init(init_rng, initial_observation)
        return TrainState.create(
            apply_fn=network.apply,
            params=variables["params"],
            run_stats=variables["run_stats"],
            tx=optimizer,
        )

    restored = agent_state.train_state
    optimizer_state = restored.opt_state
    reset_learning_rate = apply_resume_resets and bool(config.get("reset_lr_schedule_on_resume", False))
    if reset_learning_rate:
        print("[resume] Resetting LR schedule counter to 0 (keeping optimizer momentum)")
        optimizer_state = reset_lr_schedule_count(optimizer_state)

    params = restored.params
    reset_std = config.get("reset_std_on_resume", None) if apply_resume_resets else None
    if reset_std is not None and "log_std" in params:
        params = {
            **params,
            "log_std": jnp.full_like(params["log_std"], jnp.log(reset_std)),
        }
        print(f"[resume] Reset action std -> {reset_std}")

    if not apply_resume_resets:
        skipped_resets = []
        if config.get("reset_lr_schedule_on_resume", False):
            skipped_resets.append("lr_schedule")
        if config.get("reset_std_on_resume", None) is not None:
            skipped_resets.append("action_std")
        if skipped_resets:
            print(
                "[resume] Skipping one-shot resume resets for local auto-resume "
                f"checkpoint: {', '.join(skipped_resets)}"
            )

    return TrainState(
        apply_fn=network.apply,
        params=params,
        tx=optimizer,
        opt_state=optimizer_state,
        step=0 if reset_learning_rate else restored.step,
        run_stats=restored.run_stats,
    )


def update_observation_stats(
    train_state: TrainState,
    observations: jax.Array,
    *,
    network: Any,
    axis_name: str | None = None,
) -> TrainState:
    """Merge one rollout's global observations into running statistics."""
    flat_observations = observations.reshape((-1, observations.shape[-1]))
    _, updates = network.apply(
        {"params": train_state.params, "run_stats": train_state.run_stats},
        flat_observations,
        axis_name=axis_name,
        method=network.update_observation_stats,
        mutable=["run_stats"],
    )
    return train_state.replace(run_stats=updates["run_stats"])


def update(
    train_state: TrainState,
    rollout: Any,
    advantages: jax.Array,
    targets: jax.Array,
    rng: jax.Array,
    learning_rate: jax.Array,
    *,
    network: Any,
    config: Any,
    base_learning_rate: float,
    use_adaptive_learning_rate: bool,
    axis_name: str | None = None,
) -> tuple[TrainState, jax.Array, jax.Array, PPOUpdateMetrics]:
    """Run all PPO minibatch and epoch updates for one rollout."""

    def _replica_mean(value):
        if axis_name is None:
            return value
        return jax.lax.pmean(value, axis_name=axis_name)

    def _update_epoch(epoch_state, _unused):
        def _update_minibatch(current_train_state, batch):
            minibatch, minibatch_advantages, minibatch_targets = batch

            def _loss(params):
                use_moe = bool(config.get("use_moe", False))
                if use_moe:
                    policy, value, moe_metrics = network.apply(
                        {
                            "params": params,
                            "run_stats": current_train_state.run_stats,
                        },
                        minibatch.obs,
                        return_metrics=True,
                        axis_name=axis_name,
                    )
                    (
                        moe_loss,
                        gate_entropy,
                        expert_variance,
                        top2_usage,
                        gate_weight_mean,
                        gate_weight_std,
                    ) = aggregate_moe_metrics(moe_metrics)
                else:
                    policy, value = network.apply(
                        {
                            "params": params,
                            "run_stats": current_train_state.run_stats,
                        },
                        minibatch.obs,
                        axis_name=axis_name,
                    )
                    (
                        moe_loss,
                        gate_entropy,
                        expert_variance,
                        top2_usage,
                        gate_weight_mean,
                        gate_weight_std,
                    ) = zero_moe_metrics()

                log_probability = policy.log_prob(minibatch.action)
                entropy = policy.entropy().mean()
                value_loss = ppo_value_loss(
                    value,
                    minibatch.value,
                    minibatch_targets,
                    config.get("clip_eps_vf", config.clip_eps),
                )
                normalized_advantages = normalize_advantages_global(
                    minibatch_advantages,
                    axis_name=axis_name,
                )
                policy_loss, ratio_stats = ppo_actor_loss(
                    log_probability,
                    minibatch.log_prob,
                    normalized_advantages,
                    config.clip_eps,
                    return_ratio_stats=True,
                    axis_name=axis_name,
                )
                approximate_kl = approx_kl(minibatch.log_prob, log_probability)
                total_loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy + moe_loss
                metrics = PPOUpdateMetrics(
                    total_loss=total_loss,
                    value_loss=value_loss,
                    policy_loss=policy_loss,
                    entropy=entropy,
                    approximate_kl=approximate_kl,
                    moe_loss=moe_loss,
                    gate_entropy=gate_entropy,
                    expert_utilization_variance=expert_variance,
                    top2_expert_usage=top2_usage,
                    gate_weight_mean=gate_weight_mean,
                    gate_weight_std=gate_weight_std,
                    ratio_mean=ratio_stats["ratio_mean"],
                    ratio_std=ratio_stats["ratio_std"],
                    ratio_min=ratio_stats["ratio_min"],
                    ratio_max=ratio_stats["ratio_max"],
                    clipped_ratio_fraction=ratio_stats["clipped_ratio_frac"],
                )
                return total_loss, metrics

            (_, metrics), gradients = jax.value_and_grad(_loss, has_aux=True)(current_train_state.params)
            gradients = jax.tree.map(_replica_mean, gradients)
            metrics = jax.tree.map(_replica_mean, metrics)

            if use_adaptive_learning_rate:
                scale = learning_rate / jnp.asarray(
                    base_learning_rate,
                    dtype=jnp.float32,
                )
                gradients = jax.tree.map(
                    lambda gradient: gradient * scale.astype(gradient.dtype),
                    gradients,
                )

            current_train_state = current_train_state.apply_gradients(grads=gradients)
            return current_train_state, metrics

        current_train_state, rng = epoch_state
        rng, permutation_rng = jax.random.split(rng)
        minibatches = create_minibatches(
            rollout,
            advantages,
            targets,
            config.num_minibatches,
            permutation_rng,
        )
        current_train_state, minibatch_metrics = jax.lax.scan(
            _update_minibatch,
            current_train_state,
            minibatches,
        )
        epoch_metrics = jax.tree.map(jnp.mean, minibatch_metrics)
        return (current_train_state, rng), epoch_metrics

    (train_state, rng), epoch_metrics = jax.lax.scan(
        _update_epoch,
        (train_state, rng),
        None,
        config.update_epochs,
    )
    metrics = jax.tree.map(jnp.mean, epoch_metrics)

    desired_kl = config.get("desired_kl", None)
    if desired_kl is not None and config.get("schedule", "fixed") == "adaptive":
        learning_rate = jax.lax.cond(
            metrics.approximate_kl > desired_kl * 2.0,
            lambda: jnp.maximum(
                jnp.asarray(1e-5, learning_rate.dtype),
                learning_rate / 1.5,
            ),
            lambda: jax.lax.cond(
                jnp.logical_and(
                    metrics.approximate_kl < desired_kl / 2.0,
                    metrics.approximate_kl > 0.0,
                ),
                lambda: jnp.minimum(
                    jnp.asarray(1e-2, learning_rate.dtype),
                    learning_rate * 1.5,
                ),
                lambda: learning_rate,
            ),
        )

    return train_state, rng, learning_rate, metrics


def global_moments(
    values: jax.Array,
    axis_name: str | None,
) -> tuple[jax.Array, jax.Array]:
    """Return mean and variance across local values and all replicas."""
    mean = jnp.mean(values)
    second_moment = jnp.mean(jnp.square(values))
    if axis_name is not None:
        mean = jax.lax.pmean(mean, axis_name=axis_name)
        second_moment = jax.lax.pmean(second_moment, axis_name=axis_name)
    variance = jnp.maximum(second_moment - jnp.square(mean), 0.0)
    return mean, variance
