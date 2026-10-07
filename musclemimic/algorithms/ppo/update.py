"""One numerical PPO update without logging or checkpointing."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.adaptive_sampling import (
    compute_adaptive_weights,
    compute_per_traj_termination_stats,
    compute_topk_weights,
)
from musclemimic.algorithms.common.checkpoint_utils import optimizer_step_to_update
from musclemimic.algorithms.common.curriculum import (
    CurriculumParams,
    compute_early_termination_stats,
    update_curriculum_state,
    update_reward_curriculum_state,
)
from musclemimic.algorithms.common.distributed import (
    mark_data_varying,
    pmax,
    pmean,
    psum,
)
from musclemimic.algorithms.common.env_state_utils import (
    get_carry_normalized,
    get_carry_unnormalized,
    update_carry_ema_normalized,
    update_carry_ema_unnormalized,
    update_carry_reward_weights_normalized,
    update_carry_reward_weights_unnormalized,
    update_carry_threshold_normalized,
    update_carry_threshold_unnormalized,
    update_carry_weights_normalized,
    update_carry_weights_unnormalized,
)
from musclemimic.algorithms.common.optimizer import (
    linear_lr_schedule,
    warmup_cosine_lr_schedule,
)
from musclemimic.algorithms.common.trajectory_sharding import (
    ShardSpec,
    apply_shard_mask_and_renormalize,
)
from musclemimic.algorithms.ppo.learning import (
    PPOUpdateMetrics,
    global_moments,
    update,
    update_observation_stats,
)
from musclemimic.algorithms.ppo.rollout import collect_rollout
from musclemimic.algorithms.ppo.state import PPORuntimeState
from musclemimic.core.wrappers import SummaryMetrics
from musclemimic.rl_core import compute_gae
from musclemimic.utils.debug_tools import (
    DebugFlags,
    maybe_debug_callback,
    maybe_profile_traj_batch,
)


class PPOUpdateDiagnostics(NamedTuple):
    """Scalar diagnostics needed by host-side training logging."""

    current_learning_rate: jax.Array
    policy_std: jax.Array
    advantage_mean: jax.Array
    advantage_std: jax.Array
    target_mean: jax.Array
    target_std: jax.Array
    value_mean: jax.Array
    explained_variance: jax.Array
    adaptive_topk_values: jax.Array
    adaptive_topk_ids: jax.Array
    adaptive_weight_min: jax.Array
    adaptive_weight_max: jax.Array
    adaptive_termination_rate_mean: jax.Array
    early_termination_rate: jax.Array
    curriculum_threshold: jax.Array
    curriculum_ema_rate: jax.Array
    reward_qvel_weight: jax.Array
    reward_root_velocity_weight: jax.Array
    reward_termination_rate_ema: jax.Array
    reward_consecutive_below: jax.Array


class PPOUpdateResult(NamedTuple):
    """Output of one PPO rollout and optimizer update."""

    runtime_state: PPORuntimeState
    training_metrics: SummaryMetrics
    ppo_metrics: PPOUpdateMetrics
    diagnostics: PPOUpdateDiagnostics
    adaptive_sampling_weights: jax.Array | None
    counter: jax.Array


def run_ppo_update(
    runtime_state: PPORuntimeState,
    *,
    env: Any,
    network: Any,
    config: Any,
    debug_flags: DebugFlags,
    base_learning_rate: float,
    use_adaptive_learning_rate: bool,
    adaptive_termination_enabled: bool,
    curriculum_params: CurriculumParams | None,
    shard_spec: ShardSpec | None,
    axis_name: str | None,
) -> PPOUpdateResult:
    """Collect a rollout and apply one PPO update.

    All changing values enter and leave through ``runtime_state``. The outer
    JAX transform captures the environment, network, configuration, and shard
    description.
    """
    train_state = runtime_state.train_state
    env_state = runtime_state.env_state
    last_obs = runtime_state.last_obs
    rng = runtime_state.rng
    learning_rate = runtime_state.lr
    curriculum_state = runtime_state.curriculum_state
    reward_curriculum_state = runtime_state.reward_curriculum_state

    if config.normalize_env:
        env_state = update_carry_reward_weights_normalized(
            env_state,
            reward_curriculum_state.qvel_w_sum,
            reward_curriculum_state.root_vel_w_sum,
        )
    else:
        env_state = update_carry_reward_weights_unnormalized(
            env_state,
            reward_curriculum_state.qvel_w_sum,
            reward_curriculum_state.root_vel_w_sum,
        )
    env_state = mark_data_varying(env_state, axis_name)

    rollout_state, trajectory_batch = collect_rollout(
        train_state,
        env_state,
        last_obs,
        rng,
        network=network,
        env=env,
        num_steps=config.num_steps,
        debug_flags=debug_flags,
        axis_name=axis_name,
    )
    maybe_profile_traj_batch(trajectory_batch, debug_flags)
    train_state, env_state, last_obs, rng = rollout_state

    early_count, early_termination_rate = compute_early_termination_stats(
        trajectory_batch.metrics.done,
        trajectory_batch.absorbing,
    )
    done_count = jnp.sum(trajectory_batch.metrics.done.astype(jnp.float32))
    if axis_name is not None:
        done_count = jax.lax.psum(done_count, axis_name=axis_name)
        early_count = jax.lax.psum(early_count, axis_name=axis_name)
        early_termination_rate = early_count / jnp.maximum(done_count, 1.0)

    if adaptive_termination_enabled:
        if curriculum_params is None:
            raise ValueError("curriculum_params is required when adaptive termination is enabled")

        def _update_curriculum(state_and_rate):
            state, rate = state_and_rate
            updated_state, _, _ = update_curriculum_state(state, rate, curriculum_params)
            return updated_state

        curriculum_state = jax.lax.cond(
            done_count > 0,
            _update_curriculum,
            lambda state_and_rate: state_and_rate[0],
            (curriculum_state, early_termination_rate),
        )
        if config.normalize_env:
            env_state = update_carry_threshold_normalized(
                env_state,
                curriculum_state.current_threshold,
            )
        else:
            env_state = update_carry_threshold_unnormalized(
                env_state,
                curriculum_state.current_threshold,
            )

    adaptive_weights = jnp.zeros((1,), dtype=jnp.float32)
    adaptive_sampling_weights = None
    adaptive_termination_rates = jnp.zeros((1,), dtype=jnp.float32)
    adaptive_config = config.get("adaptive_sampling", {})
    if adaptive_config.get("enabled", False):
        num_trajectories = config.n_trajectories
        _, done_counts, early_counts = compute_per_traj_termination_stats(
            trajectory_batch.info,
            trajectory_batch.metrics.done,
            trajectory_batch.absorbing,
            num_trajectories,
        )
        if axis_name is not None:
            done_counts = jax.lax.psum(done_counts, axis_name=axis_name)
            early_counts = jax.lax.psum(early_counts, axis_name=axis_name)

        if config.normalize_env:
            carry = get_carry_normalized(env_state)
        else:
            carry = get_carry_unnormalized(env_state)
        ema_done = carry.ema_done_counts[0]
        ema_early = carry.ema_early_counts[0]

        adaptive_weights, new_ema_done, new_ema_early, adaptive_termination_rates = compute_adaptive_weights(
            done_counts,
            early_counts,
            ema_done,
            ema_early,
            beta=adaptive_config.get("beta", 0.2),
            alpha=adaptive_config.get("alpha", 1.0),
            floor_mix=adaptive_config.get("floor_mix", 0.1),
            eps_div=adaptive_config.get("eps_div", 1e-6),
            eps_pow=adaptive_config.get("eps_pow", 1e-6),
        )

        adaptive_sampling_weights = adaptive_weights
        if shard_spec is not None:
            if axis_name is None:
                raise ValueError("axis_name is required when a trajectory shard is configured")
            device_id = jax.lax.axis_index(axis_name)
            adaptive_sampling_weights = apply_shard_mask_and_renormalize(
                adaptive_weights,
                shard_spec,
                device_id,
            )

        weights = jnp.broadcast_to(
            adaptive_sampling_weights,
            (config.num_envs, num_trajectories),
        )
        ema_done = jnp.broadcast_to(new_ema_done, (config.num_envs, num_trajectories))
        ema_early = jnp.broadcast_to(new_ema_early, (config.num_envs, num_trajectories))
        if config.normalize_env:
            env_state = update_carry_weights_normalized(env_state, weights)
            env_state = update_carry_ema_normalized(env_state, ema_done, ema_early)
        else:
            env_state = update_carry_weights_unnormalized(env_state, weights)
            env_state = update_carry_ema_unnormalized(env_state, ema_done, ema_early)

    _, last_value = network.apply(
        {"params": train_state.params, "run_stats": train_state.run_stats},
        last_obs,
    )
    bootstrap_stop = trajectory_batch.done | trajectory_batch.absorbing | trajectory_batch.traj_end
    advantages, targets = compute_gae(
        trajectory_batch.reward,
        trajectory_batch.value,
        trajectory_batch.done,
        trajectory_batch.absorbing,
        last_value,
        config.gamma,
        config.gae_lambda,
        bootstrap_mask=jnp.logical_not(bootstrap_stop),
    )

    train_state, rng, learning_rate, metrics = update(
        train_state,
        trajectory_batch,
        advantages,
        targets,
        rng,
        learning_rate,
        network=network,
        config=config,
        base_learning_rate=base_learning_rate,
        use_adaptive_learning_rate=use_adaptive_learning_rate,
        axis_name=axis_name,
    )

    new_observations = jnp.concatenate(
        (trajectory_batch.obs[1:], last_obs[None]),
        axis=0,
    )
    train_state = update_observation_stats(
        train_state,
        new_observations,
        network=network,
        axis_name=axis_name,
    )

    reward_curriculum_config = config.get("reward_curriculum", {})
    if reward_curriculum_config.get("enabled", False):
        reward_curriculum_state = update_reward_curriculum_state(
            reward_curriculum_state,
            early_termination_rate,
            float(reward_curriculum_config.get("ema_alpha", 0.2)),
            float(reward_curriculum_config.get("eta", 0.015)),
            float(reward_curriculum_config.get("qvel_w_sum_max", 0.4)),
            float(reward_curriculum_config.get("root_vel_w_sum_max", 0.4)),
            float(reward_curriculum_config.get("term_rate_threshold", 0.08)),
            int(reward_curriculum_config.get("consecutive_k", 5)),
        )

    training_metrics = summarize_training_rollout(
        trajectory_batch,
        config,
        axis_name=axis_name,
    )
    policy_std = jnp.mean(jnp.exp(train_state.params["log_std"]))
    if config.anneal_lr:
        schedule_type = config.get("lr_schedule_type", "linear")
        if schedule_type == "warmup_cosine":
            current_learning_rate = warmup_cosine_lr_schedule(
                train_state.step,
                config.num_minibatches,
                config.update_epochs,
                base_learning_rate,
                config.num_updates,
                config.get("warmup_steps", None),
                config.get("min_lr_ratio", 0.0),
            )
        else:
            current_learning_rate = linear_lr_schedule(
                train_state.step,
                config.num_minibatches,
                config.update_epochs,
                base_learning_rate,
                config.num_updates,
            )
    elif use_adaptive_learning_rate:
        current_learning_rate = learning_rate
    else:
        current_learning_rate = jnp.asarray(base_learning_rate, dtype=jnp.float32)

    if adaptive_config.get("enabled", False):
        topk_values, topk_ids = compute_topk_weights(adaptive_weights, k=3)
        adaptive_weight_min = jnp.min(adaptive_weights)
        adaptive_weight_max = jnp.max(adaptive_weights)
        adaptive_termination_rate_mean = jnp.mean(adaptive_termination_rates)
    else:
        topk_values = jnp.zeros((3,), dtype=jnp.float32)
        topk_ids = jnp.zeros((3,), dtype=jnp.int32)
        adaptive_weight_min = jnp.asarray(0.0, dtype=jnp.float32)
        adaptive_weight_max = jnp.asarray(0.0, dtype=jnp.float32)
        adaptive_termination_rate_mean = jnp.asarray(0.0, dtype=jnp.float32)

    advantage_mean, advantage_variance = global_moments(advantages, axis_name)
    target_mean, target_variance = global_moments(targets, axis_name)
    value_mean, _ = global_moments(trajectory_batch.value, axis_name)
    _, residual_variance = global_moments(
        targets - trajectory_batch.value,
        axis_name,
    )
    explained_variance = jnp.where(
        target_variance > 1e-8,
        1.0 - residual_variance / target_variance,
        0.0,
    )

    updated_runtime_state = runtime_state.replace(
        train_state=train_state,
        env_state=env_state,
        last_obs=last_obs,
        rng=rng,
        lr=learning_rate,
        curriculum_state=curriculum_state,
        reward_curriculum_state=reward_curriculum_state,
    )
    maybe_debug_callback(env_state, config, debug_flags)
    return PPOUpdateResult(
        runtime_state=updated_runtime_state,
        training_metrics=training_metrics,
        ppo_metrics=metrics,
        diagnostics=PPOUpdateDiagnostics(
            current_learning_rate=current_learning_rate,
            policy_std=policy_std,
            advantage_mean=advantage_mean,
            advantage_std=jnp.sqrt(advantage_variance),
            target_mean=target_mean,
            target_std=jnp.sqrt(target_variance),
            value_mean=value_mean,
            explained_variance=explained_variance,
            adaptive_topk_values=topk_values,
            adaptive_topk_ids=topk_ids,
            adaptive_weight_min=adaptive_weight_min,
            adaptive_weight_max=adaptive_weight_max,
            adaptive_termination_rate_mean=adaptive_termination_rate_mean,
            early_termination_rate=early_termination_rate,
            curriculum_threshold=curriculum_state.current_threshold,
            curriculum_ema_rate=curriculum_state.ema_rate,
            reward_qvel_weight=reward_curriculum_state.qvel_w_sum,
            reward_root_velocity_weight=reward_curriculum_state.root_vel_w_sum,
            reward_termination_rate_ema=reward_curriculum_state.ema_term_rate,
            reward_consecutive_below=reward_curriculum_state.consecutive_below,
        ),
        adaptive_sampling_weights=adaptive_sampling_weights,
        counter=optimizer_step_to_update(
            train_state.step,
            config.num_minibatches,
            config.update_epochs,
        ),
    )


def summarize_training_rollout(
    trajectory_batch: Any,
    config: Any,
    *,
    axis_name: str | None = None,
) -> SummaryMetrics:
    """Compute scalar training metrics from one trajectory batch."""
    logged_metrics = trajectory_batch.metrics
    done = logged_metrics.done.astype(jnp.float32)
    num_done = jnp.sum(done)

    global_num_done = psum(num_done, axis_name)
    early_termination_count = psum(
        jnp.sum(jnp.logical_and(logged_metrics.done, trajectory_batch.absorbing).astype(jnp.float32)),
        axis_name,
    )
    early_termination_rate = early_termination_count / jnp.maximum(global_num_done, 1.0)
    episode_return_sum = psum(
        jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)),
        axis_name,
    )
    episode_length_sum = psum(
        jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)),
        axis_name,
    )
    horizon_timeout = (
        logged_metrics.done
        & trajectory_batch.info["done_horizon"]
        & jnp.logical_not(trajectory_batch.absorbing)
        & jnp.logical_not(trajectory_batch.traj_end)
    )
    horizon_timeout_count = psum(jnp.sum(horizon_timeout.astype(jnp.float32)), axis_name)
    horizon_timeout_rate = jnp.where(
        global_num_done > 0,
        horizon_timeout_count / global_num_done,
        0.0,
    )
    reward_termination = logged_metrics.done & trajectory_batch.info["done_reward"]
    reward_termination_count = psum(
        jnp.sum(reward_termination.astype(jnp.float32)),
        axis_name,
    )
    reward_termination_rate = jnp.where(
        global_num_done > 0,
        reward_termination_count / global_num_done,
        0.0,
    )

    info = trajectory_batch.info
    global_num_envs = int(getattr(config, "global_num_envs", config.num_envs))
    return SummaryMetrics(
        mean_episode_return=jnp.where(
            global_num_done > 0,
            episode_return_sum / global_num_done,
            0.0,
        ),
        mean_episode_length=jnp.where(
            global_num_done > 0,
            episode_length_sum / global_num_done,
            0.0,
        ),
        max_timestep=pmax(
            jnp.max(logged_metrics.timestep * global_num_envs),
            axis_name,
        ),
        early_termination_count=early_termination_count,
        early_termination_rate=early_termination_rate,
        horizon_timeout_count=horizon_timeout_count,
        horizon_timeout_rate=horizon_timeout_rate,
        reward_termination_count=reward_termination_count,
        reward_termination_rate=reward_termination_rate,
        reward_total=pmean(jnp.mean(info["reward_total"]), axis_name),
        reward_qpos=pmean(jnp.mean(info["reward_qpos"]), axis_name),
        reward_qvel=pmean(jnp.mean(info["reward_qvel"]), axis_name),
        reward_root_pos=pmean(jnp.mean(info["reward_root_pos"]), axis_name),
        reward_root_quat=pmean(jnp.mean(info["reward_root_quat"]), axis_name),
        reward_rpos=pmean(jnp.mean(info["reward_rpos"]), axis_name),
        reward_rquat=pmean(jnp.mean(info["reward_rquat"]), axis_name),
        reward_rvel_rot=pmean(jnp.mean(info["reward_rvel_rot"]), axis_name),
        reward_rvel_lin=pmean(jnp.mean(info["reward_rvel_lin"]), axis_name),
        reward_root_vel=pmean(jnp.mean(info["reward_root_vel"]), axis_name),
        penalty_total=pmean(jnp.mean(info["penalty_total"]), axis_name),
        penalty_activation_energy=pmean(
            jnp.mean(info["penalty_activation_energy"]),
            axis_name,
        ),
        err_root_xyz=pmean(jnp.mean(info["err_root_xyz"]), axis_name),
        err_root_quat=pmean(jnp.mean(info["err_root_quat"]), axis_name),
        err_root_yaw=pmean(jnp.mean(info["err_root_yaw"]), axis_name),
        err_joint_pos=pmean(jnp.mean(info["err_joint_pos"]), axis_name),
        err_joint_vel=pmean(jnp.mean(info["err_joint_vel"]), axis_name),
        err_site_abs=pmean(jnp.mean(info["err_site_abs"]), axis_name),
        err_rpos=pmean(jnp.mean(info["err_rpos"]), axis_name),
    )
