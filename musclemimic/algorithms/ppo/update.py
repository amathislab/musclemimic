"""One numerical PPO update without logging or checkpointing."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.adaptive_sampling import (
    compute_adaptive_weights,
    compute_per_traj_difficulty_stats,
    compute_topk_weights,
)
from musclemimic.algorithms.common.checkpoint_utils import optimizer_step_to_update
from musclemimic.algorithms.common.curriculum import (
    CurriculumParams,
    compute_early_termination_stats,
    update_curriculum_state,
    update_reward_curriculum_state,
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
    warmup_hold_cosine_lr_schedule,
)
from musclemimic.algorithms.common.ppo_trajectory_sharding import (
    ShardSpec,
    apply_shard_mask_and_renormalize,
)
from musclemimic.algorithms.ppo.learning import (
    PPOUpdateMetrics,
    global_moments,
    update,
    update_observation_stats,
)
from musclemimic.algorithms.ppo.numerics import (
    numerical_transition_mask,
    summarize_numerical_transitions,
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
    adaptive_topk_rates: jax.Array
    adaptive_weight_min: jax.Array
    adaptive_weight_max: jax.Array
    adaptive_weight_sum: jax.Array
    adaptive_uniform_weight: jax.Array
    adaptive_floor_weight: jax.Array
    adaptive_effective_sample_size: jax.Array
    adaptive_observed_motion_count: jax.Array
    adaptive_termination_rate_mean: jax.Array
    adaptive_realized_reset_count: jax.Array
    adaptive_realized_l1_divergence: jax.Array
    adaptive_realized_effective_sample_size: jax.Array
    adaptive_desired_group_weights: jax.Array
    adaptive_realized_group_frequencies: jax.Array
    adaptive_desired_start_mode_weights: jax.Array
    adaptive_realized_start_mode_frequencies: jax.Array
    early_termination_rate: jax.Array
    curriculum_threshold: jax.Array
    curriculum_ema_rate: jax.Array
    reward_qvel_weight: jax.Array
    reward_root_velocity_weight: jax.Array
    reward_termination_rate_ema: jax.Array
    reward_consecutive_below: jax.Array
    numerics_invalid_transition_count: jax.Array
    numerics_invalid_transition_rate: jax.Array
    numerics_reason_counts: jax.Array
    numerics_offending_trajectory_count: jax.Array
    numerics_topk_trajectory_ids: jax.Array
    numerics_topk_trajectory_counts: jax.Array


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

    rollout_state, trajectory_batch = collect_rollout(
        train_state,
        env_state,
        last_obs,
        rng,
        network=network,
        env=env,
        num_steps=config.num_steps,
        debug_flags=debug_flags,
    )
    maybe_profile_traj_batch(trajectory_batch, debug_flags)
    train_state, env_state, last_obs, rng = rollout_state
    nonfinite_transitions = numerical_transition_mask(
        trajectory_batch.info,
        trajectory_batch.reward,
    )
    numerical_diagnostics = summarize_numerical_transitions(
        trajectory_batch,
        config,
        axis_name=axis_name,
    )

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
    adaptive_outcome_counts = jnp.zeros((1,), dtype=jnp.float32)
    adaptive_config = config.get("adaptive_sampling", {})
    configured_group_ids = adaptive_config.get("group_ids", None)
    if configured_group_ids is None:
        adaptive_group_ids = None
        adaptive_num_groups = 1
    else:
        group_ids_tuple = tuple(int(value) for value in configured_group_ids)
        adaptive_num_groups = max(group_ids_tuple) + 1
        if set(group_ids_tuple) != set(range(adaptive_num_groups)):
            raise ValueError("adaptive sampling group ids must be contiguous from zero")
        adaptive_group_ids = jnp.asarray(group_ids_tuple, dtype=jnp.int32)
    adaptive_realized_reset_count = jnp.asarray(0.0, dtype=jnp.float32)
    adaptive_realized_l1_divergence = jnp.asarray(0.0, dtype=jnp.float32)
    adaptive_realized_effective_sample_size = jnp.asarray(0.0, dtype=jnp.float32)
    adaptive_desired_group_weights = jnp.zeros(
        (adaptive_num_groups,),
        dtype=jnp.float32,
    )
    adaptive_realized_group_frequencies = jnp.zeros_like(
        adaptive_desired_group_weights
    )
    adaptive_desired_start_mode_weights = jnp.zeros((3,), dtype=jnp.float32)
    adaptive_realized_start_mode_frequencies = jnp.zeros((3,), dtype=jnp.float32)
    if adaptive_config.get("enabled", False):
        num_trajectories = config.n_trajectories
        trajectory_split_points = jnp.asarray(env.th.traj.data.split_points)
        _, done_counts, difficulty_counts = compute_per_traj_difficulty_stats(
            trajectory_batch.traj_state,
            trajectory_batch.metrics.done,
            trajectory_batch.absorbing,
            trajectory_batch.traj_end,
            trajectory_split_points,
            num_trajectories,
            late_failure_floor=adaptive_config.get("late_failure_floor", 0.25),
            excluded_transitions=nonfinite_transitions,
        )
        if axis_name is not None:
            done_counts = jax.lax.psum(done_counts, axis_name=axis_name)
            difficulty_counts = jax.lax.psum(difficulty_counts, axis_name=axis_name)
        adaptive_outcome_counts = done_counts

        if config.normalize_env:
            carry = get_carry_normalized(env_state)
        else:
            carry = get_carry_unnormalized(env_state)
        ema_done = carry.ema_done_counts[0]
        ema_early = carry.ema_early_counts[0]
        sampling_weights_used = carry.sampling_weights[0]
        if axis_name is not None:
            sampling_weights_used = jax.lax.pmean(
                sampling_weights_used,
                axis_name=axis_name,
            )

        reset_trajectory_ids = trajectory_batch.info["reset_traj_no"].flatten().astype(jnp.int32)
        reset_events = trajectory_batch.metrics.done.flatten().astype(jnp.float32)
        realized_reset_counts = jax.ops.segment_sum(
            reset_events,
            reset_trajectory_ids,
            num_segments=num_trajectories,
        )
        reset_start_modes = trajectory_batch.info["reset_start_mode"].flatten().astype(jnp.int32)
        valid_start_modes = jnp.logical_and(reset_events > 0, reset_start_modes >= 0)
        realized_start_mode_counts = jax.ops.segment_sum(
            valid_start_modes.astype(jnp.float32),
            jnp.maximum(reset_start_modes, 0),
            num_segments=3,
        )
        if axis_name is not None:
            realized_reset_counts = jax.lax.psum(
                realized_reset_counts,
                axis_name=axis_name,
            )
            realized_start_mode_counts = jax.lax.psum(
                realized_start_mode_counts,
                axis_name=axis_name,
            )
        adaptive_realized_reset_count = jnp.sum(realized_reset_counts)
        realized_reset_frequencies = realized_reset_counts / jnp.maximum(
            adaptive_realized_reset_count,
            1.0,
        )
        adaptive_realized_l1_divergence = jnp.where(
            adaptive_realized_reset_count > 0,
            jnp.sum(jnp.abs(realized_reset_frequencies - sampling_weights_used)),
            0.0,
        )
        adaptive_realized_effective_sample_size = jnp.where(
            adaptive_realized_reset_count > 0,
            1.0 / jnp.maximum(jnp.sum(jnp.square(realized_reset_frequencies)), 1e-12),
            0.0,
        )

        group_ids_for_metrics = (
            jnp.zeros((num_trajectories,), dtype=jnp.int32)
            if adaptive_group_ids is None
            else adaptive_group_ids
        )
        adaptive_desired_group_weights = jax.ops.segment_sum(
            sampling_weights_used,
            group_ids_for_metrics,
            num_segments=adaptive_num_groups,
        )
        adaptive_realized_group_frequencies = jax.ops.segment_sum(
            realized_reset_frequencies,
            group_ids_for_metrics,
            num_segments=adaptive_num_groups,
        )
        frame_zero_probability = float(
            getattr(env.th, "frame_zero_reset_probability", 0.0)
        )
        first_quarter_probability = float(
            getattr(env.th, "first_quarter_reset_probability", 0.0)
        )
        adaptive_desired_start_mode_weights = jnp.asarray(
            (
                frame_zero_probability,
                first_quarter_probability,
                1.0 - frame_zero_probability - first_quarter_probability,
            ),
            dtype=jnp.float32,
        )
        realized_start_mode_total = jnp.sum(realized_start_mode_counts)
        adaptive_realized_start_mode_frequencies = (
            realized_start_mode_counts / jnp.maximum(realized_start_mode_total, 1.0)
        )

        adaptive_weights, new_ema_done, new_ema_early, adaptive_termination_rates = compute_adaptive_weights(
            done_counts,
            difficulty_counts,
            ema_done,
            ema_early,
            beta=adaptive_config.get("beta", 0.2),
            alpha=adaptive_config.get("alpha", 1.0),
            floor_mix=adaptive_config.get("floor_mix", 0.1),
            eps_div=adaptive_config.get("eps_div", 1e-6),
            eps_pow=adaptive_config.get("eps_pow", 1e-6),
            group_ids=adaptive_group_ids,
            num_groups=adaptive_num_groups,
            difficulty_mix=adaptive_config.get("difficulty_mix", None),
            max_weight_ratio=adaptive_config.get("max_weight_ratio", None),
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
        elif schedule_type == "warmup_hold_cosine":
            current_learning_rate = warmup_hold_cosine_lr_schedule(
                train_state.step,
                config.num_minibatches,
                config.update_epochs,
                base_learning_rate,
                config.num_updates,
                config.get("warmup_steps", None),
                config.get("hold_fraction", 0.5),
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
        topk_rates = adaptive_termination_rates[topk_ids]
        adaptive_weight_min = jnp.min(adaptive_weights)
        adaptive_weight_max = jnp.max(adaptive_weights)
        adaptive_weight_sum = jnp.sum(adaptive_weights)
        adaptive_uniform_weight = jnp.asarray(
            1.0 / adaptive_weights.shape[0],
            dtype=adaptive_weights.dtype,
        )
        difficulty_mix = adaptive_config.get(
            "difficulty_mix",
            1.0 - adaptive_config.get("floor_mix", 0.1),
        )
        if adaptive_group_ids is None:
            adaptive_floor_weight = (1.0 - difficulty_mix) * adaptive_uniform_weight
        else:
            group_counts = jax.ops.segment_sum(
                jnp.ones_like(adaptive_weights),
                adaptive_group_ids,
                num_segments=adaptive_num_groups,
            )
            balanced_uniform = 1.0 / (
                adaptive_num_groups * group_counts[adaptive_group_ids]
            )
            adaptive_floor_weight = jnp.min((1.0 - difficulty_mix) * balanced_uniform)
        adaptive_effective_sample_size = 1.0 / jnp.sum(jnp.square(adaptive_weights))
        adaptive_observed_motion_count = jnp.sum(adaptive_outcome_counts > 0)
        adaptive_termination_rate_mean = jnp.mean(adaptive_termination_rates)
    else:
        topk_values = jnp.zeros((3,), dtype=jnp.float32)
        topk_ids = jnp.zeros((3,), dtype=jnp.int32)
        topk_rates = jnp.zeros((3,), dtype=jnp.float32)
        adaptive_weight_min = jnp.asarray(0.0, dtype=jnp.float32)
        adaptive_weight_max = jnp.asarray(0.0, dtype=jnp.float32)
        adaptive_weight_sum = jnp.asarray(0.0, dtype=jnp.float32)
        adaptive_uniform_weight = jnp.asarray(0.0, dtype=jnp.float32)
        adaptive_floor_weight = jnp.asarray(0.0, dtype=jnp.float32)
        adaptive_effective_sample_size = jnp.asarray(0.0, dtype=jnp.float32)
        adaptive_observed_motion_count = jnp.asarray(0, dtype=jnp.int32)
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
            adaptive_topk_rates=topk_rates,
            adaptive_weight_min=adaptive_weight_min,
            adaptive_weight_max=adaptive_weight_max,
            adaptive_weight_sum=adaptive_weight_sum,
            adaptive_uniform_weight=adaptive_uniform_weight,
            adaptive_floor_weight=adaptive_floor_weight,
            adaptive_effective_sample_size=adaptive_effective_sample_size,
            adaptive_observed_motion_count=adaptive_observed_motion_count,
            adaptive_termination_rate_mean=adaptive_termination_rate_mean,
            adaptive_realized_reset_count=adaptive_realized_reset_count,
            adaptive_realized_l1_divergence=adaptive_realized_l1_divergence,
            adaptive_realized_effective_sample_size=adaptive_realized_effective_sample_size,
            adaptive_desired_group_weights=adaptive_desired_group_weights,
            adaptive_realized_group_frequencies=adaptive_realized_group_frequencies,
            adaptive_desired_start_mode_weights=adaptive_desired_start_mode_weights,
            adaptive_realized_start_mode_frequencies=adaptive_realized_start_mode_frequencies,
            early_termination_rate=early_termination_rate,
            curriculum_threshold=curriculum_state.current_threshold,
            curriculum_ema_rate=curriculum_state.ema_rate,
            reward_qvel_weight=reward_curriculum_state.qvel_w_sum,
            reward_root_velocity_weight=reward_curriculum_state.root_vel_w_sum,
            reward_termination_rate_ema=reward_curriculum_state.ema_term_rate,
            reward_consecutive_below=reward_curriculum_state.consecutive_below,
            numerics_invalid_transition_count=(
                numerical_diagnostics.invalid_transition_count
            ),
            numerics_invalid_transition_rate=(
                numerical_diagnostics.invalid_transition_rate
            ),
            numerics_reason_counts=numerical_diagnostics.reason_counts,
            numerics_offending_trajectory_count=(
                numerical_diagnostics.offending_trajectory_count
            ),
            numerics_topk_trajectory_ids=(
                numerical_diagnostics.topk_trajectory_ids
            ),
            numerics_topk_trajectory_counts=(
                numerical_diagnostics.topk_trajectory_counts
            ),
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

    def _sum(value):
        return jax.lax.psum(value, axis_name=axis_name) if axis_name is not None else value

    def _mean(value):
        return jax.lax.pmean(value, axis_name=axis_name) if axis_name is not None else value

    global_num_done = _sum(num_done)
    early_termination_count = _sum(
        jnp.sum(jnp.logical_and(logged_metrics.done, trajectory_batch.absorbing).astype(jnp.float32))
    )
    early_termination_rate = early_termination_count / jnp.maximum(global_num_done, 1.0)
    episode_return_sum = _sum(jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)))
    episode_length_sum = _sum(jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)))
    info = trajectory_batch.info
    done_horizon = info.get(
        "done_horizon",
        jnp.ones_like(logged_metrics.done, dtype=bool),
    )
    horizon_timeout = (
        logged_metrics.done
        & done_horizon
        & jnp.logical_not(trajectory_batch.absorbing)
        & jnp.logical_not(trajectory_batch.traj_end)
    )
    horizon_timeout_count = _sum(jnp.sum(horizon_timeout.astype(jnp.float32)))
    horizon_timeout_rate = jnp.where(
        global_num_done > 0,
        horizon_timeout_count / global_num_done,
        0.0,
    )
    done_reward = info.get(
        "done_reward",
        jnp.zeros_like(logged_metrics.done, dtype=bool),
    )
    reward_termination = logged_metrics.done & done_reward
    reward_termination_count = _sum(jnp.sum(reward_termination.astype(jnp.float32)))
    reward_termination_rate = jnp.where(
        global_num_done > 0,
        reward_termination_count / global_num_done,
        0.0,
    )

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
        max_timestep=jnp.max(logged_metrics.timestep * global_num_envs),
        early_termination_count=early_termination_count,
        early_termination_rate=early_termination_rate,
        horizon_timeout_count=horizon_timeout_count,
        horizon_timeout_rate=horizon_timeout_rate,
        reward_termination_count=reward_termination_count,
        reward_termination_rate=reward_termination_rate,
        reward_total=_mean(jnp.mean(info["reward_total"])),
        reward_qpos=_mean(jnp.mean(info["reward_qpos"])),
        reward_qvel=_mean(jnp.mean(info["reward_qvel"])),
        reward_root_pos=_mean(jnp.mean(info["reward_root_pos"])),
        reward_root_quat=_mean(
            jnp.mean(info.get("reward_root_quat", jnp.zeros_like(info["reward_root_pos"])))
        ),
        reward_rpos=_mean(jnp.mean(info["reward_rpos"])),
        reward_rquat=_mean(jnp.mean(info["reward_rquat"])),
        reward_rvel_rot=_mean(jnp.mean(info["reward_rvel_rot"])),
        reward_rvel_lin=_mean(jnp.mean(info["reward_rvel_lin"])),
        reward_root_vel=_mean(jnp.mean(info["reward_root_vel"])),
        penalty_total=_mean(jnp.mean(info["penalty_total"])),
        penalty_activation_energy=_mean(jnp.mean(info["penalty_activation_energy"])),
        activation_energy_raw=_mean(
            jnp.mean(info.get("activation_energy_raw", jnp.zeros_like(info["reward_total"])))
        ),
        reward_core_upper_body=_mean(
            jnp.mean(info.get("reward_core_upper_body", jnp.zeros_like(info["reward_total"])))
        ),
        reward_terminal_quality_bonus=_mean(
            jnp.mean(info.get("reward_terminal_quality_bonus", jnp.zeros_like(info["reward_total"])))
        ),
        err_root_xyz=_mean(jnp.mean(info["err_root_xyz"])),
        err_root_yaw=_mean(jnp.mean(info["err_root_yaw"])),
        err_joint_pos=_mean(jnp.mean(info["err_joint_pos"])),
        err_joint_vel=_mean(jnp.mean(info["err_joint_vel"])),
        err_site_abs=_mean(jnp.mean(info["err_site_abs"])),
        err_rpos=_mean(jnp.mean(info["err_rpos"])),
        err_core_upper_body_relative=_mean(
            jnp.mean(info.get("err_core_upper_body_relative", jnp.zeros_like(info["err_rpos"])))
        ),
    )
