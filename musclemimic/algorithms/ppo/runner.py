"""
PPO training loop (JAX scan-based).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.adaptive_sampling import (
    compute_adaptive_weights,
    compute_per_traj_termination_stats,
    compute_topk_weights,
)
from musclemimic.algorithms.common.checkpoint_utils import (
    TrainingConfig,
    compute_resume_state,
    optimizer_step_to_update,
    reset_lr_schedule_count,
)
from musclemimic.algorithms.common.curriculum import (
    compute_early_termination_stats,
    create_curriculum_params,
    create_curriculum_state,
    create_reward_curriculum_state,
    update_curriculum_state,
    update_reward_curriculum_state,
    validate_curriculum_config,
)
from musclemimic.algorithms.common.dataclasses import (
    MetricHandlerTransition,
    TrainState,
    Transition,
    ValidationCarry,
    ValidationData,
    ValidationDataFields,
)
from musclemimic.algorithms.common.env_state_utils import (
    get_carry_normalized,
    get_carry_unnormalized,
    unwrap_to_mjx,
    update_carry_ema_normalized,
    update_carry_ema_unnormalized,
    update_carry_reward_weights_normalized,
    update_carry_reward_weights_unnormalized,
    update_carry_threshold_normalized,
    update_carry_threshold_unnormalized,
    update_carry_weights_normalized,
    update_carry_weights_unnormalized,
)
from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.algorithms.common.optimizer import (
    linear_lr_schedule,
    warmup_cosine_lr_schedule,
)
from musclemimic.algorithms.ppo.loss import (
    approx_kl,
    normalize_advantages,
    ppo_actor_loss,
    ppo_value_loss,
)
from musclemimic.algorithms.ppo.moe import aggregate_moe_metrics, zero_moe_metrics
from musclemimic.core.wrappers import LogEnvState, SummaryMetrics
from musclemimic.rl_core import compute_gae, create_minibatches
from musclemimic.utils.debug_tools import (
    DebugFlags,
    maybe_debug_callback,
    maybe_profile_traj_batch,
    maybe_profile_val_batch,
    maybe_track_nacon,
)
from musclemimic.utils.metrics import (
    VALIDATION_STEP_METRIC_KEYS,
    ValidationSummary,
    flatten_validation_metrics,
)
from musclemimic.utils.model import count_actor_critic_params, log_network_architecture

if TYPE_CHECKING:
    from musclemimic.algorithms.ppo.config import PPOAgentConf, PPOAgentState
    from musclemimic.utils.metrics import MetricsHandler


def train(
    rng: jax.Array,
    env: Any,
    agent_conf: PPOAgentConf,
    agent_state_cls: type,
    agent_state: PPOAgentState | None = None,
    mh: MetricsHandler | None = None,
    online_logging_callback: Callable[[dict], None] | None = None,
    logging_interval: int = 10,
    resume_info: dict | None = None,
    val_env: Any | None = None,
    apply_resume_resets: bool = True,
) -> dict[str, Any]:
    """
    Run PPO training loop.

    Args:
        rng: random key
        env: training environment
        agent_conf: agent configuration
        agent_state_cls: class to construct agent state
        agent_state: optional existing state for resume
        mh: metrics handler for validation
        online_logging_callback: callback for logging metrics
        logging_interval: how often to log
        resume_info: checkpoint metadata for resume
        val_env: optional separate validation environment

    Returns:
        dict with agent_state, training_metrics, validation_metrics
    """
    config = agent_conf.config.experiment
    network = agent_conf.network
    tx = agent_conf.tx
    debug_flags = DebugFlags.from_config(config.debug)

    env = wrap_env(env, config)
    val_env = wrap_env(val_env, config) if val_env is not None else env

    base_lr = float(config.lr)
    use_adaptive_lr = config.get("schedule", "fixed") == "adaptive" and config.get("desired_kl", None) is not None

    # initialize or restore train state
    train_state = _init_train_state(rng, env, network, tx, agent_state, config, apply_resume_resets=apply_resume_resets)
    param_counts = count_actor_critic_params(train_state.params)
    print(f"Trainable parameters: {param_counts['total']:,} (actor: {param_counts['actor']:,}, critic: {param_counts['critic']:,}, shared: {param_counts['shared']:,})")

    # Print network architecture
    log_network_architecture(train_state.params)
    if online_logging_callback is not None:
        try:
            online_logging_callback(
                {
                    "_log_only": True,
                    "model/param_count": param_counts["total"],
                    "model/actor_param_count": param_counts["actor"],
                    "model/critic_param_count": param_counts["critic"],
                    "model/shared_param_count": param_counts["shared"],
                    "max_timestep": 0,
                    "jax_raw_timestep": 0,
                }
            )
        except Exception as e:
            print(f"warning: online logging callback failed: {e}")
    if agent_state is None:
        rng, _, _ = jax.random.split(rng, 3)

    adaptive_term_cfg = config.get("adaptive_termination", {})
    adaptive_term_enabled = adaptive_term_cfg.get("enabled", False)
    if adaptive_term_enabled:
        validate_curriculum_config(adaptive_term_cfg)

    term_params = None
    if hasattr(config, "env_params"):
        term_params = config.env_params.get("terminal_state_params", None)
    if term_params is None:
        term_params = config.get("terminal_state_params", None)
    if term_params is None:
        term_params = {}
    default_threshold = term_params.get("mean_site_deviation_threshold", 0.3)

    if adaptive_term_enabled:
        init_threshold = adaptive_term_cfg.get("init_threshold", default_threshold)
        curriculum_params = create_curriculum_params(adaptive_term_cfg)
        curriculum_state = create_curriculum_state(init_threshold, curriculum_params.init_ema_val)
    else:
        init_threshold = default_threshold
        curriculum_params = None
        curriculum_state = create_curriculum_state(init_threshold, adaptive_term_cfg.get("init_ema_val", 1.0))

    # Initialize reward curriculum (when disabled, use reward_params weights from config)
    reward_curriculum_cfg = config.get("reward_curriculum", {})
    reward_curriculum_enabled = reward_curriculum_cfg.get("enabled", False)
    reward_params = config.env_params.get("reward_params", {})
    qvel_w_sum_init = (
        reward_curriculum_cfg.get("qvel_w_sum_init", 0.2) if reward_curriculum_enabled
        else reward_params.get("qvel_w_sum", 0.2)
    )
    root_vel_w_sum_init = (
        reward_curriculum_cfg.get("root_vel_w_sum_init", 0.2) if reward_curriculum_enabled
        else reward_params.get("root_vel_w_sum", 0.2)
    )
    reward_curriculum_state = create_reward_curriculum_state(reward_curriculum_cfg).replace(
        qvel_w_sum=jnp.array(qvel_w_sum_init, dtype=jnp.float32),
        root_vel_w_sum=jnp.array(root_vel_w_sum_init, dtype=jnp.float32),
    )

    # Extract static params for JIT (reward curriculum)
    rc_alpha = float(reward_curriculum_cfg.get("ema_alpha", 0.2))
    rc_eta = float(reward_curriculum_cfg.get("eta", 0.015))
    rc_qvel_w_max = float(reward_curriculum_cfg.get("qvel_w_sum_max", 0.4))
    rc_root_vel_w_max = float(reward_curriculum_cfg.get("root_vel_w_sum_max", 0.4))
    rc_term_rate_threshold = float(reward_curriculum_cfg.get("term_rate_threshold", 0.08))
    rc_consecutive_k = int(reward_curriculum_cfg.get("consecutive_k", 5))

    # reset env
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.num_envs)
    obsv, env_state = env.reset(reset_rng)

    if config.normalize_env:
        env_state = update_carry_threshold_normalized(env_state, init_threshold)
        env_state = update_carry_reward_weights_normalized(
            env_state, reward_curriculum_state.qvel_w_sum, reward_curriculum_state.root_vel_w_sum
        )
    else:
        env_state = update_carry_threshold_unnormalized(env_state, init_threshold)
        env_state = update_carry_reward_weights_unnormalized(
            env_state, reward_curriculum_state.qvel_w_sum, reward_curriculum_state.root_vel_w_sum
        )

    # Initialize adaptive sampling weights and EMA if enabled
    adaptive_cfg = config.get("adaptive_sampling", {})
    if adaptive_cfg.get("enabled", False):
        n_traj = config.get("n_trajectories", None)
        if n_traj is None:
            raise ValueError("adaptive_sampling enabled but experiment.n_trajectories is missing")

        # Uniform initial weights
        uniform = jnp.ones(n_traj) / n_traj
        uniform_batched = jnp.broadcast_to(uniform, (config.num_envs, n_traj))

        # EMA pseudo-count prior (1D, then broadcast to 2D)
        ema_done_init = jnp.full((n_traj,), adaptive_cfg.get("ema_done_init", 10.0))
        ema_early_init = jnp.full((n_traj,), adaptive_cfg.get("ema_early_init", 5.0))
        ema_done_batched = jnp.broadcast_to(ema_done_init, (config.num_envs, n_traj))
        ema_early_batched = jnp.broadcast_to(ema_early_init, (config.num_envs, n_traj))

        if config.normalize_env:
            env_state = update_carry_weights_normalized(env_state, uniform_batched)
            env_state = update_carry_ema_normalized(env_state, ema_done_batched, ema_early_batched)
        else:
            env_state = update_carry_weights_unnormalized(env_state, uniform_batched)
            env_state = update_carry_ema_unnormalized(env_state, ema_done_batched, ema_early_batched)

    # compute resume state
    completed_updates, remaining_updates, base_global_ts0, base_opt_step0, target_global_ts = _compute_resume_info(
        config,
        agent_state,
        resume_info,
        train_state,
        preserve_checkpoint_target=not apply_resume_resets,
    )

    if remaining_updates == 0:
        return _handle_no_training(
            config,
            agent_state_cls,
            train_state,
            base_lr,
            base_global_ts0,
            completed_updates,
            online_logging_callback,
        )

    base_global_ts0_py = int(base_global_ts0)
    base_opt_step0_py = int(base_opt_step0)
    steps_per_update_py = int(config.num_steps) * int(config.num_envs)
    steps_per_update_opt_py = int(config.num_minibatches) * int(config.update_epochs)

    # main training loop
    def _update_step(runner_state, unused):
        train_state, env_state, last_obs, rng, lr, val_rng, curriculum_state, reward_curriculum_state = runner_state

        # Write current reward weights to env carry (baseline or curriculum-dynamic)
        # Note: Curriculum update happens at end of step after computing early_rate
        if config.normalize_env:
            env_state = update_carry_reward_weights_normalized(
                env_state, reward_curriculum_state.qvel_w_sum, reward_curriculum_state.root_vel_w_sum
            )
        else:
            env_state = update_carry_reward_weights_unnormalized(
                env_state, reward_curriculum_state.qvel_w_sum, reward_curriculum_state.root_vel_w_sum
            )

        # collect trajectories
        runner_state, traj_batch = _collect_trajectories(
            train_state,
            env_state,
            last_obs,
            rng,
            lr,
            val_rng,
            curriculum_state,
            network,
            env,
            config,
            debug_flags,
        )

        maybe_profile_traj_batch(traj_batch, debug_flags)

        (
            train_state,
            env_state,
            last_obs,
            rng,
            lr,
            val_rng,
            curriculum_state,
        ) = runner_state

        _early_count, early_rate = compute_early_termination_stats(
            traj_batch.metrics.done,
            traj_batch.absorbing,
        )
        done_count = jnp.sum(traj_batch.metrics.done.astype(jnp.float32))

        curriculum_threshold = curriculum_state.current_threshold
        curriculum_ema_rate = curriculum_state.ema_rate
        if adaptive_term_enabled:
            def _do_update(state_and_rate):
                state, rate = state_and_rate
                new_state, _, _ = update_curriculum_state(state, rate, curriculum_params)
                return new_state

            def _skip_update(state_and_rate):
                state, _ = state_and_rate
                return state

            curriculum_state = jax.lax.cond(
                done_count > 0,
                _do_update,
                _skip_update,
                (curriculum_state, early_rate),
            )
            curriculum_threshold = curriculum_state.current_threshold
            curriculum_ema_rate = curriculum_state.ema_rate

            if config.normalize_env:
                env_state = update_carry_threshold_normalized(env_state, curriculum_threshold)
            else:
                env_state = update_carry_threshold_unnormalized(env_state, curriculum_threshold)

        # === ADAPTIVE SAMPLING: Update weights at rollout boundary using count-EMA ===
        # config.adaptive_sampling.enabled is static (captured at trace time)
        adaptive_weights_1d = jnp.zeros((1,), dtype=jnp.float32)
        rate_hat_1d = jnp.zeros((1,), dtype=jnp.float32)
        if config.get("adaptive_sampling", {}).get("enabled", False):
            n_traj = config.n_trajectories
            adaptive_cfg = config.adaptive_sampling

            # 1. Get per-trajectory termination stats from current rollout
            _, done_counts, early_counts = compute_per_traj_termination_stats(
                traj_batch.info,
                traj_batch.metrics.done,
                traj_batch.absorbing,
                n_traj,
            )

            # 2. Read current EMA state from carry (canonical env0)
            if config.normalize_env:
                carry = get_carry_normalized(env_state)
            else:
                carry = get_carry_unnormalized(env_state)
            ema_done_1d = carry.ema_done_counts[0]    # (n_traj,)
            ema_early_1d = carry.ema_early_counts[0]  # (n_traj,)

            # 3. Compute new weights and update EMA (pure 1D)
            weights_1d, new_ema_done_1d, new_ema_early_1d, rate_hat_1d = compute_adaptive_weights(
                done_counts,
                early_counts,
                ema_done_1d,
                ema_early_1d,
                beta=adaptive_cfg.get("beta", 0.2),
                alpha=adaptive_cfg.get("alpha", 1.0),
                floor_mix=adaptive_cfg.get("floor_mix", 0.1),
                eps_div=adaptive_cfg.get("eps_div", 1e-6),
                eps_pow=adaptive_cfg.get("eps_pow", 1e-6),
            )

            # 4. Broadcast back to 2D and update carry
            weights_2d = jnp.broadcast_to(weights_1d, (config.num_envs, n_traj))
            ema_done_2d = jnp.broadcast_to(new_ema_done_1d, (config.num_envs, n_traj))
            ema_early_2d = jnp.broadcast_to(new_ema_early_1d, (config.num_envs, n_traj))

            if config.normalize_env:
                env_state = update_carry_weights_normalized(env_state, weights_2d)
                env_state = update_carry_ema_normalized(env_state, ema_done_2d, ema_early_2d)
            else:
                env_state = update_carry_weights_unnormalized(env_state, weights_2d)
                env_state = update_carry_ema_unnormalized(env_state, ema_done_2d, ema_early_2d)

            adaptive_weights_1d = weights_1d  # for logging
        # === END ADAPTIVE SAMPLING ===

        # compute advantages
        y, _ = network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            last_obs,
            mutable=["run_stats"],
        )
        _, last_val = y
        terminal_mask = jnp.logical_or(traj_batch.absorbing, traj_batch.traj_end)
        horizon_timeout = jnp.logical_and(traj_batch.done, jnp.logical_not(terminal_mask))
        bootstrap_mask = jnp.logical_not(jnp.logical_or(terminal_mask, horizon_timeout))

        advantages, targets = compute_gae(
            traj_batch.reward,
            traj_batch.value,
            traj_batch.done,
            traj_batch.absorbing,
            last_val,
            config.gamma,
            config.gae_lambda,
            bootstrap_mask=bootstrap_mask,
        )

        # update network
        train_state, rng, lr, ppo_losses = _update_network(
            train_state,
            traj_batch,
            advantages,
            targets,
            rng,
            lr,
            network,
            config,
            base_lr,
            use_adaptive_lr,
        )

        (
            ppo_total_loss,
            ppo_value_loss_val,
            ppo_actor_loss_val,
            ppo_entropy,
            ppo_kl,
            ppo_moe_loss,
            ppo_gate_entropy,
            ppo_expert_var,
            ppo_top2_usage,
            ppo_gate_w_mean,
            ppo_gate_w_std,
            ppo_ratio_mean,
            ppo_ratio_std,
            ppo_ratio_min,
            ppo_ratio_max,
            ppo_clipped_ratio_frac,
        ) = ppo_losses

        counter = optimizer_step_to_update(train_state.step, config.num_minibatches, config.update_epochs)
        counter_dtype = counter.dtype
        updates_since_resume = jnp.maximum(
            jnp.asarray(0, dtype=counter_dtype),
            (jnp.asarray(train_state.step, dtype=counter_dtype) - jnp.asarray(base_opt_step0_py, dtype=counter_dtype))
            // jnp.asarray(steps_per_update_opt_py, dtype=counter_dtype),
        )
        updates_done = jnp.asarray(completed_updates, dtype=counter_dtype) + updates_since_resume

        # compute training metrics
        metric = _compute_training_metrics(traj_batch, config)

        # validation
        validation_metrics, val_rng = _run_validation(
            train_state,
            val_rng,
            val_env,
            config,
            mh,
            counter,
        )

        # debug logging
        maybe_debug_callback(env_state, config, debug_flags)

        # checkpointing
        _handle_checkpointing(
            train_state,
            rng,
            lr,
            counter,
            updates_done,
            base_global_ts0_py,
            completed_updates,
            config,
            agent_conf,
            env,
            target_global_ts,
        )

        # Logging
        if online_logging_callback is not None:
            is_validation_update = (counter % config.validation_interval) == 0
            should_log = jnp.logical_or(
                (counter % logging_interval) == 0,
                is_validation_update,
            )

            policy_std_mean = jnp.mean(jnp.exp(train_state.params["log_std"]))

            # Current learning rate
            if config.anneal_lr:
                lr_type = config.get("lr_schedule_type", "linear")
                if lr_type == "warmup_cosine":
                    current_lr = warmup_cosine_lr_schedule(
                        train_state.step, config.num_minibatches, config.update_epochs,
                        base_lr, config.num_updates, config.get("warmup_steps", None),
                        config.get("min_lr_ratio", 0.0),
                    )
                else:
                    current_lr = linear_lr_schedule(
                        train_state.step, config.num_minibatches, config.update_epochs,
                        base_lr, config.num_updates,
                    )
            elif use_adaptive_lr:
                current_lr = lr
            else:
                current_lr = jnp.asarray(base_lr, dtype=jnp.float32)

            # Adaptive sampling stats
            adaptive_cfg = config.get("adaptive_sampling", {})
            if adaptive_cfg.get("enabled", False):
                topk_vals, topk_ids = compute_topk_weights(adaptive_weights_1d, k=3)
                weight_min, weight_max = jnp.min(adaptive_weights_1d), jnp.max(adaptive_weights_1d)
                rate_hat_mean = jnp.mean(rate_hat_1d)
            else:
                topk_vals = jnp.zeros((3,), dtype=jnp.float32)
                topk_ids = jnp.zeros((3,), dtype=jnp.int32)
                weight_min = weight_max = rate_hat_mean = jnp.asarray(0.0, dtype=jnp.float32)

            # Advantage and return statistics
            adv_mean, adv_std = jnp.mean(advantages), jnp.std(advantages)
            target_mean, target_std = jnp.mean(targets), jnp.std(targets)
            value_mean = jnp.mean(traj_batch.value)
            # Explained variance: how well value function explains returns
            target_var = jnp.var(targets)
            explained_var = jnp.where(
                target_var > 1e-8,
                1.0 - jnp.var(targets - traj_batch.value) / target_var,
                0.0,
            )

            ppo_m = {
                "actor_loss": ppo_actor_loss_val, "value_loss": ppo_value_loss_val,
                "total_loss": ppo_total_loss, "entropy": ppo_entropy, "kl": ppo_kl,
                "ratio_mean": ppo_ratio_mean, "ratio_std": ppo_ratio_std,
                "ratio_min": ppo_ratio_min, "ratio_max": ppo_ratio_max,
                "clipped_ratio_frac": ppo_clipped_ratio_frac,
                "moe_loss": ppo_moe_loss, "gate_entropy": ppo_gate_entropy,
                "expert_var": ppo_expert_var, "top2_usage": ppo_top2_usage,
                "gw_mean": ppo_gate_w_mean, "gw_std": ppo_gate_w_std,
                "adv_mean": adv_mean, "adv_std": adv_std,
                "target_mean": target_mean, "target_std": target_std,
                "value_mean": value_mean, "explained_var": explained_var,
            }

            # Get enabled_measures from config for validation metric filtering
            val_cfg = config.get("validation", {})
            enabled_measures = val_cfg.get("measures", None)
            enabled_quantities = val_cfg.get("quantities", None)

            def _do_log():
                def _cb(m, train_m, val_m, is_val, cur_params, cur_run_stats,
                        step_val, cur_lr, std_mean, topk_v, topk_i, w_min, w_max,
                        r_hat, thresh, ema, rate, rc_qvel, rc_root, rc_ema, rc_consec):
                    # Compute global_timestep using Python int to avoid int32 overflow
                    # Closure captures: base_global_ts0_py, base_opt_step0_py, steps_per_update_py, steps_per_update_opt_py
                    cur_step = int(step_val)
                    updates_since_resume = max(0, (cur_step - base_opt_step0_py) // steps_per_update_opt_py)
                    global_timestep = base_global_ts0_py + updates_since_resume * steps_per_update_py

                    # jax_raw_timestep: timesteps since resume
                    jax_raw_timestep = int(updates_since_resume * steps_per_update_py)
                    log = {
                        "mean_episode_return": float(train_m.mean_episode_return),
                        "mean_episode_length": float(train_m.mean_episode_length),
                        "max_timestep": global_timestep,
                        "jax_raw_timestep": jax_raw_timestep,
                        "learning_rate": float(cur_lr),
                        "ppo/policy_loss": float(m["actor_loss"]),
                        "ppo/value_loss": float(m["value_loss"]),
                        "ppo/total_loss": float(m["total_loss"]),
                        "ppo/entropy": float(m["entropy"]),
                        "ppo/kl_divergence": float(m["kl"]),
                        "ppo/policy_std": float(std_mean),
                        "ppo/ratio_mean": float(m["ratio_mean"]),
                        "ppo/ratio_std": float(m["ratio_std"]),
                        "ppo/ratio_min": float(m["ratio_min"]),
                        "ppo/ratio_max": float(m["ratio_max"]),
                        "ppo/clipped_ratio_frac": float(m["clipped_ratio_frac"]),
                        "ppo/explained_variance": float(m["explained_var"]),
                        "ppo/advantage_mean": float(m["adv_mean"]),
                        "ppo/advantage_std": float(m["adv_std"]),
                        "ppo/return_mean": float(m["target_mean"]),
                        "ppo/return_std": float(m["target_std"]),
                        "ppo/value_mean": float(m["value_mean"]),
                        "ppo/early_termination_count": float(train_m.early_termination_count),
                        "ppo/early_termination_rate": float(train_m.early_termination_rate),
                        "ppo/horizon_timeout_count": float(train_m.horizon_timeout_count),
                        "ppo/horizon_timeout_rate": float(train_m.horizon_timeout_rate),
                        "ppo/utd": float(config.get("effective_utd", config.update_epochs)),
                        "ppo/sample_reuse": float(config.get("sample_reuse", config.update_epochs)),
                        "ppo/unrolls_per_1m": float(config.get("unrolls_per_1m", 0.0)),
                        "ppo/grad_steps_per_1m": float(config.get("grad_steps_per_1m", 0.0)),
                        "reward/total": float(train_m.reward_total),
                        "reward/qpos": float(train_m.reward_qpos),
                        "reward/qvel": float(train_m.reward_qvel),
                        "reward/root_pos": float(train_m.reward_root_pos),
                        "reward/rpos": float(train_m.reward_rpos),
                        "reward/rquat": float(train_m.reward_rquat),
                        "reward/rvel_rot": float(train_m.reward_rvel_rot),
                        "reward/rvel_lin": float(train_m.reward_rvel_lin),
                        "reward/root_vel": float(train_m.reward_root_vel),
                        "reward/penalty": float(train_m.penalty_total),
                        "reward/penalty_activation_energy": float(train_m.penalty_activation_energy),
                        "err/root_xyz": float(train_m.err_root_xyz),
                        "err/root_yaw": float(train_m.err_root_yaw),
                        "err/joint_pos": float(train_m.err_joint_pos),
                        "err/joint_vel": float(train_m.err_joint_vel),
                        "err/site_abs": float(train_m.err_site_abs),
                        "err/rpos": float(train_m.err_rpos),
                    }
                    if adaptive_term_enabled:
                        log["curriculum/termination_threshold"] = float(thresh)
                        log["curriculum/ema_rate"] = float(ema)
                        log["curriculum/termination_rate"] = float(rate)
                    if reward_curriculum_enabled:
                        log["reward_curriculum/qvel_w_sum"] = float(rc_qvel)
                        log["reward_curriculum/root_vel_w_sum"] = float(rc_root)
                    if bool(config.get("use_moe", False)):
                        log["ppo/moe_loss"] = float(m["moe_loss"])
                        log["ppo/moe_gate_entropy"] = float(m["gate_entropy"])
                        log["ppo/moe_expert_utilization_var"] = float(m["expert_var"])
                        log["ppo/moe_top2_expert_usage"] = float(m["top2_usage"])
                    desired_kl = config.get("desired_kl", None)
                    if use_adaptive_lr and desired_kl is not None and float(desired_kl) > 0.0:
                        base_lr_val = float(base_lr)
                        log["ppo/kl_target"] = float(desired_kl)
                        log["ppo/kl_ratio"] = float(m["kl"]) / float(desired_kl)
                        log["ppo/lr_scale"] = float(cur_lr) / base_lr_val if base_lr_val > 0.0 else 0.0
                    if adaptive_cfg.get("enabled", False):
                        log["adaptive/weight_min"] = float(w_min)
                        log["adaptive/weight_max"] = float(w_max)
                        log["adaptive/rate_hat_mean"] = float(r_hat)
                        for i in range(3):
                            log[f"adaptive/topk_{i}_traj_id"] = int(topk_i[i])
                            log[f"adaptive/topk_{i}_weight"] = float(topk_v[i])
                    if config.get("reward_curriculum", {}).get("enabled", False):
                        log["reward_curriculum/ema_term_rate"] = float(rc_ema)
                        log["reward_curriculum/consecutive_below"] = int(rc_consec)
                    # Add validation metrics when it's a validation update
                    if bool(is_val):
                        log["has_validation_update"] = True
                        val_flat = flatten_validation_metrics(val_m, enabled_measures, enabled_quantities)
                        log.update(val_flat)
                        # Add train params for video recording
                        log["_train_params"] = {"params": cur_params, "run_stats": cur_run_stats}
                    try:
                        online_logging_callback(log)
                    except Exception as e:
                        print(f"warning: logging failed: {e}")

                return jax.debug.callback(
                    _cb, ppo_m, metric, validation_metrics, is_validation_update,
                    train_state.params, train_state.run_stats,
                    train_state.step, current_lr, policy_std_mean,
                    topk_vals, topk_ids, weight_min, weight_max, rate_hat_mean,
                    curriculum_threshold, curriculum_ema_rate, early_rate,
                    reward_curriculum_state.qvel_w_sum, reward_curriculum_state.root_vel_w_sum,
                    reward_curriculum_state.ema_term_rate, reward_curriculum_state.consecutive_below,
                )

            jax.lax.cond(should_log, _do_log, lambda: None)

        # Update reward curriculum with current termination rate
        if reward_curriculum_enabled:
            reward_curriculum_state = update_reward_curriculum_state(
                reward_curriculum_state,
                early_rate,
                rc_alpha,
                rc_eta,
                rc_qvel_w_max,
                rc_root_vel_w_max,
                rc_term_rate_threshold,
                rc_consecutive_k,
            )

        runner_state = (train_state, env_state, last_obs, rng, lr, val_rng, curriculum_state, reward_curriculum_state)
        return runner_state, (metric, validation_metrics)

    # run training
    rng, _rng = jax.random.split(rng)
    _, val_rng = jax.random.split(rng)

    try:
        lr_override = getattr(config, "resume_lr_override", None)
    except Exception:
        lr_override = None
    init_lr = float(lr_override) if lr_override is not None else base_lr

    runner_state = (
        train_state,
        env_state,
        obsv,
        _rng,
        jnp.asarray(init_lr, dtype=jnp.float32),
        val_rng,
        curriculum_state,
        reward_curriculum_state,
    )

    if hasattr(env, "mjx_backend") and env.mjx_backend == "warp":
        print("using warp backend...")

    runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, remaining_updates)

    # final checkpoint
    _save_final_checkpoint(
        runner_state,
        base_global_ts0_py,
        completed_updates,
        remaining_updates,
        agent_conf,
        env,
        target_global_ts,
    )

    return {
        "agent_state": agent_state_cls(train_state=runner_state[0]),
        "training_metrics": metrics[0],
        "validation_metrics": metrics[1],
    }


def _init_train_state(rng, env, network, tx, agent_state, config=None, apply_resume_resets: bool = True):
    """Initialize or restore train state."""
    if agent_state is not None:
        loaded_ts = agent_state.train_state
        try:
            fresh_opt_state = tx.init(loaded_ts.params)
            if jax.tree_util.tree_structure(fresh_opt_state) != jax.tree_util.tree_structure(loaded_ts.opt_state):
                opt_state = fresh_opt_state
            else:
                opt_state = loaded_ts.opt_state
        except Exception:
            print("warning: failed to restore optimizer state, re-initializing")
            opt_state = tx.init(loaded_ts.params)

        # Reset LR schedule counter if requested (keeps momentum/Adam stats)
        reset_lr = (
            apply_resume_resets
            and config is not None
            and config.get("reset_lr_schedule_on_resume", False)
        )
        if reset_lr:
            print("[resume] Resetting LR schedule counter to 0 (keeping optimizer momentum)")
            opt_state = reset_lr_schedule_count(opt_state)

        params = loaded_ts.params

        # Reset action std for continual training exploration
        reset_std = (
            config.get("reset_std_on_resume", None)
            if apply_resume_resets and config is not None
            else None
        )
        if reset_std is not None and "log_std" in params:
            params = {**params, "log_std": jnp.full_like(params["log_std"], jnp.log(reset_std))}
            print(f"[resume] Reset action std -> {reset_std}")

        if not apply_resume_resets and config is not None:
            requested_resets = []
            if config.get("reset_lr_schedule_on_resume", False):
                requested_resets.append("lr_schedule")
            if config.get("reset_std_on_resume", None) is not None:
                requested_resets.append("action_std")
            if requested_resets:
                skipped = ", ".join(requested_resets)
                print(f"[resume] Skipping one-shot resume resets for local auto-resume checkpoint: {skipped}")

        return TrainState(
            apply_fn=network.apply,
            params=params,
            tx=tx,
            opt_state=opt_state,
            step=0 if reset_lr else loaded_ts.step,
            run_stats=loaded_ts.run_stats,
        )
    else:
        rng, _rng1, _ = jax.random.split(rng, 3)
        init_x = jnp.zeros(env.info.observation_space.shape)
        network_params = network.init(_rng1, init_x)
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"],
            run_stats=network_params["run_stats"],
            tx=tx,
        )


def _resolve_target_global_timestep(
    config,
    resume_info,
    base_global_ts0,
    preserve_checkpoint_target=False,
):
    """Resolve the absolute target budget for the current resume path."""
    configured_total_timesteps = int(config.total_timesteps)
    if bool(config.get("total_timesteps_is_absolute", False)):
        return configured_total_timesteps

    stored_target = int(resume_info.get("target_global_timestep", 0) or 0)
    if preserve_checkpoint_target and stored_target > 0:
        return stored_target

    return configured_total_timesteps + int(base_global_ts0)


def _compute_resume_info(config, agent_state, resume_info, train_state, preserve_checkpoint_target=False):
    """Compute resume state information.

    Returns:
        (completed_updates, remaining_updates, base_global_ts0, base_opt_step,
         target_global_timestep)
    """
    total_updates = int(config.num_updates)
    completed_updates = 0
    remaining_updates = total_updates
    base_global_ts0 = 0
    target_global_ts = int(config.total_timesteps)

    if agent_state is not None and resume_info is not None:
        current_config = TrainingConfig.from_experiment_config(config)
        base_global_ts0 = int(resume_info["global_timestep"])
        target_global_ts = _resolve_target_global_timestep(
            config,
            resume_info,
            base_global_ts0,
            preserve_checkpoint_target=preserve_checkpoint_target,
        )

        completed_updates, remaining_updates, config_changed = compute_resume_state(
            checkpoint_metadata=resume_info,
            current_config=current_config,
            total_timesteps=target_global_ts,
        )

        print(f"\n{'=' * 60}")
        print("RESUMING FROM CHECKPOINT")
        print(f"{'=' * 60}")
        print(f"Completed updates:       {completed_updates}")
        print(f"Remaining updates:       {remaining_updates}")
        print(f"Global timestep:         {base_global_ts0:,}")
        print(f"Target global timestep:  {target_global_ts:,}")

        if config_changed:
            print("\nTraining config changed:")
            print(f"   Checkpoint: num_envs={resume_info['num_envs']}, num_steps={resume_info['num_steps']}")
            print(f"   Current:    num_envs={config.num_envs}, num_steps={config.num_steps}")
        print(f"{'=' * 60}\n")

    return completed_updates, remaining_updates, base_global_ts0, int(train_state.step), target_global_ts


def _handle_no_training(config, agent_state_cls, train_state, base_lr, base_global_ts0, completed_updates, callback):
    """Handle case where training is already complete."""
    if callback is not None:
        final_timestep = int(base_global_ts0 + completed_updates * int(config.num_steps) * int(config.num_envs))
        try:
            callback(
                {
                    "mean_episode_return": 0.0,
                    "mean_episode_length": 0.0,
                    "max_timestep": final_timestep,
                    "learning_rate": float(base_lr),
                }
            )
        except Exception as e:
            raise RuntimeError("online logging callback failed") from e
    return {
        "agent_state": agent_state_cls(train_state=train_state),
        "training_metrics": None,
        "validation_metrics": None,
    }


def _collect_trajectories(
    train_state, env_state, last_obs, rng, lr, val_rng, curriculum_state, network, env, config, debug_flags
):
    """Collect trajectories via jax.lax.scan."""
    traj_split_points = jnp.asarray(env.th.traj.data.split_points)

    def _env_step(runner_state, unused):
        train_state, env_state, last_obs, rng, lr, val_rng, curriculum_state = runner_state

        rng, _rng = jax.random.split(rng)
        y, updates = network.apply(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            last_obs,
            mutable=["run_stats"],
        )
        pi, value = y
        train_state = train_state.replace(run_stats=updates["run_stats"])
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        obsv, reward, absorbing, done, info, env_state, transition_state, _transition_obsv = env.step_with_transition(
            env_state, action
        )
        log_env_state = transition_state.find(LogEnvState)
        traj_state = transition_state.additional_carry.traj_state
        traj_len = traj_split_points[traj_state.traj_no + 1] - traj_split_points[traj_state.traj_no]
        traj_end = traj_state.subtraj_step_no >= (traj_len - 1)

        # Track max ncon for buffer sizing
        # (Warp uses data._impl.nacon for number of contacts)
        if debug_flags.track_nacon:
            mjx_state, _ = unwrap_to_mjx(log_env_state.env_state)
            maybe_track_nacon(mjx_state.data._impl, debug_flags)

        transition = Transition(
            done,
            absorbing,
            traj_end,
            action,
            value,
            reward,
            log_prob,
            last_obs,
            info,
            traj_state,
            log_env_state.metrics,
        )
        runner_state = (train_state, env_state, obsv, rng, lr, val_rng, curriculum_state)
        return runner_state, transition

    runner_state = (train_state, env_state, last_obs, rng, lr, val_rng, curriculum_state)
    return jax.lax.scan(_env_step, runner_state, None, config.num_steps)


def _update_network(train_state, traj_batch, advantages, targets, rng, lr, network, config, base_lr, use_adaptive_lr):
    """Run PPO update epochs."""

    def _update_epoch(update_state, unused):
        def _update_minibatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info

            def _loss_fn(params, traj_batch, gae, targets):
                use_moe = config.get("use_moe", False)

                if use_moe:
                    y, _ = network.apply(
                        {"params": params, "run_stats": train_state.run_stats},
                        traj_batch.obs,
                        return_metrics=True,
                        mutable=["run_stats"],
                    )
                    pi, value, moe_metrics = y
                    moe_loss, gate_ent, exp_var, top2, gw_mean, gw_std = aggregate_moe_metrics(moe_metrics)
                else:
                    y, _ = network.apply(
                        {"params": params, "run_stats": train_state.run_stats},
                        traj_batch.obs,
                        mutable=["run_stats"],
                    )
                    pi, value = y
                    moe_loss, gate_ent, exp_var, top2, gw_mean, gw_std = zero_moe_metrics()

                log_prob = pi.log_prob(traj_batch.action)
                entropy = pi.entropy().mean()

                clip_eps_vf = config.get("clip_eps_vf", config.clip_eps)
                value_loss = ppo_value_loss(value, traj_batch.value, targets, clip_eps_vf)
                normalized_gae = normalize_advantages(gae)
                actor_loss, ratio_stats = ppo_actor_loss(
                    log_prob, traj_batch.log_prob, normalized_gae, config.clip_eps,
                    return_ratio_stats=True
                )
                kl_mean = approx_kl(traj_batch.log_prob, log_prob)

                total_loss = actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy + moe_loss

                return total_loss, (
                    value_loss,
                    actor_loss,
                    entropy,
                    kl_mean,
                    moe_loss,
                    gate_ent,
                    exp_var,
                    top2,
                    gw_mean,
                    gw_std,
                    ratio_stats["ratio_mean"],
                    ratio_stats["ratio_std"],
                    ratio_stats["ratio_min"],
                    ratio_stats["ratio_max"],
                    ratio_stats["clipped_ratio_frac"],
                )

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            ((total_loss, aux), grads) = grad_fn(train_state.params, traj_batch, advantages, targets)

            if use_adaptive_lr:
                scale = lr / jnp.asarray(base_lr, dtype=jnp.float32)
                grads = jax.tree.map(lambda g: g * scale.astype(g.dtype), grads)

            train_state = train_state.apply_gradients(grads=grads)
            return train_state, (total_loss, *aux)

        train_state, traj_batch, advantages, targets, rng, lr = update_state
        rng, _rng = jax.random.split(rng)
        minibatches = create_minibatches(traj_batch, advantages, targets, config.num_minibatches, _rng)
        train_state, mb_losses = jax.lax.scan(_update_minibatch, train_state, minibatches)

        epoch_losses = tuple(jnp.mean(x) for x in mb_losses)
        update_state = (train_state, traj_batch, advantages, targets, rng, lr)
        return update_state, epoch_losses

    update_state = (train_state, traj_batch, advantages, targets, rng, lr)
    update_state, epoch_losses = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
    train_state = update_state[0]
    rng = update_state[-2]
    lr = update_state[-1]

    ppo_losses = tuple(jnp.mean(x) for x in epoch_losses)

    # adaptive lr adjustment
    desired_kl = config.get("desired_kl", None)
    schedule = config.get("schedule", "fixed")

    if desired_kl is not None and schedule == "adaptive":
        ppo_kl = ppo_losses[4]  # kl_mean is 5th element (index 4)
        lr = jax.lax.cond(
            ppo_kl > desired_kl * 2.0,
            lambda: jnp.maximum(jnp.asarray(1e-5, lr.dtype), lr / 1.5),
            lambda: jax.lax.cond(
                jnp.logical_and(ppo_kl < desired_kl / 2.0, ppo_kl > 0.0),
                lambda: jnp.minimum(jnp.asarray(1e-2, lr.dtype), lr * 1.5),
                lambda: lr,
            ),
        )

    return train_state, rng, lr, ppo_losses


def _compute_training_metrics(traj_batch, config):
    """Compute training metrics from trajectory batch."""
    logged_metrics = traj_batch.metrics
    done_f = logged_metrics.done.astype(jnp.float32)
    num_done = jnp.sum(done_f)
    early_termination_count, early_termination_rate = compute_early_termination_stats(
        logged_metrics.done,
        traj_batch.absorbing,
    )
    terminal_mask = jnp.logical_or(traj_batch.absorbing, traj_batch.traj_end)
    horizon_timeout = jnp.logical_and(traj_batch.done, jnp.logical_not(terminal_mask))
    horizon_timeout_count = jnp.sum(horizon_timeout.astype(jnp.float32))
    horizon_timeout_rate = jnp.where(num_done > 0, horizon_timeout_count / num_done, 0.0)

    # Extract reward sub-terms from info dict (mean over all steps)
    info = traj_batch.info
    reward_total = jnp.mean(info["reward_total"])
    reward_qpos = jnp.mean(info["reward_qpos"])
    reward_qvel = jnp.mean(info["reward_qvel"])
    reward_root_pos = jnp.mean(info["reward_root_pos"])
    reward_rpos = jnp.mean(info["reward_rpos"])
    reward_rquat = jnp.mean(info["reward_rquat"])
    reward_rvel_rot = jnp.mean(info["reward_rvel_rot"])
    reward_rvel_lin = jnp.mean(info["reward_rvel_lin"])
    reward_root_vel = jnp.mean(info["reward_root_vel"])
    penalty_total = jnp.mean(info["penalty_total"])
    penalty_activation_energy = jnp.mean(info["penalty_activation_energy"])
    # Diagnostic error metrics
    err_root_xyz = jnp.mean(info["err_root_xyz"])
    err_root_yaw = jnp.mean(info["err_root_yaw"])
    err_joint_pos = jnp.mean(info["err_joint_pos"])
    err_joint_vel = jnp.mean(info["err_joint_vel"])
    err_site_abs = jnp.mean(info["err_site_abs"])
    err_rpos = jnp.mean(info["err_rpos"])

    return SummaryMetrics(
        mean_episode_return=jnp.where(
            num_done > 0,
            jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / num_done,
            0.0,
        ),
        mean_episode_length=jnp.where(
            num_done > 0,
            jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / num_done,
            0.0,
        ),
        max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
        early_termination_count=early_termination_count,
        early_termination_rate=early_termination_rate,
        horizon_timeout_count=horizon_timeout_count,
        horizon_timeout_rate=horizon_timeout_rate,
        reward_total=reward_total,
        reward_qpos=reward_qpos,
        reward_qvel=reward_qvel,
        reward_root_pos=reward_root_pos,
        reward_rpos=reward_rpos,
        reward_rquat=reward_rquat,
        reward_rvel_rot=reward_rvel_rot,
        reward_rvel_lin=reward_rvel_lin,
        reward_root_vel=reward_root_vel,
        penalty_total=penalty_total,
        penalty_activation_energy=penalty_activation_energy,
        err_root_xyz=err_root_xyz,
        err_root_yaw=err_root_yaw,
        err_joint_pos=err_joint_pos,
        err_joint_vel=err_joint_vel,
        err_site_abs=err_site_abs,
        err_rpos=err_rpos,
    )


def _run_validation(train_state, val_rng, val_env, config, mh, counter):
    """Run validation if scheduled."""
    debug_flags = DebugFlags.from_config(config.get("debug", False))

    def _evaluation_step(val_rng_in):
        def _eval_env(runner_state, unused):
            train_state, env_state, last_obs, eval_rng = runner_state
            eval_rng, _rng = jax.random.split(eval_rng)
            y, updates = train_state.apply_fn(
                {"params": train_state.params, "run_stats": train_state.run_stats},
                last_obs,
                mutable=["run_stats"],
            )
            pi, _ = y
            train_state = train_state.replace(run_stats=updates["run_stats"])

            if config.validation.get("deterministic", False):
                action = pi.mode()
            else:
                action = pi.sample(seed=_rng)

            # Validation metrics must read the pre-reset transition, but the scan
            # carry must advance with the post-reset rollout state.
            obsv, _, _, _, _, env_state, transition_state, _ = val_env.step_with_transition(env_state, action)
            log_env_state = transition_state.find(LogEnvState)

            # Extract only fields needed by MetricsHandler, with site arrays pre-sliced to K
            mjx_state, _ = unwrap_to_mjx(log_env_state.env_state)
            data = mjx_state.data
            site_ids_global = mh.rel_site_ids  # Global model IDs, shape (K,)
            val_data = ValidationData(
                metrics=log_env_state.metrics,
                data=ValidationDataFields(
                    qpos=data.qpos,
                    qvel=data.qvel,
                    xpos=data.xpos,
                    xquat=data.xquat,
                    cvel=data.cvel,
                    subtree_com=data.subtree_com,
                    site_xpos=data.site_xpos[:, site_ids_global, :],
                    site_xmat=data.site_xmat[:, site_ids_global, :, :],
                ),
                additional_carry=ValidationCarry(traj_state=mjx_state.additional_carry.traj_state),
            )
            transition = MetricHandlerTransition(val_data=val_data)
            runner_state = (train_state, env_state, obsv, eval_rng)
            return runner_state, transition

        val_rng_out, val_reset_rng, eval_rng = jax.random.split(val_rng_in, 3)
        reset_rng = jax.random.split(val_reset_rng, config.validation.num_envs)
        obsv, env_state = val_env.reset(reset_rng)
        runner_state_eval = (train_state, env_state, obsv, eval_rng)

        _, traj_batch_eval = jax.lax.scan(_eval_env, runner_state_eval, None, config.validation.num_steps)

        # Pass local site indices (0..num_sites-1) since site arrays are pre-sliced
        num_sites = mh.rel_site_ids.shape[0]
        maybe_profile_val_batch(traj_batch_eval, num_sites, debug_flags)
        sim_site_idx_local = jnp.arange(num_sites, dtype=mh.rel_site_ids.dtype)
        validation_metrics = mh(traj_batch_eval.val_data, sim_site_idx=sim_site_idx_local)
        return validation_metrics, val_rng_out

    def _skip_evaluation(val_rng_in):
        return mh.get_zero_container(), val_rng_in

    if mh is None:
        validation_metrics = ValidationSummary()
        val_rng = jax.lax.cond(
            counter % config.validation_interval == 0,
            lambda v: jax.random.split(v, 3)[0],
            lambda v: v,
            val_rng,
        )
    else:
        validation_metrics, val_rng = jax.lax.cond(
            counter % config.validation_interval == 0,
            _evaluation_step,
            _skip_evaluation,
            val_rng,
        )

    return validation_metrics, val_rng


# ---------------------------------------------------------------------------
# evaluate_all helpers  (MJX GPU path)
# ---------------------------------------------------------------------------


def _tree_where_batch(completed, old_tree, new_tree):
    """Keep finished envs unchanged inside a batched pytree."""

    def _select(old_val, new_val):
        if hasattr(new_val, "shape") and new_val.shape:
            if completed.ndim > 0 and new_val.shape[0] == completed.shape[0]:
                d = completed
                while d.ndim < new_val.ndim:
                    d = d[..., None]
                return jnp.where(d, old_val, new_val)
        return new_val

    return jax.tree.map(_select, old_tree, new_tree)


@functools.partial(jax.jit, static_argnums=(0,))
def _reset_eval_all_batch_jitted(val_env, reset_keys, traj_indices):
    """Reset a trajectory batch inside one compiled graph."""
    return val_env.reset_to(reset_keys, traj_indices)


def _rollout_eval_all_batch(
    network, params, run_stats, val_env, obs, env_state, deterministic, horizon, eval_rng
):
    """Run one fixed-shape rollout batch for evaluate_all."""
    num_envs = obs.shape[0]
    completed_init = jnp.zeros(num_envs, dtype=bool)

    def _scan_body(carry, _unused):
        cur_obs, cur_env_state, completed, rng, rs = carry
        was_completed = completed
        rng, _rng = jax.random.split(rng)

        y, updates = network.apply(
            {"params": params, "run_stats": rs}, cur_obs, mutable=["run_stats"]
        )
        pi, _ = y
        rs = updates["run_stats"]

        if deterministic:
            action = pi.mode()
        else:
            action = pi.sample(seed=_rng)

        action = jnp.where(was_completed[:, None], 0.0, action)

        next_obs, reward, absorbing, done, info, next_env_state = val_env.step(cur_env_state, action)

        valid_mask = ~was_completed
        completed = was_completed | done
        next_env_state = _tree_where_batch(was_completed, cur_env_state, next_env_state)
        next_obs = jnp.where(was_completed[:, None], cur_obs, next_obs)
        reward = jnp.where(valid_mask, reward, 0.0)
        metric_info = {key: info[key] for key in VALIDATION_STEP_METRIC_KEYS if key in info}

        carry = (next_obs, next_env_state, completed, rng, rs)
        per_step = {
            "reward": reward,
            "done": done,
            "valid_mask": valid_mask,
            "absorbing": absorbing,
            "info": metric_info,
        }
        return carry, per_step

    init_carry = (obs, env_state, completed_init, eval_rng, run_stats)
    _, scan_out = jax.lax.scan(_scan_body, init_carry, None, horizon)
    return scan_out


def _make_rollout_eval_all_batch_fn(network, val_env, deterministic, horizon):
    """Build the reusable jitted rollout for evaluate_all."""

    @jax.jit
    def _rollout(params, run_stats, obs, env_state, eval_rng):
        return _rollout_eval_all_batch(
            network, params, run_stats, val_env, obs, env_state, deterministic, horizon, eval_rng
        )

    return _rollout


def _reduce_eval_all_batch(scan_out, traj_lengths, active_mask):
    """Summarize one rollout batch into per-trajectory metrics."""
    rewards = scan_out["reward"]
    dones = scan_out["done"]
    valid_mask = scan_out["valid_mask"]
    absorbing = scan_out["absorbing"]

    episode_returns = jnp.sum(rewards, axis=0)
    any_done = jnp.any(dones, axis=0)
    first_done_step = jnp.argmax(dones, axis=0) + 1
    episode_lengths = jnp.where(any_done, first_done_step, dones.shape[0])

    terminal_step_idx = jnp.clip(first_done_step - 1, 0, dones.shape[0] - 1)
    terminal_absorbing = absorbing[terminal_step_idx, jnp.arange(absorbing.shape[1])]
    early_terminated = any_done & terminal_absorbing & (episode_lengths < traj_lengths) & active_mask

    frame_coverage = episode_lengths / jnp.maximum(traj_lengths, 1)

    info = scan_out["info"]
    valid_count = jnp.sum(valid_mask, axis=0)
    step_metrics = {}
    for key in info:
        val = info[key]
        if hasattr(val, "shape") and val.ndim >= 2 and val.shape[1] == active_mask.shape[0]:
            masked_sum = jnp.sum(val * valid_mask, axis=0)
            per_env_mean = masked_sum / jnp.maximum(valid_count, 1)
            step_metrics[key] = per_env_mean

    return {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "early_terminated": early_terminated,
        "frame_coverage": frame_coverage,
        "step_metrics": step_metrics,
    }


def _run_validation_all(
    network,
    params,
    run_stats,
    traj_env,
    val_env,
    num_envs,
    deterministic,
    eval_seed: int = 0,
):
    """Evaluate every trajectory with batched reset and rollout on MJX."""
    n_traj = int(traj_env.th.n_trajectories)
    rng = jax.random.key(eval_seed)
    max_horizon = max(int(traj_env.th.len_trajectory(i)) for i in range(n_traj))

    rollout_fn = _make_rollout_eval_all_batch_fn(network, val_env, deterministic, max_horizon)

    all_returns, all_lengths, all_early, all_coverage, all_traj_lens = [], [], [], [], []
    all_step_metrics = {}
    total_valid_steps = 0

    for batch_start in range(0, n_traj, num_envs):
        batch_end = min(batch_start + num_envs, n_traj)
        batch_indices = list(range(batch_start, batch_end))
        active_count = len(batch_indices)

        batch_traj_lens = [int(traj_env.th.len_trajectory(traj_idx)) for traj_idx in batch_indices]
        while len(batch_indices) < num_envs:
            batch_indices.append(batch_indices[-1])
            batch_traj_lens.append(batch_traj_lens[-1])

        rng, batch_rng, eval_rng = jax.random.split(rng, 3)
        reset_keys = jax.random.split(batch_rng, num_envs)
        traj_indices = jnp.asarray(batch_indices, dtype=jnp.int32)
        obs, env_state = _reset_eval_all_batch_jitted(val_env, reset_keys, traj_indices)
        active_mask = jnp.arange(num_envs) < active_count
        traj_lens = jnp.asarray(batch_traj_lens, dtype=jnp.int32)

        scan_out = rollout_fn(params, run_stats, obs, env_state, eval_rng)
        batch_metrics = jax.device_get(_reduce_eval_all_batch(scan_out, traj_lens, active_mask))

        for local_idx, traj_idx in enumerate(batch_indices[:active_count]):
            ep_return = float(batch_metrics["episode_returns"][local_idx])
            ep_length = int(batch_metrics["episode_lengths"][local_idx])
            early = bool(batch_metrics["early_terminated"][local_idx])
            traj_len = int(traj_lens[local_idx])
            coverage = float(batch_metrics["frame_coverage"][local_idx])

            all_returns.append(ep_return)
            all_lengths.append(ep_length)
            all_early.append(early)
            all_coverage.append(coverage)
            all_traj_lens.append(traj_len)

            total_valid_steps += ep_length
            for key, per_env_mean in batch_metrics["step_metrics"].items():
                weighted = float(per_env_mean[local_idx]) * ep_length
                if key not in all_step_metrics:
                    all_step_metrics[key] = 0.0
                all_step_metrics[key] += weighted

            suffix = f"  EARLY at {ep_length}/{traj_len}" if early else ""
            print(
                f"  traj {traj_idx + 1}/{n_traj}: "
                f"len={ep_length}, return={ep_return:.4f}{suffix}",
                flush=True,
            )

    all_returns = jnp.array(all_returns)
    all_lengths = jnp.array(all_lengths)
    all_early = jnp.array(all_early)
    all_coverage = jnp.array(all_coverage)
    all_traj_lens = jnp.array(all_traj_lens)
    total_frame = float(jnp.sum(all_traj_lens))

    metrics = {
        "val_mean_episode_return": float(jnp.mean(all_returns)),
        "val_mean_episode_length": float(jnp.mean(all_lengths)),
        "val_early_termination_count": float(jnp.sum(all_early)),
        "val_early_termination_rate": float(jnp.mean(all_early)),
        "val_frame_coverage": float(jnp.sum(all_coverage * all_traj_lens)) / total_frame if total_frame > 0 else 0.0,
        "val_total_frame": total_frame,
    }

    if total_valid_steps > 0:
        for key, weighted_sum in all_step_metrics.items():
            metrics[f"val_{key}"] = weighted_sum / total_valid_steps

    print(
        f"Completed {n_traj} trajectories, "
        f"{int(metrics['val_early_termination_count'])} early terminations"
    )
    return metrics


def _handle_checkpointing(
    train_state,
    rng,
    lr,
    counter,
    updates_done,
    base_global_timestep,
    base_completed_updates,
    config,
    agent_conf,
    env,
    target_global_timestep=0,
):
    """Handle periodic checkpointing."""
    save_ckpt_enabled = bool(getattr(config, "save_checkpoints", False))
    ckpt_interval = int(getattr(config, "checkpoint_interval", 0) or 0)
    save_on_validation = bool(getattr(config, "save_checkpoints_on_validation", True))

    if not (save_ckpt_enabled and (ckpt_interval > 0 or save_on_validation)):
        return

    from musclemimic.algorithms.common.checkpoint_hooks import create_jax_checkpoint_host_callback
    from musclemimic.algorithms.ppo.ppo import PPOJax

    cache_key = (
        int(base_global_timestep),
        int(base_completed_updates),
        int(target_global_timestep),
        int(getattr(config, "num_steps", 1) or 1),
        int(getattr(config, "num_envs", 1) or 1),
    )
    cached = getattr(create_jax_checkpoint_host_callback, "__cached_instance__", None)
    if cached is None or cached[0] != cache_key:
        ckpt_cb, ckpt_mgr = create_jax_checkpoint_host_callback(
            PPOJax,
            agent_conf,
            agent_conf.config,
            env,
            base_global_timestep=int(base_global_timestep),
            base_completed_updates=int(base_completed_updates),
            target_global_timestep=int(target_global_timestep),
        )
        create_jax_checkpoint_host_callback.__cached_instance__ = (cache_key, ckpt_cb, ckpt_mgr)
    else:
        ckpt_cb, _ = cached[1:]

    def _do_checkpoint():
        return jax.experimental.io_callback(
            ckpt_cb,
            jnp.int32(0),
            train_state.params,
            train_state.run_stats,
            train_state.opt_state,
            jnp.asarray(train_state.step, dtype=jnp.int32),
            jnp.asarray(updates_done, dtype=jnp.int32),
            rng,
            lr,
        )

    if ckpt_interval > 0:
        should_ckpt = jnp.logical_and(counter > 0, (counter % ckpt_interval) == 0)
        jax.lax.cond(should_ckpt, _do_checkpoint, lambda: jnp.int32(0))

    if save_on_validation:
        is_val_step = (counter % config.validation_interval) == 0
        also_interval = jnp.logical_and(ckpt_interval > 0, (counter % ckpt_interval) == 0)
        should_ckpt_val = jnp.logical_and(is_val_step, jnp.logical_not(also_interval))
        jax.lax.cond(should_ckpt_val, _do_checkpoint, lambda: jnp.int32(0))


def _save_final_checkpoint(
    runner_state,
    base_global_timestep,
    base_completed_updates,
    remaining_updates,
    agent_conf,
    env,
    target_global_timestep=0,
):
    """Save final checkpoint at end of training."""
    from musclemimic.algorithms.common.checkpoint_hooks import create_jax_checkpoint_host_callback
    from musclemimic.algorithms.ppo.ppo import PPOJax

    updates_done = int(base_completed_updates + remaining_updates)

    cache_key = (
        int(base_global_timestep),
        int(base_completed_updates),
        int(target_global_timestep),
        int(getattr(agent_conf.config.experiment, "num_steps", 1) or 1),
        int(getattr(agent_conf.config.experiment, "num_envs", 1) or 1),
    )
    cached = getattr(create_jax_checkpoint_host_callback, "__cached_instance__", None)
    if cached is None or cached[0] != cache_key:
        ckpt_cb, ckpt_mgr = create_jax_checkpoint_host_callback(
            PPOJax,
            agent_conf,
            agent_conf.config,
            env,
            base_global_timestep=int(base_global_timestep),
            base_completed_updates=int(base_completed_updates),
            target_global_timestep=int(target_global_timestep),
        )
        create_jax_checkpoint_host_callback.__cached_instance__ = (cache_key, ckpt_cb, ckpt_mgr)
    else:
        ckpt_cb, _ = cached[1:]

    final_train_state = runner_state[0]
    # runner_state = (train_state, env_state, last_obs, rng, lr, val_rng, curriculum_state, reward_curriculum_state)
    jax.experimental.io_callback(
        ckpt_cb,
        jnp.int32(0),
        final_train_state.params,
        final_train_state.run_stats,
        final_train_state.opt_state,
        jnp.asarray(final_train_state.step, dtype=jnp.int32),
        jnp.asarray(updates_done, dtype=jnp.int32),
        runner_state[3],  # rng
        runner_state[4],  # lr
    )
