"""Host-side formatting for PPO training metrics."""

from __future__ import annotations

from typing import Any

from musclemimic.algorithms.ppo.numerics import (
    NUMERICAL_REASON_INFO_KEYS,
    NUMERICAL_TOPK_TRAJECTORIES,
)
from musclemimic.utils.metrics import flatten_validation_metrics


def format_ppo_log(
    training_metrics: Any,
    ppo_metrics: Any,
    diagnostics: Any,
    config: Any,
    *,
    global_timestep: int,
    raw_timestep: int,
    validation_metrics: Any | None = None,
    checkpoint_update: int = -1,
) -> dict[str, Any]:
    """Convert one host-resident PPO update result to the stable log schema."""
    log = {
        "mean_episode_return": float(training_metrics.mean_episode_return),
        "mean_episode_length": float(training_metrics.mean_episode_length),
        "max_timestep": global_timestep,
        "jax_raw_timestep": raw_timestep,
        "learning_rate": float(diagnostics.current_learning_rate),
        "ppo/policy_loss": float(ppo_metrics.policy_loss),
        "ppo/value_loss": float(ppo_metrics.value_loss),
        "ppo/total_loss": float(ppo_metrics.total_loss),
        "ppo/entropy": float(ppo_metrics.entropy),
        "ppo/kl_divergence": float(ppo_metrics.approximate_kl),
        "ppo/policy_std": float(diagnostics.policy_std),
        "ppo/ratio_mean": float(ppo_metrics.ratio_mean),
        "ppo/ratio_std": float(ppo_metrics.ratio_std),
        "ppo/ratio_min": float(ppo_metrics.ratio_min),
        "ppo/ratio_max": float(ppo_metrics.ratio_max),
        "ppo/clipped_ratio_frac": float(ppo_metrics.clipped_ratio_fraction),
        "ppo/explained_variance": float(diagnostics.explained_variance),
        "ppo/advantage_mean": float(diagnostics.advantage_mean),
        "ppo/advantage_std": float(diagnostics.advantage_std),
        "ppo/return_mean": float(diagnostics.target_mean),
        "ppo/return_std": float(diagnostics.target_std),
        "ppo/value_mean": float(diagnostics.value_mean),
        "ppo/early_termination_count": float(training_metrics.early_termination_count),
        "ppo/early_termination_rate": float(training_metrics.early_termination_rate),
        "ppo/horizon_timeout_count": float(training_metrics.horizon_timeout_count),
        "ppo/horizon_timeout_rate": float(training_metrics.horizon_timeout_rate),
        "ppo/reward_termination_count": float(training_metrics.reward_termination_count),
        "ppo/reward_termination_rate": float(training_metrics.reward_termination_rate),
        "ppo/utd": float(config.get("effective_utd", config.update_epochs)),
        "ppo/sample_reuse": float(config.get("sample_reuse", config.update_epochs)),
        "ppo/unrolls_per_1m": float(config.get("unrolls_per_1m", 0.0)),
        "ppo/grad_steps_per_1m": float(config.get("grad_steps_per_1m", 0.0)),
        "reward/total": float(training_metrics.reward_total),
        "reward/qpos": float(training_metrics.reward_qpos),
        "reward/qvel": float(training_metrics.reward_qvel),
        "reward/root_pos": float(training_metrics.reward_root_pos),
        "reward/root_quat": float(training_metrics.reward_root_quat),
        "reward/rpos": float(training_metrics.reward_rpos),
        "reward/rquat": float(training_metrics.reward_rquat),
        "reward/rvel_rot": float(training_metrics.reward_rvel_rot),
        "reward/rvel_lin": float(training_metrics.reward_rvel_lin),
        "reward/root_vel": float(training_metrics.reward_root_vel),
        "reward/penalty": float(training_metrics.penalty_total),
        "reward/penalty_activation_energy": float(training_metrics.penalty_activation_energy),
        "reward/activation_energy_raw": float(training_metrics.activation_energy_raw),
        "reward/core_upper_body": float(training_metrics.reward_core_upper_body),
        "reward/terminal_quality_bonus": float(training_metrics.reward_terminal_quality_bonus),
        "err/root_xyz": float(training_metrics.err_root_xyz),
        "err/root_yaw": float(training_metrics.err_root_yaw),
        "err/joint_pos": float(training_metrics.err_joint_pos),
        "err/joint_vel": float(training_metrics.err_joint_vel),
        "err/site_abs": float(training_metrics.err_site_abs),
        "err/rpos": float(training_metrics.err_rpos),
        "errors/core_upper_body_relative": float(training_metrics.err_core_upper_body_relative),
    }

    reason_counts = getattr(
        diagnostics,
        "numerics_reason_counts",
        (0.0,) * len(NUMERICAL_REASON_INFO_KEYS),
    )
    topk_ids = getattr(
        diagnostics,
        "numerics_topk_trajectory_ids",
        (-1,) * NUMERICAL_TOPK_TRAJECTORIES,
    )
    topk_counts = getattr(
        diagnostics,
        "numerics_topk_trajectory_counts",
        (0.0,) * NUMERICAL_TOPK_TRAJECTORIES,
    )
    log.update(
        {
            "numerics/nonfinite_transition_count": float(
                getattr(diagnostics, "numerics_invalid_transition_count", 0.0)
            ),
            "numerics/nonfinite_transition_rate": float(
                getattr(diagnostics, "numerics_invalid_transition_rate", 0.0)
            ),
            "numerics/offending_trajectory_count": int(
                getattr(diagnostics, "numerics_offending_trajectory_count", 0)
            ),
        }
    )
    for index, (reason, _key) in enumerate(NUMERICAL_REASON_INFO_KEYS):
        log[f"numerics/nonfinite_{reason}_count"] = float(reason_counts[index])
    for index in range(NUMERICAL_TOPK_TRAJECTORIES):
        log[f"numerics/topk_{index}_traj_id"] = int(topk_ids[index])
        log[f"numerics/topk_{index}_count"] = float(topk_counts[index])

    if config.get("adaptive_termination", {}).get("enabled", False):
        log.update(
            {
                "curriculum/termination_threshold": float(diagnostics.curriculum_threshold),
                "curriculum/ema_rate": float(diagnostics.curriculum_ema_rate),
                "curriculum/termination_rate": float(diagnostics.early_termination_rate),
            }
        )
    if config.get("reward_curriculum", {}).get("enabled", False):
        log.update(
            {
                "reward_curriculum/qvel_w_sum": float(diagnostics.reward_qvel_weight),
                "reward_curriculum/root_vel_w_sum": float(diagnostics.reward_root_velocity_weight),
                "reward_curriculum/ema_term_rate": float(diagnostics.reward_termination_rate_ema),
                "reward_curriculum/consecutive_below": int(diagnostics.reward_consecutive_below),
            }
        )
    if bool(config.get("use_moe", False)):
        log.update(
            {
                "ppo/moe_loss": float(ppo_metrics.moe_loss),
                "ppo/moe_gate_entropy": float(ppo_metrics.gate_entropy),
                "ppo/moe_expert_utilization_var": float(ppo_metrics.expert_utilization_variance),
                "ppo/moe_top2_expert_usage": float(ppo_metrics.top2_expert_usage),
            }
        )

    desired_kl = config.get("desired_kl")
    adaptive_learning_rate = (
        config.get("schedule", "fixed") == "adaptive" and desired_kl is not None and float(desired_kl) > 0.0
    )
    if adaptive_learning_rate:
        base_learning_rate = float(config.lr)
        log.update(
            {
                "ppo/kl_target": float(desired_kl),
                "ppo/kl_ratio": float(ppo_metrics.approximate_kl) / float(desired_kl),
                "ppo/lr_scale": (
                    float(diagnostics.current_learning_rate) / base_learning_rate if base_learning_rate > 0.0 else 0.0
                ),
            }
        )

    if config.get("adaptive_sampling", {}).get("enabled", False):
        log.update(
            {
                "adaptive/weight_min": float(diagnostics.adaptive_weight_min),
                "adaptive/weight_max": float(diagnostics.adaptive_weight_max),
                "adaptive/weight_sum": float(diagnostics.adaptive_weight_sum),
                "adaptive/uniform_weight": float(diagnostics.adaptive_uniform_weight),
                "adaptive/floor_weight": float(diagnostics.adaptive_floor_weight),
                "adaptive/effective_sample_size": float(diagnostics.adaptive_effective_sample_size),
                "adaptive/observed_motion_count": int(diagnostics.adaptive_observed_motion_count),
                "adaptive/difficulty_mean": float(diagnostics.adaptive_termination_rate_mean),
                "adaptive/rate_hat_mean": float(diagnostics.adaptive_termination_rate_mean),
                "adaptive/realized_reset_count": float(diagnostics.adaptive_realized_reset_count),
                "adaptive/realized_l1_divergence": float(
                    diagnostics.adaptive_realized_l1_divergence
                ),
                "adaptive/realized_effective_sample_size": float(
                    diagnostics.adaptive_realized_effective_sample_size
                ),
            }
        )
        for index in range(3):
            log[f"adaptive/topk_{index}_traj_id"] = int(diagnostics.adaptive_topk_ids[index])
            log[f"adaptive/topk_{index}_weight"] = float(diagnostics.adaptive_topk_values[index])
            log[f"adaptive/topk_{index}_difficulty"] = float(diagnostics.adaptive_topk_rates[index])
        group_names = list(
            config.get("adaptive_sampling", {}).get("group_names", ["all"])
        )
        for index, name in enumerate(group_names):
            log[f"adaptive/group/{name}/desired"] = float(
                diagnostics.adaptive_desired_group_weights[index]
            )
            log[f"adaptive/group/{name}/realized"] = float(
                diagnostics.adaptive_realized_group_frequencies[index]
            )
        for index, name in enumerate(("frame_zero", "first_quarter", "full_clip")):
            log[f"adaptive/start/{name}/desired"] = float(
                diagnostics.adaptive_desired_start_mode_weights[index]
            )
            log[f"adaptive/start/{name}/realized"] = float(
                diagnostics.adaptive_realized_start_mode_frequencies[index]
            )

    if validation_metrics is not None:
        validation_config = config.get("validation", {})
        log["has_validation_update"] = True
        log["_checkpoint_update"] = int(checkpoint_update)
        log.update(
            flatten_validation_metrics(
                validation_metrics,
                validation_config.get("measures"),
                validation_config.get("quantities"),
            )
        )
    return log
