"""Stable host log schema for PPO and TERRA reward diagnostics."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from musclemimic.core.wrappers import SummaryMetrics
from musclemimic.runner.ppo_logging import format_ppo_log


def _scalar_namespace(names, value=0.0):
    return SimpleNamespace(**dict.fromkeys(names, value))


def test_ppo_log_includes_terra_reward_diagnostics():
    training_metrics = SummaryMetrics(
        reward_root_quat=1.0,
        activation_energy_raw=2.0,
        reward_core_upper_body=3.0,
        reward_terminal_quality_bonus=4.0,
        err_core_upper_body_relative=5.0,
    )
    ppo_metrics = _scalar_namespace(
        (
            "policy_loss",
            "value_loss",
            "total_loss",
            "entropy",
            "approximate_kl",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "clipped_ratio_fraction",
        )
    )
    diagnostics = _scalar_namespace(
        (
            "current_learning_rate",
            "policy_std",
            "explained_variance",
            "advantage_mean",
            "advantage_std",
            "target_mean",
            "target_std",
            "value_mean",
        )
    )
    diagnostics.numerics_invalid_transition_count = 7.0
    diagnostics.numerics_invalid_transition_rate = 0.125
    diagnostics.numerics_reason_counts = np.asarray([2.0, 3.0, 4.0, 5.0, 6.0])
    diagnostics.numerics_offending_trajectory_count = 2
    diagnostics.numerics_topk_trajectory_ids = np.asarray([41, 9, -1])
    diagnostics.numerics_topk_trajectory_counts = np.asarray([5.0, 2.0, 0.0])
    config = OmegaConf.create(
        {
            "update_epochs": 1,
            "adaptive_termination": {"enabled": False},
            "reward_curriculum": {"enabled": False},
            "adaptive_sampling": {"enabled": False},
            "use_moe": False,
        }
    )

    log = format_ppo_log(
        training_metrics,
        ppo_metrics,
        diagnostics,
        config,
        global_timestep=100,
        raw_timestep=100,
    )

    assert log["reward/root_quat"] == 1.0
    assert log["reward/activation_energy_raw"] == 2.0
    assert log["reward/core_upper_body"] == 3.0
    assert log["reward/terminal_quality_bonus"] == 4.0
    assert log["errors/core_upper_body_relative"] == 5.0
    assert log["numerics/nonfinite_transition_count"] == 7.0
    assert log["numerics/nonfinite_transition_rate"] == 0.125
    assert log["numerics/nonfinite_observation_count"] == 2.0
    assert log["numerics/nonfinite_qpos_count"] == 3.0
    assert log["numerics/nonfinite_qvel_count"] == 4.0
    assert log["numerics/nonfinite_reward_count"] == 5.0
    assert log["numerics/nonfinite_reward_info_count"] == 6.0
    assert log["numerics/offending_trajectory_count"] == 2
    assert log["numerics/topk_0_traj_id"] == 41
    assert log["numerics/topk_0_count"] == 5.0
    assert log["numerics/topk_2_traj_id"] == -1


def test_ppo_log_exposes_adaptive_sampler_invariants_and_top_difficulty():
    training_metrics = SummaryMetrics()
    ppo_metrics = _scalar_namespace(
        (
            "policy_loss",
            "value_loss",
            "total_loss",
            "entropy",
            "approximate_kl",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "clipped_ratio_fraction",
        )
    )
    diagnostics = _scalar_namespace(
        (
            "current_learning_rate",
            "policy_std",
            "explained_variance",
            "advantage_mean",
            "advantage_std",
            "target_mean",
            "target_std",
            "value_mean",
        )
    )
    diagnostics.adaptive_weight_min = 0.001
    diagnostics.adaptive_weight_max = 0.01
    diagnostics.adaptive_weight_sum = 1.0
    diagnostics.adaptive_uniform_weight = 1.0 / 300.0
    diagnostics.adaptive_floor_weight = 0.25 / 300.0
    diagnostics.adaptive_effective_sample_size = 245.0
    diagnostics.adaptive_observed_motion_count = 273
    diagnostics.adaptive_termination_rate_mean = 0.4
    diagnostics.adaptive_topk_ids = np.asarray([12, 34, 56])
    diagnostics.adaptive_topk_values = np.asarray([0.01, 0.009, 0.008])
    diagnostics.adaptive_topk_rates = np.asarray([0.9, 0.8, 0.7])
    diagnostics.adaptive_realized_reset_count = 8192
    diagnostics.adaptive_realized_l1_divergence = 0.03
    diagnostics.adaptive_realized_effective_sample_size = 240.0
    diagnostics.adaptive_desired_group_weights = np.asarray([1.0])
    diagnostics.adaptive_realized_group_frequencies = np.asarray([1.0])
    diagnostics.adaptive_desired_start_mode_weights = np.asarray([0.5, 0.25, 0.25])
    diagnostics.adaptive_realized_start_mode_frequencies = np.asarray([0.49, 0.26, 0.25])
    config = OmegaConf.create(
        {
            "update_epochs": 1,
            "adaptive_termination": {"enabled": False},
            "reward_curriculum": {"enabled": False},
            "adaptive_sampling": {"enabled": True},
            "use_moe": False,
        }
    )

    log = format_ppo_log(
        training_metrics,
        ppo_metrics,
        diagnostics,
        config,
        global_timestep=100,
        raw_timestep=100,
    )

    assert log["adaptive/weight_sum"] == 1.0
    assert log["adaptive/floor_weight"] == 0.25 / 300.0
    assert log["adaptive/observed_motion_count"] == 273
    assert log["adaptive/topk_0_traj_id"] == 12
    assert log["adaptive/topk_0_difficulty"] == 0.9
    assert log["adaptive/realized_reset_count"] == 8192
    assert log["adaptive/group/all/realized"] == 1.0
    assert log["adaptive/start/frame_zero/desired"] == 0.5
    assert log["adaptive/start/first_quarter/realized"] == 0.26
