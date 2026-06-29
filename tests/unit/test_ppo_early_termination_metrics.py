from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from musclemimic.algorithms.ppo import runner as ppo_runner
from musclemimic.core.wrappers.mjx import Metrics


def _make_metrics(done_mask: jnp.ndarray) -> Metrics:
    shape = done_mask.shape
    zeros_f = jnp.zeros(shape, dtype=jnp.float32)
    zeros_i = jnp.zeros(shape, dtype=jnp.int32)
    return Metrics(
        episode_returns=zeros_f,
        episode_lengths=zeros_i,
        returned_episode_returns=zeros_f,
        returned_episode_lengths=zeros_i,
        timestep=zeros_i,
        done=done_mask,
        absorbing=jnp.zeros(shape, dtype=bool),
    )


def _make_info(shape: tuple) -> dict:
    """Create mock info dict with all required reward sub-term fields."""
    zeros_f = jnp.zeros(shape, dtype=jnp.float32)
    return {
        "reward_total": zeros_f,
        "reward_qpos": zeros_f,
        "reward_qvel": zeros_f,
        "reward_root_pos": zeros_f,
        "reward_rpos": zeros_f,
        "reward_rquat": zeros_f,
        "reward_rvel_rot": zeros_f,
        "reward_rvel_lin": zeros_f,
        "reward_root_vel": zeros_f,
        "penalty_total": zeros_f,
        "penalty_activation_energy": zeros_f,
        "err_root_xyz": zeros_f,
        "err_root_yaw": zeros_f,
        "err_joint_pos": zeros_f,
        "err_joint_vel": zeros_f,
        "err_site_abs": zeros_f,
        "err_rpos": zeros_f,
    }


def test_compute_training_metrics_counts_early_termination():
    done = jnp.array([[True, False], [False, True]], dtype=bool)
    absorbing = jnp.array([[True, False], [False, False]], dtype=bool)
    traj_end = jnp.array([[False, False], [False, True]], dtype=bool)
    info = _make_info(done.shape)
    traj_batch = SimpleNamespace(metrics=_make_metrics(done), done=done, absorbing=absorbing, traj_end=traj_end, info=info)
    config = SimpleNamespace(num_envs=2)

    summary = ppo_runner._compute_training_metrics(traj_batch, config)

    assert float(summary.early_termination_count) == 1.0
    assert float(summary.early_termination_rate) == 0.5
    assert float(summary.horizon_timeout_count) == 0.0
    assert float(summary.horizon_timeout_rate) == 0.0


def test_compute_training_metrics_handles_no_done():
    done = jnp.zeros((2, 2), dtype=bool)
    absorbing = jnp.ones((2, 2), dtype=bool)
    traj_end = jnp.zeros((2, 2), dtype=bool)
    info = _make_info(done.shape)
    traj_batch = SimpleNamespace(metrics=_make_metrics(done), done=done, absorbing=absorbing, traj_end=traj_end, info=info)
    config = SimpleNamespace(num_envs=2)

    summary = ppo_runner._compute_training_metrics(traj_batch, config)

    assert float(summary.early_termination_count) == 0.0
    assert float(summary.early_termination_rate) == 0.0
    assert float(summary.horizon_timeout_count) == 0.0
    assert float(summary.horizon_timeout_rate) == 0.0


def test_compute_training_metrics_counts_horizon_timeout():
    done = jnp.array([[True, True], [False, True]], dtype=bool)
    absorbing = jnp.array([[True, False], [False, False]], dtype=bool)
    traj_end = jnp.array([[False, False], [False, True]], dtype=bool)
    info = _make_info(done.shape)
    traj_batch = SimpleNamespace(metrics=_make_metrics(done), done=done, absorbing=absorbing, traj_end=traj_end, info=info)
    config = SimpleNamespace(num_envs=2)

    summary = ppo_runner._compute_training_metrics(traj_batch, config)

    assert float(summary.horizon_timeout_count) == 1.0
    assert float(summary.horizon_timeout_rate) == pytest.approx(1.0 / 3.0)
