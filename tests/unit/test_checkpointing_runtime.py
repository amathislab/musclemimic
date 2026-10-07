"""Tests for checkpoint runtime preparation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from musclemimic.algorithms.ppo import checkpoint as ppo_checkpoint
from musclemimic.runner import checkpointing


@dataclass(frozen=True)
class _FakeTrajectory:
    data: Any


def test_build_resume_runtime_target_uses_local_env_batch(monkeypatch):
    captured = {}
    config = OmegaConf.create(
        {
            "experiment": {
                "num_envs": 10240,
                "env_params": {"num_envs": 10240},
            }
        }
    )

    class FakeEnv:
        mjx_enabled = False
        th = None

    def instantiate_env(target_config, *, localize_num_envs):
        captured["localize_num_envs"] = localize_num_envs
        captured["target_num_envs"] = target_config.experiment.env_params.num_envs
        return FakeEnv()

    class Algorithm:
        @staticmethod
        def init_agent_conf(_env, _config):
            return SimpleNamespace()

        @staticmethod
        def build_runtime_state_restore_target(
            _rng,
            _env,
            _agent_conf,
            *,
            local_num_envs,
        ):
            captured["local_num_envs"] = local_num_envs
            return {"ok": True}

    monkeypatch.setattr(checkpointing, "device_count", lambda _config: 4)
    monkeypatch.setattr(checkpointing, "instantiate_env", instantiate_env)

    target = checkpointing.build_resume_runtime_state_target(config, Algorithm)

    assert target == {"ok": True}
    assert captured == {
        "localize_num_envs": False,
        "target_num_envs": 2560,
        "local_num_envs": 2560,
    }


def test_static_runtime_target_resets_to_local_motion_on_host(monkeypatch):
    reset_calls = []

    class FakeWrappedEnv:
        def reset(self, _keys):
            raise AssertionError("static restore target must not sample the global library")

        def reset_to(self, keys, trajectory_ids):
            reset_calls.append(np.asarray(trajectory_ids))
            num_envs = keys.shape[0]
            return (
                jnp.zeros((num_envs, 3), dtype=jnp.float32),
                {"state": jnp.arange(num_envs, dtype=jnp.float32)},
            )

    config = OmegaConf.create(
        {
            "experiment": {
                "num_envs": 4,
                "n_trajectories": 8,
                "lr": 3e-4,
                "normalize_env": False,
                "distributed": {"num_devices": 2},
                "trajectory": {"sharding": "static"},
                "env_params": {
                    "terminal_state_params": {},
                    "reward_params": {},
                },
                "adaptive_termination": {"enabled": False},
                "adaptive_sampling": {"enabled": False},
                "reward_curriculum": {"enabled": False},
            }
        }
    )
    agent_conf = SimpleNamespace(config=config)

    monkeypatch.setattr(ppo_checkpoint, "device_count", lambda _config: 2)
    monkeypatch.setattr(ppo_checkpoint, "wrap_env", lambda _env, _config: FakeWrappedEnv())
    for name in (
        "update_carry_threshold_unnormalized",
        "update_carry_reward_weights_unnormalized",
        "update_carry_weights_unnormalized",
        "update_carry_ema_unnormalized",
    ):
        monkeypatch.setattr(
            ppo_checkpoint,
            name,
            lambda state, *_args: state,
        )

    target = ppo_checkpoint.build_runtime_state_restore_target(
        jax.random.PRNGKey(0),
        env=object(),
        agent_conf=agent_conf,
    )

    assert len(reset_calls) == 1
    np.testing.assert_array_equal(reset_calls[0], np.zeros(2, dtype=np.int32))
    assert target["last_obs"].shape == (4, 3)
    assert isinstance(target["last_obs"], np.ndarray)
    assert np.shares_memory(target["last_obs"][0], target["last_obs"][1])


def test_runtime_batch_shape_distinguishes_single_device_observation_width_one():
    single_device = {"last_obs": np.zeros((8, 1), dtype=np.float32)}
    multi_device = {"last_obs": np.zeros((4, 2, 1), dtype=np.float32)}

    assert ppo_checkpoint._runtime_batch_shape(single_device, 8, 1) == (8,)
    assert ppo_checkpoint._runtime_batch_shape(multi_device, 8, 4) == (4, 2)


def test_static_resume_target_places_only_first_trajectory_shard(monkeypatch):
    conversion_calls = []

    def fail_full_conversion():
        raise AssertionError("the full trajectory must not be moved to a device")

    class FakeShard:
        def __init__(self, name):
            self.name = name

        def to_jax(self):
            conversion_calls.append(self.name)
            return f"jax-{self.name}"

    handler = SimpleNamespace(
        is_numpy=True,
        _is_numpy=True,
        n_trajectories=8,
        traj=_FakeTrajectory(data=np.arange(9)),
        to_jax=fail_full_conversion,
    )
    env = SimpleNamespace(mjx_enabled=True, th=handler)
    config = OmegaConf.create(
        {
            "experiment": {
                "trajectory": {"sharding": "static"},
            }
        }
    )

    monkeypatch.setattr(
        checkpointing.ShardSpec,
        "build",
        lambda n_trajectories, n_devices: (n_trajectories, n_devices),
    )

    def shard_data(data, spec, *, global_indexed_split_points):
        np.testing.assert_array_equal(data, np.arange(9))
        assert spec == (8, 2)
        assert global_indexed_split_points is True
        return [FakeShard("first"), FakeShard("second")], object()

    monkeypatch.setattr(checkpointing, "shard_trajectory_data", shard_data)

    checkpointing._prepare_runtime_target_trajectory(
        env,
        config,
        n_devices=2,
    )

    assert conversion_calls == ["first"]
    assert env.th.traj.data == "jax-first"
    assert env.th._is_numpy is False


def test_generate_run_suffix_includes_hydra_job(monkeypatch):
    monkeypatch.setenv("HYDRA_JOB_NUM", "7")

    suffix = checkpointing.generate_run_suffix()

    assert "-job7-" in suffix
    assert f"-pid{checkpointing.os.getpid()}" in suffix
