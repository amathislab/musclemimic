"""Tests for training and validation environment construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from musclemimic.runner import environment_factory


class FakeFactory:
    created_with = None

    @classmethod
    def make(cls, **kwargs):
        cls.created_with = kwargs
        return SimpleNamespace(mjx_enabled=False, th=None)


@pytest.fixture(autouse=True)
def reset_fake_factory(monkeypatch):
    FakeFactory.created_with = None
    monkeypatch.setattr(
        environment_factory.TaskFactory,
        "get_factory_cls",
        lambda _name: FakeFactory,
    )


def test_training_env_uses_local_num_envs_for_multi_device(monkeypatch):
    monkeypatch.setattr(environment_factory, "device_count", lambda _config: 4)
    config = OmegaConf.create(
        {
            "experiment": {
                "num_envs": 8192,
                "distributed": {"num_devices": 4},
                "trajectory": {"sharding": "none"},
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {"dataset": "train"},
                },
                "env_params": {
                    "env_name": "MjxMyoFullBody",
                    "num_envs": 8192,
                    "mjx_backend": "warp",
                },
            }
        }
    )

    environment_factory.instantiate_env(config)

    assert FakeFactory.created_with["num_envs"] == 2048
    assert FakeFactory.created_with["dataset"] == "train"
    assert config.experiment.num_envs == 8192
    assert config.experiment.env_params.num_envs == 8192


def test_training_env_keeps_trajectory_data_on_host(monkeypatch):
    class HostTrajectoryFactory:
        @classmethod
        def make(cls, **_kwargs):
            handler = SimpleNamespace(is_numpy=True)
            handler.to_jax = lambda: pytest.fail("trajectory conversion belongs to device placement")
            return SimpleNamespace(mjx_enabled=True, th=handler)

    monkeypatch.setattr(
        environment_factory.TaskFactory,
        "get_factory_cls",
        lambda _name: HostTrajectoryFactory,
    )
    config = OmegaConf.create(
        {
            "experiment": {
                "distributed": {"num_devices": 1},
                "trajectory": {"sharding": "none"},
                "task_factory": {"name": "ImitationFactory", "params": {}},
                "env_params": {"num_envs": 1},
            }
        }
    )

    env = environment_factory.instantiate_env(config)

    assert env.th.is_numpy is True


def test_training_env_rejects_non_divisible_device_batch(monkeypatch):
    monkeypatch.setattr(environment_factory, "device_count", lambda _config: 4)
    config = OmegaConf.create(
        {
            "experiment": {
                "num_envs": 8193,
                "env_params": {"num_envs": 8193},
                "task_factory": {"name": "ImitationFactory", "params": {}},
            }
        }
    )

    with pytest.raises(ValueError, match="must be divisible"):
        environment_factory.instantiate_env(config)


def test_validation_env_applies_validation_overrides():
    config = OmegaConf.create(
        {
            "experiment": {
                "trajectory": {"sharding": "none"},
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {"dataset": "validation"},
                },
                "env_params": {
                    "env_name": "MjxMyoFullBody",
                    "num_envs": 128,
                    "nconmax": 96,
                    "naconmax": 12288,
                    "njmax": 512,
                    "reward_params": {"qpos_w_sum": 0.1},
                    "reward_termination_type": "LowRewardTerminationHandler",
                    "reward_termination_params": {"threshold": 0.968, "patience": 100},
                    "terminal_state_type": "MeanRelativeSiteDeviationWithRootTerminalStateHandler",
                    "terminal_state_params": {"mean_site_deviation_threshold": 1.0},
                },
                "validation": {
                    "active": True,
                    "num_envs": 16,
                    "nconmax": 128,
                    "naconmax": 2048,
                    "njmax": 768,
                    "terminal_state_type": "NoTerminalStateHandler",
                    "terminal_state_params": {},
                    "reward_termination_type": "NoRewardTerminationHandler",
                    "reward_termination_params": {},
                },
            }
        }
    )

    environment_factory.instantiate_validation_env(config)

    assert FakeFactory.created_with["terminal_state_type"] == "NoTerminalStateHandler"
    assert FakeFactory.created_with["terminal_state_params"] == {}
    assert FakeFactory.created_with["num_envs"] == 16
    assert FakeFactory.created_with["nconmax"] == 128
    assert FakeFactory.created_with["naconmax"] == 2048
    assert FakeFactory.created_with["njmax"] == 768
    assert FakeFactory.created_with["reward_params"] == {"qpos_w_sum": 0.1}
    assert FakeFactory.created_with["reward_termination_type"] == "NoRewardTerminationHandler"
    assert FakeFactory.created_with["reward_termination_params"] == {}


def test_shared_validation_env_skips_dataset_loading():
    config = OmegaConf.create(
        {
            "experiment": {
                "trajectory": {"sharding": "static"},
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {
                        "amass_dataset_conf": {"rel_dataset_path": ["motion_a"]},
                        "other_param": 1,
                    },
                },
                "env_params": {"num_envs": 128},
                "validation": {"active": True},
            }
        }
    )

    environment_factory.instantiate_validation_env(config, share_trajectory=True)

    assert "amass_dataset_conf" not in FakeFactory.created_with
    assert FakeFactory.created_with["other_param"] == 1


def test_validation_dataset_override_replaces_training_dataset_library():
    validation_c3d_config = {"rel_dataset_path": ["capture_a"]}
    config = OmegaConf.create(
        {
            "experiment": {
                "trajectory": {"sharding": "none"},
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {
                        "amass_dataset_conf": {"rel_dataset_path": ["motion_a"]},
                        "other_param": 1,
                    },
                },
                "env_params": {"num_envs": 16},
                "validation": {
                    "active": True,
                    "c3d_dataset_conf": validation_c3d_config,
                },
            }
        }
    )

    environment_factory.instantiate_validation_env(config)

    assert "amass_dataset_conf" not in FakeFactory.created_with
    assert FakeFactory.created_with["c3d_dataset_conf"] == validation_c3d_config
    assert FakeFactory.created_with["other_param"] == 1
