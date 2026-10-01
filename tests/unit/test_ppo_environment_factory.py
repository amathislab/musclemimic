"""PPO environment batch and validation override contracts."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import pytest
from omegaconf import OmegaConf

from musclemimic.runner.ppo_environment_factory import (
    training_env_params,
    validation_env_params,
    validation_task_factory_params,
    validation_uses_training_trajectory,
)


@pytest.fixture(scope="module", autouse=True)
def require_four_devices():
    if jax.local_device_count() != 4:
        pytest.skip("requires the forced four-device CPU simulator")


def _config(*, train_envs=64, validation_envs=12):
    return OmegaConf.create(
        {
            "experiment": {
                "num_envs": train_envs,
                "distributed": {"num_devices": 4},
                "env_params": {
                    "num_envs": train_envs,
                    "domain_randomization_type": "TrainingRandomization",
                    "terminal_state_type": "TrainingTerminal",
                },
                "validation": {
                    "active": True,
                    "num_envs": validation_envs,
                    "terminal_state_type": "ValidationTerminal",
                    "terminal_state_params": {"threshold": 0.2},
                    "env_params": {
                        "domain_randomization_type": "NoDomainRandomization",
                    },
                },
            }
        }
    )


def test_training_num_envs_is_global_and_localized_per_device():
    assert training_env_params(_config())["num_envs"] == 16


def test_validation_num_envs_is_global_and_preserves_overrides():
    params = validation_env_params(_config())

    assert params["num_envs"] == 3
    assert params["domain_randomization_type"] == "NoDomainRandomization"
    assert params["terminal_state_type"] == "ValidationTerminal"
    assert params["terminal_state_params"] == {"threshold": 0.2}


@pytest.mark.parametrize(
    ("train_envs", "validation_envs", "message"),
    [(65, 12, "num_envs"), (64, 10, "validation.num_envs")],
)
def test_global_environment_counts_must_divide_by_devices(
    train_envs,
    validation_envs,
    message,
):
    config = _config(train_envs=train_envs, validation_envs=validation_envs)
    resolver = training_env_params if message == "num_envs" else validation_env_params
    with pytest.raises(ValueError, match=message.replace(".", r"\.")):
        resolver(config)


def _sharing_config(validation_motions):
    return OmegaConf.create(
        {
            "experiment": {
                "task_factory": {
                    "params": {
                        "trajectory_cache_root": "/cache",
                        "trajectory_cache_key": "train-key",
                        "amass_dataset_conf": {"rel_dataset_path": ["A", "B"]},
                    }
                },
                "validation": {
                    "active": True,
                    "share_trajectory_handler": True,
                    "trajectory_cache_root": "/cache",
                    "trajectory_cache_key": "validation-key",
                    "amass_dataset_conf": {"rel_dataset_path": validation_motions},
                },
            }
        }
    )


def test_validation_shares_only_an_identical_resolved_dataset():
    assert validation_uses_training_trajectory(_sharing_config(["A", "B"]))
    assert not validation_uses_training_trajectory(_sharing_config(["B", "A"]))


def test_validation_uses_its_role_specific_persistent_cache():
    params = validation_task_factory_params(_sharing_config(["B", "A"]))

    assert params["trajectory_cache_root"] == "/cache"
    assert params["trajectory_cache_key"] == "validation-key"
