"""Exact PPO resume topology validation tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.common.checkpoint_manager import CheckpointMetadata
from musclemimic.algorithms.ppo import checkpoint


def _metadata(**overrides):
    values = {
        "step": 4,
        "update_number": 1,
        "global_timestep": 32,
        "target_global_timestep": 64,
        "learning_rate": 3e-4,
        "num_envs": 8,
        "num_steps": 4,
        "num_minibatches": 2,
        "update_epochs": 2,
        "num_devices": 2,
        "trajectory_sharding": "none",
    }
    values.update(overrides)
    return CheckpointMetadata(**values)


def _agent_conf(*, sharding="none"):
    return SimpleNamespace(
        config=OmegaConf.create(
            {
                "experiment": {
                    "num_envs": 8,
                    "distributed": {"num_devices": 2},
                    "trajectory": {"sharding": sharding},
                }
            }
        )
    )


def _runtime_state():
    return {"last_obs": np.zeros((2, 4, 3), dtype=np.float32)}


def test_exact_resume_accepts_matching_replica_and_environment_shape(monkeypatch):
    monkeypatch.setattr(checkpoint, "device_count", lambda _config: 2)

    batch_shape = checkpoint._validate_exact_resume_topology(
        _metadata(),
        _runtime_state(),
        _agent_conf(),
    )

    assert batch_shape == (2, 4)


@pytest.mark.parametrize(
    ("metadata", "agent_conf", "runtime_state", "message"),
    [
        (_metadata(num_devices=4), _agent_conf(), _runtime_state(), "device count"),
        (_metadata(trajectory_sharding="static"), _agent_conf(), _runtime_state(), "sharding mode"),
        (_metadata(num_envs=16), _agent_conf(), _runtime_state(), "environment count"),
        (_metadata(), _agent_conf(), {"last_obs": np.zeros((2, 5, 3))}, "replica/environment shape"),
    ],
)
def test_exact_resume_rejects_incompatible_topology(
    monkeypatch,
    metadata,
    agent_conf,
    runtime_state,
    message,
):
    monkeypatch.setattr(checkpoint, "device_count", lambda _config: 2)

    with pytest.raises(ValueError, match=message):
        checkpoint._validate_exact_resume_topology(metadata, runtime_state, agent_conf)
