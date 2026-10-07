"""Tests for host-side checkpoint ownership."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.common.curriculum import (
    create_curriculum_state,
    create_reward_curriculum_state,
)
from musclemimic.algorithms.ppo.state import PPORuntimeState
from musclemimic.runner import checkpoint_saving, flashsac_checkpoint_saving


class FakeCheckpointManager:
    def __init__(self):
        self.wait_count = 0
        self.close_count = 0
        self.saved = []

    def wait_until_finished(self):
        self.wait_count += 1

    def close(self):
        self.close_count += 1

    def save_checkpoint(self, step, agent_conf, agent_state, metadata):
        self.saved.append((step, agent_conf, agent_state, metadata))
        return f"/tmp/checkpoint_{step}"


@struct.dataclass
class _TrainState:
    params: dict
    run_stats: dict
    opt_state: dict
    step: jnp.ndarray


@struct.dataclass
class _FlashReplay:
    observations: jnp.ndarray


@struct.dataclass
class _FlashState:
    replay_buffer: _FlashReplay
    update_step: jnp.ndarray
    weights: jnp.ndarray


def _runtime_state(num_devices: int = 1) -> PPORuntimeState:
    return PPORuntimeState(
        train_state=_TrainState(
            params={"w": jnp.ones((2,))},
            run_stats={"mean": jnp.zeros((1,))},
            opt_state={"count": jnp.asarray(0, dtype=jnp.int32)},
            step=jnp.asarray(12, dtype=jnp.int32),
        ),
        env_state={"position": jnp.ones((8, 3))},
        last_obs=jnp.ones((8, 4)),
        rng=jnp.zeros((num_devices, 2), dtype=jnp.uint32),
        lr=jnp.asarray(3e-4),
        val_rng=jnp.ones((num_devices, 2), dtype=jnp.uint32),
        curriculum_state=create_curriculum_state(0.3, 1.0),
        reward_curriculum_state=create_reward_curriculum_state({}),
    )


def _agent_conf(*, exact_resume: bool) -> SimpleNamespace:
    config = OmegaConf.create(
        {
            "experiment": {
                "checkpoint_dir": "/tmp/ckpts",
                "checkpoint_format": "orbax",
                "max_checkpoints_to_keep": 3,
                "num_envs": 64,
                "num_steps": 20,
                "num_minibatches": 4,
                "update_epochs": 2,
                "async_checkpointing": True,
                "exact_resume": exact_resume,
            }
        }
    )
    return SimpleNamespace(config=config, network=object(), tx=object())


def test_checkpoint_saver_preserves_exact_global_runtime(monkeypatch):
    manager = FakeCheckpointManager()
    monkeypatch.setattr(
        checkpoint_saving,
        "create_checkpoint_manager",
        lambda *_args, **_kwargs: manager,
    )
    agent_conf = _agent_conf(exact_resume=True)
    saver = checkpoint_saving.PPOCheckpointSaver(
        agent_conf,
        SimpleNamespace(mjx_backend="jax"),
        base_global_timestep=10_000,
        base_completed_updates=5,
        target_global_timestep=100_000,
    )

    path = saver.save(_runtime_state(num_devices=2), updates_done=8)
    saver.close()

    assert path == "/tmp/checkpoint_8"
    step, _, agent_state, metadata = manager.saved[0]
    assert step == 8
    assert agent_state.train_state.params["w"].shape == (2,)
    assert agent_state.runtime_state["last_obs"].shape == (8, 4)
    assert agent_state.runtime_state["rng"].shape == (2, 2)
    assert metadata.global_timestep == 10_000 + 3 * 64 * 20
    assert metadata.target_global_timestep == 100_000
    np.testing.assert_allclose(metadata.learning_rate, 3e-4)
    assert manager.wait_count == 1
    assert manager.close_count == 1


def test_lightweight_checkpoint_omits_runtime_state(monkeypatch):
    manager = FakeCheckpointManager()
    monkeypatch.setattr(
        checkpoint_saving,
        "create_checkpoint_manager",
        lambda *_args, **_kwargs: manager,
    )
    saver = checkpoint_saving.PPOCheckpointSaver(
        _agent_conf(exact_resume=False),
        object(),
        base_global_timestep=0,
        base_completed_updates=0,
        target_global_timestep=100_000,
    )

    saver.save(_runtime_state(), updates_done=3)
    saver.close()

    assert not hasattr(manager.saved[0][2], "runtime_state")


@pytest.mark.parametrize("include_replay_buffer", [None, False, True])
def test_flashsac_checkpoint_owns_a_bounded_host_snapshot(
    monkeypatch,
    include_replay_buffer,
):
    manager = FakeCheckpointManager()
    monkeypatch.setattr(
        flashsac_checkpoint_saving,
        "create_checkpoint_manager",
        lambda *_args, **_kwargs: manager,
    )
    snapshots = []
    device_get = jax.device_get

    def record_device_get(value):
        snapshots.append(value)
        return device_get(value)

    monkeypatch.setattr(flashsac_checkpoint_saving.jax, "device_get", record_device_get)
    experiment_data = {
        "checkpoint_dir": "/tmp/flashsac_ckpts",
        "checkpoint_format": "orbax",
        "max_checkpoints_to_keep": 1,
        "async_checkpointing": True,
        "num_envs": 8,
    }
    if include_replay_buffer is not None:
        experiment_data["checkpoint_include_replay_buffer"] = include_replay_buffer
    experiment = OmegaConf.create(experiment_data)
    agent_conf = SimpleNamespace(
        config=OmegaConf.create({"experiment": experiment}),
        learning_rate_schedule=lambda _step: jnp.asarray(3e-4),
    )
    saver = flashsac_checkpoint_saving.FlashSACCheckpointSaver(
        agent_conf,
        SimpleNamespace(mjx_backend="warp"),
        base_global_timestep=0,
        base_completed_interactions=0,
        target_global_timestep=100,
    )
    state = _FlashState(
        replay_buffer=_FlashReplay(jnp.ones((8, 4))),
        update_step=jnp.asarray(3, dtype=jnp.int32),
        weights=jnp.ones((2,)),
    )

    saver.save(state, interactions_done=3)

    expected_replay = True if include_replay_buffer is None else include_replay_buffer
    assert isinstance(snapshots[0], dict)
    assert ("replay_buffer" in snapshots[0]) is expected_replay
    saved_state = manager.saved[0][2]
    assert ("replay_buffer" in saved_state) is expected_replay
    assert isinstance(saved_state["weights"], np.ndarray)
    assert manager.saved[0][3].algo_version == flashsac_checkpoint_saving.FLASHSAC_ALGO_VERSION
    if expected_replay:
        assert isinstance(saved_state["replay_buffer"]["observations"], np.ndarray)
