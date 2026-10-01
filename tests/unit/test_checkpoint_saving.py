"""Tests for host-side checkpoint ownership."""

from __future__ import annotations

from types import SimpleNamespace

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
from musclemimic.runner import checkpoint_saving


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


def _runtime_state(num_devices: int = 1) -> PPORuntimeState:
    leading_shape = () if num_devices == 1 else (num_devices,)
    return PPORuntimeState(
        train_state=_TrainState(
            params={"w": jnp.ones((*leading_shape, 2))},
            run_stats={"mean": jnp.zeros((*leading_shape, 1))},
            opt_state={"count": jnp.full(leading_shape, 12, dtype=jnp.int32)},
            step=jnp.full(leading_shape, 12, dtype=jnp.int32),
        ),
        env_state={"position": jnp.ones((*leading_shape, 3))},
        last_obs=jnp.ones((*leading_shape, 4)),
        rng=jnp.zeros((*leading_shape, 2), dtype=jnp.uint32),
        lr=jnp.full(leading_shape, 3e-4),
        val_rng=jnp.ones((*leading_shape, 2), dtype=jnp.uint32),
        curriculum_state=create_curriculum_state(0.3, 1.0),
        reward_curriculum_state=create_reward_curriculum_state({}),
    )


def _agent_conf(
    *,
    exact_resume: bool,
    save_runtime_state: bool | None = None,
    checkpoint_dir: str = "/tmp/ckpts",
) -> SimpleNamespace:
    experiment = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_format": "orbax",
        "max_checkpoints_to_keep": 3,
        "num_envs": 64,
        "num_steps": 20,
        "num_minibatches": 4,
        "update_epochs": 2,
        "async_checkpointing": True,
        "exact_resume": exact_resume,
    }
    if save_runtime_state is not None:
        experiment["save_runtime_state"] = save_runtime_state
    config = OmegaConf.create(
        {
            "experiment": experiment,
        }
    )
    return SimpleNamespace(config=config, network=object(), tx=object())


def test_checkpoint_saver_preserves_exact_multi_device_runtime(monkeypatch):
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
        num_devices=2,
    )

    path = saver.save(_runtime_state(num_devices=2), updates_done=8)
    saver.close()

    assert path == "/tmp/checkpoint_8"
    step, _, agent_state, metadata = manager.saved[0]
    assert step == 8
    assert agent_state.train_state.params["w"].shape == (2,)
    assert agent_state.runtime_state["last_obs"].shape == (2, 4)
    assert metadata.global_timestep == 10_000 + 3 * 64 * 20
    assert metadata.target_global_timestep == 100_000
    assert metadata.num_devices == 2
    assert metadata.trajectory_sharding == "none"
    np.testing.assert_allclose(metadata.learning_rate, 3e-4)
    assert manager.wait_count == 2
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
        num_devices=1,
    )

    saver.save(_runtime_state(), updates_done=3)
    saver.close()

    assert not hasattr(manager.saved[0][2], "runtime_state")


def test_compact_resume_can_write_exact_runtime_state(monkeypatch, tmp_path):
    manager = FakeCheckpointManager()
    monkeypatch.setattr(
        checkpoint_saving,
        "create_checkpoint_manager",
        lambda *_args, **_kwargs: manager,
    )
    saver = checkpoint_saving.PPOCheckpointSaver(
        _agent_conf(
            exact_resume=False,
            save_runtime_state=True,
            checkpoint_dir=str(tmp_path),
        ),
        object(),
        base_global_timestep=0,
        base_completed_updates=0,
        target_global_timestep=100_000,
        num_devices=1,
    )

    saver.save(_runtime_state(), updates_done=3)

    assert hasattr(manager.saved[0][2], "runtime_state")


def test_exact_resume_always_writes_runtime_state(monkeypatch, tmp_path):
    manager = FakeCheckpointManager()
    monkeypatch.setattr(
        checkpoint_saving,
        "create_checkpoint_manager",
        lambda *_args, **_kwargs: manager,
    )
    saver = checkpoint_saving.PPOCheckpointSaver(
        _agent_conf(
            exact_resume=True,
            save_runtime_state=False,
            checkpoint_dir=str(tmp_path),
        ),
        object(),
        base_global_timestep=0,
        base_completed_updates=0,
        target_global_timestep=100_000,
        num_devices=1,
    )

    saver.save(_runtime_state(), updates_done=3)

    assert hasattr(manager.saved[0][2], "runtime_state")


def test_checkpoint_saver_rejects_poisoned_optimizer_state(monkeypatch, tmp_path):
    manager = FakeCheckpointManager()
    monkeypatch.setattr(
        checkpoint_saving,
        "create_checkpoint_manager",
        lambda *_args, **_kwargs: manager,
    )
    saver = checkpoint_saving.PPOCheckpointSaver(
        _agent_conf(exact_resume=False, checkpoint_dir=str(tmp_path)),
        object(),
        base_global_timestep=0,
        base_completed_updates=0,
        target_global_timestep=100_000,
        num_devices=1,
    )
    runtime_state = _runtime_state()
    runtime_state = runtime_state.replace(
        train_state=runtime_state.train_state.replace(
            opt_state={
                "last_finite": jnp.asarray(False),
                "notfinite_count": jnp.asarray(1, dtype=jnp.int32),
                "total_notfinite": jnp.asarray(1, dtype=jnp.int32),
                "inner_state": {"count": jnp.asarray(12, dtype=jnp.int32)},
            }
        )
    )

    with pytest.raises(ValueError, match=r"last_finite.*False"):
        saver.save(runtime_state, updates_done=3)

    assert manager.saved == []
