"""Integration tests for auto-resume with real Orbax checkpoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.common.checkpoint_manager import (
    CheckpointMetadata,
    create_checkpoint_manager,
)
from musclemimic.runner.checkpointing import find_latest_checkpoint, resolve_checkpoint_dir


@pytest.mark.integration
def test_auto_resume_orbax_roundtrip(tmp_path):
    launch_dir = tmp_path / "launch"
    result_dir = tmp_path / "results" / "run1"
    launch_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)

    resolved_dir = resolve_checkpoint_dir(
        configured_ckpt_dir="checkpoints",
        launch_dir=str(launch_dir),
        result_dir=str(result_dir),
        experiment_id="exp123",
        auto_resume=True,
    )
    ckpt_dir = Path(resolved_dir)

    agent_conf = SimpleNamespace(config=OmegaConf.create({"experiment": {"num_envs": 1}}))
    train_state = SimpleNamespace(
        params={"w": jnp.array([1.0])},
        opt_state={"o": jnp.array([0.0])},
        step=jnp.array(1),
        run_stats={},
    )
    agent_state = SimpleNamespace(train_state=train_state)

    manager = create_checkpoint_manager(str(ckpt_dir), async_save=False)
    try:
        md1 = CheckpointMetadata(
            step=1,
            update_number=1,
            global_timestep=10,
            target_global_timestep=10,
            learning_rate=0.1,
            num_envs=1,
            num_steps=1,
            num_minibatches=1,
            update_epochs=1,
        )
        manager.save_checkpoint(1, agent_conf, agent_state, md1)

        md2 = CheckpointMetadata(
            step=2,
            update_number=2,
            global_timestep=20,
            target_global_timestep=20,
            learning_rate=0.1,
            num_envs=1,
            num_steps=1,
            num_minibatches=1,
            update_epochs=1,
        )
        manager.save_checkpoint(2, agent_conf, agent_state, md2)
    finally:
        manager.close()

    (ckpt_dir / "checkpoint_999").mkdir(parents=True)

    latest = find_latest_checkpoint(ckpt_dir)
    assert latest == str(ckpt_dir / "checkpoint_2")

    loader = create_checkpoint_manager(str(ckpt_dir), async_save=False)
    try:
        (_, _), metadata = loader.load_checkpoint(latest)
        assert metadata.update_number == 2
        assert metadata.global_timestep == 20
    finally:
        loader.close()
