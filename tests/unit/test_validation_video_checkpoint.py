"""Tests for path-backed validation policy checkpoint loading."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.common.checkpoint_manager import (
    UnifiedCheckpointManager,
    checkpoint_path_for_step,
)
from musclemimic.runner import validation_video_checkpoint


def test_wait_for_checkpoint_observes_exact_path(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint_9"
    readiness = iter((False, False, True))
    sleeps = []

    monkeypatch.setattr(
        validation_video_checkpoint,
        "is_checkpoint_complete",
        lambda path: next(readiness),
    )
    timestamps = iter((0.0, 0.1, 0.2))
    monkeypatch.setattr(validation_video_checkpoint, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(validation_video_checkpoint, "sleep", sleeps.append)

    ready_path = validation_video_checkpoint.wait_for_checkpoint(
        checkpoint_path,
        timeout_s=1.0,
        poll_interval_s=0.25,
    )

    assert ready_path == checkpoint_path.resolve()
    assert sleeps == [0.25, 0.25]


def test_wait_for_checkpoint_times_out_without_falling_back(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint_11"
    monkeypatch.setattr(
        validation_video_checkpoint,
        "is_checkpoint_complete",
        lambda path: False,
    )
    timestamps = iter((0.0, 2.0))
    monkeypatch.setattr(validation_video_checkpoint, "monotonic", lambda: next(timestamps))

    with pytest.raises(TimeoutError, match=str(checkpoint_path.resolve())):
        validation_video_checkpoint.wait_for_checkpoint(
            checkpoint_path,
            timeout_s=1.0,
        )


@pytest.mark.parametrize("async_save", [False, True])
@pytest.mark.parametrize(
    ("algorithm", "policy_state", "expected_field"),
    [
        (
            "PPOJax",
            {
                "params": {"weights": jnp.asarray([1.0, 2.0])},
                "run_stats": {"mean": jnp.asarray([3.0])},
            },
            "params",
        ),
        (
            "FlashSACJax",
            {
                "actor_params": {"weights": jnp.asarray([1.0, 2.0])},
                "actor_batch_stats": {"mean": jnp.asarray([3.0])},
            },
            "actor_params",
        ),
    ],
)
def test_policy_loader_restores_only_inference_state(
    tmp_path,
    async_save,
    algorithm,
    policy_state,
    expected_field,
):
    checkpoint_root = tmp_path / "checkpoints"
    manager = UnifiedCheckpointManager(
        str(checkpoint_root),
        async_save=async_save,
    )
    if algorithm == "PPOJax":
        train_state = SimpleNamespace(
            **policy_state,
            opt_state={"momentum": jnp.asarray([4.0, 5.0, 6.0], dtype=jnp.float32)},
            step=jnp.int32(7),
        )
        checkpoint_state = SimpleNamespace(train_state=train_state)
    else:
        checkpoint_state = {
            **policy_state,
            "critic_params": {"weights": jnp.asarray([4.0, 5.0, 6.0])},
            "update_step": jnp.int32(7),
        }
    config = OmegaConf.create({"experiment": {"algorithm": algorithm}})
    try:
        checkpoint_path = manager.save_checkpoint(
            7,
            SimpleNamespace(config=config),
            checkpoint_state,
        )
        restored = validation_video_checkpoint.load_validation_policy_checkpoint(
            checkpoint_path,
            timeout_s=5.0,
        )
    finally:
        manager.close()

    assert checkpoint_path == checkpoint_path_for_step(checkpoint_root, 7)
    assert restored.config == {"experiment": {"algorithm": algorithm}}
    np.testing.assert_array_equal(
        restored.policy_state[expected_field]["weights"],
        [1.0, 2.0],
    )
    assert set(restored.policy_state) == set(policy_state)
