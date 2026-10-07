"""Save PPO checkpoints from the training loop."""

from __future__ import annotations

from time import monotonic
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np

from musclemimic.algorithms.common.checkpoint_manager import (
    CheckpointMetadata,
    create_checkpoint_manager,
)
from musclemimic.algorithms.common.checkpoint_utils import (
    TrainingConfig,
    compute_checkpoint_metadata,
)
from musclemimic.algorithms.ppo.state import PPORuntimeState
from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__)


class PPOCheckpointSaver:
    """Save PPO state with one checkpoint manager."""

    def __init__(
        self,
        agent_conf: Any,
        env: Any,
        *,
        base_global_timestep: int,
        base_completed_updates: int,
        target_global_timestep: int,
    ) -> None:
        self._agent_conf = agent_conf
        self._experiment = agent_conf.config.experiment
        self._base_global_timestep = base_global_timestep
        self._base_completed_updates = base_completed_updates
        self._target_global_timestep = target_global_timestep
        self._manager = create_checkpoint_manager(
            str(self._experiment.checkpoint_dir),
            format=str(self._experiment.get("checkpoint_format", "auto")),
            max_to_keep=int(self._experiment.get("max_checkpoints_to_keep", 5)),
            async_save=bool(self._experiment.get("async_checkpointing", True)),
            save_concurrent_gb=int(self._experiment.get("checkpoint_save_concurrent_gb", 4)),
            restore_concurrent_gb=int(self._experiment.get("checkpoint_restore_concurrent_gb", 4)),
        )
        self._backend = getattr(env, "mjx_backend", "jax")
        self._env_name = type(env).__name__

    def save(
        self,
        runtime_state: PPORuntimeState,
        *,
        updates_done: int,
    ) -> str:
        """Transfer one runtime state to host and enqueue its checkpoint save."""
        started_at = monotonic()
        train_state = jax.device_get(runtime_state.train_state)

        agent_state = SimpleNamespace(train_state=train_state)
        if bool(self._experiment.get("exact_resume", False)):
            agent_state.runtime_state = jax.device_get(runtime_state.to_checkpoint_state())

        training_config = TrainingConfig.from_experiment_config(self._experiment)
        optimizer_step = int(np.asarray(train_state.step))
        learning_rate = float(np.asarray(jax.device_get(runtime_state.lr)))
        metadata_values = compute_checkpoint_metadata(
            optimizer_step,
            training_config,
            learning_rate,
        )
        updates_since_resume = max(
            0,
            int(updates_done) - self._base_completed_updates,
        )
        metadata_values["update_number"] = int(updates_done)
        metadata_values["global_timestep"] = (
            self._base_global_timestep + updates_since_resume * training_config.num_steps * training_config.num_envs
        )
        metadata = CheckpointMetadata(
            **metadata_values,
            target_global_timestep=self._target_global_timestep,
            backend=self._backend,
            env_name=self._env_name,
            wandb_run_id=str(self._experiment.get("wandb_run_id", "") or ""),
        )
        path = self._manager.save_checkpoint(
            metadata.update_number,
            self._agent_conf,
            agent_state,
            metadata,
        )
        mode = "async" if bool(self._experiment.get("async_checkpointing", True)) else "sync"
        logger.info(
            "Saved checkpoint (%s) at update %d: %s",
            mode,
            metadata.update_number,
            path,
        )
        logger.info("  Global timestep: %s", f"{metadata.global_timestep:,}")
        logger.info(
            "Checkpoint host save returned: update=%d host_seconds=%.3f",
            metadata.update_number,
            monotonic() - started_at,
        )
        return path

    def close(self) -> None:
        """Wait for pending asynchronous saves and close the manager."""
        self._manager.wait_until_finished()
        self._manager.close()

    def __enter__(self) -> PPOCheckpointSaver:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
