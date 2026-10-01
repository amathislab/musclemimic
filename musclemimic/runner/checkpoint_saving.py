"""Save PPO checkpoints from the training loop."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np

from musclemimic.algorithms.common.checkpoint_manager import (
    CheckpointMetadata,
    checkpoint_path_for_step,
    create_checkpoint_manager,
)
from musclemimic.algorithms.common.checkpoint_utils import (
    TrainingConfig,
    compute_checkpoint_metadata,
)
from musclemimic.algorithms.common.ppo_trajectory_sharding import trajectory_sharding_mode
from musclemimic.algorithms.ppo.checkpoint_integrity import validate_ppo_checkpoint_train_state
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
        num_devices: int,
    ) -> None:
        self._agent_conf = agent_conf
        self._experiment = agent_conf.config.experiment
        self._base_global_timestep = base_global_timestep
        self._base_completed_updates = base_completed_updates
        self._target_global_timestep = target_global_timestep
        self._num_devices = num_devices
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
        checkpoint_path = checkpoint_path_for_step(
            self._experiment.checkpoint_dir,
            updates_done,
        )
        # A numerical failure can land exactly on an interval checkpoint. That
        # existing pre-chunk checkpoint is already the desired recovery state;
        # Orbax does not permit saving the same step twice.
        self._manager.wait_until_finished()
        if Path(checkpoint_path).is_dir():
            logger.info(
                "Reusing existing safe checkpoint at update %d: %s",
                updates_done,
                checkpoint_path,
            )
            return checkpoint_path

        train_state = runtime_state.train_state
        if self._num_devices > 1:
            train_state = jax.tree.map(lambda value: value[0], train_state)
        train_state = jax.device_get(train_state)
        validate_ppo_checkpoint_train_state(
            train_state,
            context=f"checkpoint save at update {updates_done}",
        )

        agent_state = SimpleNamespace(train_state=train_state)
        save_runtime_state = bool(self._experiment.get("save_runtime_state", False)) or bool(
            self._experiment.get("exact_resume", False)
        )
        if save_runtime_state:
            agent_state.runtime_state = jax.device_get(runtime_state.to_checkpoint_state())

        training_config = TrainingConfig.from_experiment_config(self._experiment)
        optimizer_step = int(np.asarray(train_state.step))
        learning_rate_value = runtime_state.lr[0] if self._num_devices > 1 else runtime_state.lr
        learning_rate = float(np.asarray(jax.device_get(learning_rate_value)))
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
            num_devices=self._num_devices,
            trajectory_sharding=trajectory_sharding_mode(self._agent_conf.config),
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
