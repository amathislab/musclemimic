"""Host-side FlashSAC checkpoint saving."""

from __future__ import annotations

from time import monotonic
from typing import Any

import jax
import numpy as np

from musclemimic.algorithms.common.checkpoint_manager import (
    CheckpointMetadata,
    create_checkpoint_manager,
)
from musclemimic.algorithms.flashsac.checkpoint import FLASHSAC_ALGO_VERSION
from musclemimic.algorithms.flashsac.state import FlashSACAgentState, FlashSACTrainState
from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__)


class FlashSACCheckpointSaver:
    """Save FlashSAC learner state, with replay storage when configured."""

    def __init__(
        self,
        agent_conf: Any,
        env: Any,
        *,
        base_global_timestep: int,
        base_completed_interactions: int,
        target_global_timestep: int,
    ) -> None:
        self._agent_conf = agent_conf
        self._experiment = agent_conf.config.experiment
        self._base_global_timestep = int(base_global_timestep)
        self._base_completed_interactions = int(base_completed_interactions)
        self._target_global_timestep = int(target_global_timestep)
        self._manager = create_checkpoint_manager(
            str(self._experiment.checkpoint_dir),
            format=str(self._experiment.get("checkpoint_format", "auto")),
            max_to_keep=int(self._experiment.get("max_checkpoints_to_keep", 5)),
            async_save=bool(self._experiment.get("async_checkpointing", True)),
            save_concurrent_gb=int(self._experiment.get("checkpoint_save_concurrent_gb", 4)),
            restore_concurrent_gb=int(self._experiment.get("checkpoint_restore_concurrent_gb", 4)),
        )
        self._backend = env.mjx_backend
        self._env_name = type(env).__name__

    def save(self, train_state: FlashSACTrainState, *, interactions_done: int) -> str:
        """Enqueue a learner checkpoint with optional replay array storage."""
        started_at = monotonic()
        include_replay_buffer = bool(self._experiment.get("checkpoint_include_replay_buffer", True))
        # The next update donates train_state. Complete the device-to-host copy
        # before returning so Orbax never reads a buffer after it is donated.
        # Remove replay before the transfer when it is disabled; removing it
        # after device_get would still gather the global replay on the host.
        checkpoint_payload = FlashSACAgentState(train_state=train_state).serialize(
            include_replay_buffer=include_replay_buffer
        )
        checkpoint_state = jax.device_get(checkpoint_payload)
        updates_since_resume = max(
            0,
            int(interactions_done) - self._base_completed_interactions,
        )
        global_timestep = self._base_global_timestep + updates_since_resume * int(self._experiment.num_envs)
        optimizer_step = int(np.asarray(checkpoint_state["update_step"]))
        learning_rate = float(
            np.asarray(
                jax.device_get(
                    self._agent_conf.learning_rate_schedule(checkpoint_state["update_step"]),
                )
            )
        )
        metadata = CheckpointMetadata(
            step=optimizer_step,
            update_number=int(interactions_done),
            global_timestep=global_timestep,
            target_global_timestep=self._target_global_timestep,
            learning_rate=learning_rate,
            num_envs=int(self._experiment.num_envs),
            num_steps=1,
            num_minibatches=1,
            update_epochs=1,
            algo_version=FLASHSAC_ALGO_VERSION,
            backend=self._backend,
            env_name=self._env_name,
            wandb_run_id=str(self._experiment.get("wandb_run_id", "") or ""),
        )
        path = self._manager.save_checkpoint(
            metadata.update_number,
            self._agent_conf,
            checkpoint_state,
            metadata,
        )
        mode = "async" if bool(self._experiment.get("async_checkpointing", True)) else "sync"
        logger.info(
            "Saved FlashSAC checkpoint (%s, replay=%s) at interaction %d: %s",
            mode,
            "included" if include_replay_buffer else "omitted",
            metadata.update_number,
            path,
        )
        logger.info("  Global timestep: %s", f"{metadata.global_timestep:,}")
        logger.info(
            "FlashSAC checkpoint host save returned: interaction=%d host_seconds=%.3f",
            metadata.update_number,
            monotonic() - started_at,
        )
        return path

    def close(self) -> None:
        self._manager.wait_until_finished()
        self._manager.close()
