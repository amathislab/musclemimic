from __future__ import annotations

import threading
from typing import Any

from musclemimic.runner.validation_video_recorder import AsyncValidationVideoRecorder
from musclemimic.utils.logging import setup_logger

from .validation_video_checkpoint import DEFAULT_CHECKPOINT_WAIT_TIMEOUT_S

logger = setup_logger(__name__)


class ExperimentHooks:
    """Hooks to customize logging and video behavior across experiments."""

    def build_video_recorder(self, result_dir: str, config: Any) -> AsyncValidationVideoRecorder | None:
        experiment = getattr(config, "experiment", {})
        val = experiment.get("validation", {}) if hasattr(config, "experiment") else {}
        active = bool(val.get("active", False))
        if not active:
            return None
        frequency = int(val.get("video_frequency", 10))
        if frequency <= 0:
            return None
        length = int(val.get("video_length", 250))
        if length <= 0:
            return None
        if not bool(experiment.get("save_checkpoints", False)):
            raise ValueError("Validation video requires save_checkpoints=true.")
        deterministic = bool(val.get("deterministic", True))
        return AsyncValidationVideoRecorder(
            video_dir=result_dir,
            frequency=frequency,
            length=length,
            deterministic=deterministic,
            checkpoint_wait_timeout_s=float(
                val.get(
                    "video_checkpoint_wait_timeout_s",
                    DEFAULT_CHECKPOINT_WAIT_TIMEOUT_S,
                )
            ),
        )

    def enrich_log(self, log_dict: dict[str, Any], metrics_dict: dict[str, Any], env) -> None:
        """Opportunity to add environment-specific fields to logs."""
        # Default: no-op
        return None

    def on_validation_video(
        self,
        use_wandb: bool,
        wandb,
        video_path: str | None,
        timestep: int,
        *,
        log_step: int | None = None,
    ) -> None:
        if not use_wandb or not video_path:
            return
        try:
            step = int(timestep if log_step is None else log_step)
            wandb.log(
                {
                    "Validation/Video": wandb.Video(video_path, format="mp4"),
                    "Validation/Video Timestep": int(timestep),
                },
                step=step,
            )
        except Exception:
            # Keep evaluation robust; do not raise during logging
            logger.warning("Failed to log validation video to W&B", exc_info=True)


class ExperimentLogger:
    """Serialize scalar logs and track the latest experiment step."""

    def __init__(self, hooks: ExperimentHooks, wandb: Any | None) -> None:
        self._hooks = hooks
        self._wandb = wandb
        self._step_lock = threading.Lock()
        self._last_step: int | None = None

    @property
    def wandb_enabled(self) -> bool:
        """Return whether logs are emitted to W&B."""
        return self._wandb is not None

    def enrich(self, log: dict[str, Any], metrics: dict[str, Any], env: Any) -> None:
        """Apply experiment-specific fields before emitting a scalar log."""
        self._hooks.enrich_log(log, metrics, env)

    def log(self, data: dict[str, Any], *, step: int) -> None:
        """Emit one scalar payload and advance the monotonic step watermark."""
        with self._step_lock:
            if self._wandb is not None:
                self._wandb.log(data, step=step)
            current = self._last_step
            self._last_step = int(step) if current is None else max(current, int(step))

    def log_video(self, video_path: str, *, timestep: int) -> None:
        """Emit a video without holding the scalar-log lock during upload."""
        with self._step_lock:
            log_step = timestep if self._last_step is None else max(self._last_step, int(timestep))

        self._hooks.on_validation_video(
            self._wandb is not None,
            self._wandb,
            video_path,
            timestep,
            log_step=log_step,
        )
