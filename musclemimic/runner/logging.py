from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from musclemimic.runner.validation_video_recorder import ValidationVideoRecorder


class ExperimentHooks:
    """Hooks to customize logging and video behavior across experiments."""

    def build_video_recorder(self, result_dir: str, config) -> ValidationVideoRecorder | None:
        val = getattr(config, "experiment", {}).get("validation", {}) if hasattr(config, "experiment") else {}
        active = bool(val.get("active", False))
        if not active:
            return None
        frequency = int(val.get("video_frequency", 10))
        length = int(val.get("video_length", 250))
        deterministic = bool(val.get("deterministic", True))
        return ValidationVideoRecorder(
            video_dir=result_dir,
            frequency=frequency,
            length=length,
            deterministic=deterministic,
            named_motions=val.get("video_motions", ()),
        )

    def enrich_log(self, log_dict: dict[str, Any], metrics_dict: dict[str, Any], env) -> None:
        """Opportunity to add environment-specific fields to logs."""
        # Default: no-op
        return None

    def on_validation_video(
        self,
        use_wandb: bool,
        wandb,
        video_path: str | Mapping[str, str] | None,
        timestep: int,
    ) -> None:
        if not use_wandb or not video_path:
            return
        try:
            if isinstance(video_path, Mapping):
                videos = {
                    f"Validation/Video/{name}": wandb.Video(path, format="mp4")
                    for name, path in video_path.items()
                }
            else:
                videos = {"Validation/Video": wandb.Video(video_path, format="mp4")}
            wandb.log(videos, step=int(timestep))
        except Exception:
            # Keep evaluation robust; do not raise during logging
            print("Warning: failed to log video to wandb.")


class UnifiedHooks(ExperimentHooks):
    """Unified hooks used by both fullbody and bimanual experiments."""

    pass
