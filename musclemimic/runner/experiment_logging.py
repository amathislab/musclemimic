"""Experiment metrics, W&B, and training callback setup."""

from __future__ import annotations

import os
from typing import Any

from omegaconf import OmegaConf, open_dict

from musclemimic.algorithms.common.checkpoint_manager import checkpoint_path_for_step
from musclemimic.utils.logging import setup_logger
from musclemimic.utils.metrics import MetricsHandler

from .logging import ExperimentLogger
from .validation_video_recorder import AsyncValidationVideoRecorder, validation_video_due

logger = setup_logger(__name__)


def setup_wandb(config: Any) -> tuple[bool, Any]:
    """Initialize W&B when experiment logging is enabled."""
    import wandb

    mode = str(config.wandb.get("mode", "online"))
    if mode == "disabled":
        logger.info("Wandb logging disabled")
        return False, None

    if mode == "online":
        wandb.login()
    params = {
        "project": config.wandb.project,
        "mode": mode,
        "config": OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    }
    for key in ("entity", "tags", "name"):
        if config.wandb.get(key):
            params[key] = config.wandb[key]

    if config.wandb.get("fork_from"):
        params["fork_from"] = config.wandb.fork_from
    elif config.wandb.get("id"):
        params["id"] = config.wandb.id
    if not config.wandb.get("fork_from") and config.wandb.get("resume"):
        params["resume"] = config.wandb.resume
    return True, wandb.init(**params)


def configure_wandb_for_checkpoint_resume(config: Any, checkpoint_path: str | None) -> None:
    """Force local checkpoint resume to start a fresh W&B run."""
    if checkpoint_path is None or config.wandb.get("mode", "online") == "disabled":
        return

    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)
    with open_dict(config.wandb):
        config.wandb.id = None
        config.wandb.resume = None
        config.wandb.fork_from = None


def build_metrics_handler(config: Any, env: Any) -> MetricsHandler | None:
    """Create validation metrics only when validation is active."""
    validation_active = config.experiment.get("validation", {}).get("active", False)
    return MetricsHandler(config, env) if validation_active else None


def log_trajectory_summary(label: str, env: Any | None) -> None:
    """Log total duration for an environment's concatenated motion library."""
    if env is None or env.th is None or env.th.traj is None:
        return

    frequency = float(env.th.traj.info.frequency)
    n_frames = int(env.th.traj.data.qpos.shape[0])
    if frequency <= 0.0 or n_frames <= 0:
        return

    duration_seconds = (n_frames - 1) / frequency
    logger.info(
        f"[{label}] total concatenated motion duration: {duration_seconds / 60.0:.2f} minutes "
        f"({duration_seconds:.1f} s, {n_frames} frames @ {frequency:.1f} Hz)"
    )


def build_logging_callback(
    env: Any,
    config: Any,
    experiment_logger: ExperimentLogger,
    video_recorder: AsyncValidationVideoRecorder | None,
):
    """Build the host callback that formats and emits training metrics."""
    validation_number = 0

    def callback(metrics: dict[str, Any]) -> None:
        nonlocal validation_number
        if metrics.get("_log_only", False):
            model_metrics = {
                f"Model/{key.split('/', 1)[1]}": value for key, value in metrics.items() if key.startswith("model/")
            }
            if model_metrics:
                if not experiment_logger.wandb_enabled:
                    logger.info(model_metrics)
                experiment_logger.log(model_metrics, step=0)
            return

        reset_timestep = config.experiment.get("reset_logging_timestep", False)
        raw_timestep = metrics.get("jax_raw_timestep", metrics.get("max_timestep", 0.0))
        current_timestep = int(
            metrics.get("jax_raw_timestep", 0.0) if reset_timestep else metrics.get("max_timestep", raw_timestep)
        )
        log_values: dict[str, Any] = {
            "Learning Rate": metrics.get("learning_rate", 0.0),
            "Current Timestep": current_timestep,
            "Raw JAX Timestep": int(raw_timestep),
        }
        if "mean_episode_return" in metrics:
            log_values["Mean Episode Return"] = metrics["mean_episode_return"]
        if "mean_episode_length" in metrics:
            log_values["Mean Episode Length"] = metrics["mean_episode_length"]

        prefix_display = {"ppo": "PPO"}
        for key, value in metrics.items():
            if "/" not in key:
                continue
            prefix, suffix = key.split("/", 1)
            log_values[f"{prefix_display.get(prefix, prefix)}/{suffix}"] = value

        if metrics.get("has_validation_update", False):
            _add_validation_metrics(log_values, metrics)

        experiment_logger.enrich(log_values, metrics, env)
        experiment_logger.log(log_values, step=current_timestep)

        if (
            video_recorder is not None
            and metrics.get("has_validation_update", False)
            and "_checkpoint_update" in metrics
        ):
            scheduled_number = metrics.get("_validation_number")
            if scheduled_number is None:
                validation_number += 1
            else:
                validation_number = int(scheduled_number)
            _record_validation_video(
                metrics,
                config,
                video_recorder,
                validation_number,
                current_timestep,
            )

    return callback


def _add_validation_metrics(log_values: dict[str, Any], metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if not key.startswith("val_"):
            continue
        name = key.removeprefix("val_")
        if name == "mean_episode_return":
            log_values["Validation/Mean Episode Return"] = value
        elif name == "mean_episode_length":
            log_values["Validation/Mean Episode Length"] = value
        else:
            log_values[f"Validation Measures/{name}"] = value

    error_keys = (
        "val_euclidean_distance_site_rpos",
        "val_euclidean_distance_site_rrotvec",
        "val_euclidean_distance_site_rvel",
    )
    if all(metrics.get(key) is not None for key in error_keys):
        sweep_metric = sum(float(metrics[key]) for key in error_keys)
        log_values["Metric for Sweep"] = sweep_metric
        logger.info(f"Combined sweep metric: {sweep_metric:.4f}")


def _record_validation_video(
    metrics: dict[str, Any],
    config: Any,
    video_recorder: AsyncValidationVideoRecorder,
    validation_number: int,
    timestep: int,
) -> None:
    try:
        if not validation_video_due(
            validation_number,
            video_recorder.frequency,
        ):
            return
        checkpoint_update = int(metrics["_checkpoint_update"])
        if checkpoint_update < 0:
            raise RuntimeError("Video validation did not produce a checkpoint")
        checkpoint_path = checkpoint_path_for_step(
            config.experiment.checkpoint_dir,
            checkpoint_update,
        )
        video_recorder.record_episode(
            checkpoint_path,
            validation_number=validation_number,
            timestep=timestep,
        )
    except Exception:
        logger.warning("Video recording failed", exc_info=True)
