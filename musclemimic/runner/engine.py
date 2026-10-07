"""Top-level experiment orchestration."""

from __future__ import annotations

import os
import sys
from typing import Any

from omegaconf import open_dict

from musclemimic.algorithms.common.distributed import device_count
from musclemimic.algorithms.common.trajectory_sharding import (
    prepare_trajectory_data_shards,
    prepare_validation_trajectory_data_shards,
)
from musclemimic.utils.logging import setup_logger

from .checkpointing import (
    build_resume_runtime_state_target,
    config_hash,
    find_latest_checkpoint,
    generate_run_suffix,
    resolve_checkpoint_dir,
    resume_or_fresh,
    validate_checkpoint_compatibility,
    write_manifest,
)
from .environment_factory import instantiate_env, instantiate_validation_env
from .experiment_logging import (
    build_logging_callback,
    build_metrics_handler,
    configure_wandb_for_checkpoint_resume,
    log_trajectory_summary,
    setup_wandb,
)
from .logging import ExperimentHooks, ExperimentLogger
from .training_runtime import (
    compute_training_rngs,
    release_unused_host_memory,
    run_training,
    setup_jax_cache,
)
from .trajectory_sharing import (
    attach_shared_validation_trajectory,
    should_share_validation_trajectory,
)
from .validation_video_logging import ValidationVideoLogMonitor, flush_validation_videos

logger = setup_logger(__name__)


def pick_algorithm(config: Any) -> Any:
    """Resolve the configured training algorithm."""
    from musclemimic.algorithms import FlashSACJax, PPOJax

    algorithm_name = config.experiment.get("algorithm", "PPOJax")
    if algorithm_name == "PPOJax":
        return PPOJax
    if algorithm_name == "FlashSACJax":
        return FlashSACJax
    raise ValueError(f"Unknown algorithm: {algorithm_name}")


def run_experiment(config: Any, hooks: ExperimentHooks) -> None:
    """Configure, execute, and clean up one training experiment."""
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_gpu_triton_gemm_any=True --xla_gpu_graph_min_graph_size=1",
    )
    setup_jax_cache()

    from hydra.core.hydra_config import HydraConfig

    hydra_runtime = HydraConfig.get().runtime
    result_dir = hydra_runtime.output_dir
    launch_dir = hydra_runtime.cwd

    explicit_resume = getattr(config.experiment, "resume_from", None)
    auto_resume = getattr(config.experiment, "auto_resume", True)
    exact_resume = bool(getattr(config.experiment, "exact_resume", False))
    experiment_hash = config_hash(config.experiment)
    experiment_id = getattr(config.experiment, "run_id", None) or experiment_hash

    checkpoint_dir = resolve_checkpoint_dir(
        configured_ckpt_dir=getattr(config.experiment, "checkpoint_dir", "checkpoints") or "checkpoints",
        launch_dir=launch_dir,
        result_dir=result_dir,
        experiment_id=experiment_id,
        auto_resume=auto_resume,
        checkpoint_root=getattr(config.experiment, "checkpoint_root", None),
    )
    if not auto_resume:
        checkpoint_dir = os.path.join(checkpoint_dir, generate_run_suffix())
    with open_dict(config.experiment):
        config.experiment.checkpoint_dir = checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_from = None
    apply_resume_resets = True
    using_local_auto_resume = False
    if auto_resume:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            resume_from = latest_checkpoint
            using_local_auto_resume = True
            apply_resume_resets = not exact_resume
            logger.info(f"Auto-resume: found checkpoint: {latest_checkpoint}")
            mode = "exact runtime-state restore" if exact_resume else "lightweight restore with env reset"
            logger.info(f"Auto-resume mode: {mode}")
            validate_checkpoint_compatibility(checkpoint_dir, experiment_hash)
        elif explicit_resume:
            resume_from = explicit_resume
            logger.info(f"Auto-resume: no local checkpoint, using explicit: {explicit_resume}")
        else:
            logger.info(f"Auto-resume: no checkpoint in {checkpoint_dir}, starting fresh")
    elif explicit_resume:
        resume_from = explicit_resume
        logger.info(f"Resuming from explicit path: {explicit_resume}")

    with open_dict(config.experiment):
        config.experiment.resume_from = resume_from

    configure_wandb_for_checkpoint_resume(
        config,
        resume_from if using_local_auto_resume else None,
    )
    use_wandb, wandb_run = setup_wandb(config)
    wandb = None
    if use_wandb:
        import wandb as wandb_module

        wandb = wandb_module
    if use_wandb and wandb_run is not None:
        with open_dict(config.experiment):
            config.experiment.wandb_run_id = str(wandb_run.id)

    algorithm_cls = pick_algorithm(config)
    runtime_state_target = None
    if using_local_auto_resume and exact_resume:
        runtime_state_target = build_resume_runtime_state_target(config, algorithm_cls)
        release_unused_host_memory()

    env = instantiate_env(config)
    agent_conf = algorithm_cls.init_agent_conf(env, config)
    n_devices = device_count(config)
    shares_training_trajectory = should_share_validation_trajectory(config) and env.th is not None

    log_trajectory_summary("Train", env)
    if shares_training_trajectory:
        log_trajectory_summary("Validation", env)

    trajectory_data_shards = prepare_trajectory_data_shards(env, config, n_devices)
    release_unused_host_memory()

    validation_env = instantiate_validation_env(
        config,
        share_trajectory=shares_training_trajectory,
    )
    if shares_training_trajectory and validation_env is not None:
        attach_shared_validation_trajectory(env, validation_env)
    if not shares_training_trajectory:
        log_trajectory_summary("Validation", validation_env)
    validation_trajectory_data_shards = prepare_validation_trajectory_data_shards(
        validation_env,
        config,
        n_devices,
        shares_training=shares_training_trajectory,
    )
    release_unused_host_memory()

    metrics_env = validation_env if validation_env is not None else env
    metrics_handler = build_metrics_handler(config, metrics_env)
    experiment_logger = ExperimentLogger(hooks, wandb)
    video_recorder = hooks.build_video_recorder(result_dir=result_dir, config=config)
    video_monitor = ValidationVideoLogMonitor(video_recorder, experiment_logger) if video_recorder is not None else None
    if video_monitor is not None:
        video_monitor.start()

    logging_callback = build_logging_callback(
        env,
        config,
        experiment_logger,
        video_recorder,
    )
    write_manifest(checkpoint_dir, config.experiment, experiment_hash)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Experiment ID (config hash): {experiment_hash}")

    training, resume_agent_state = resume_or_fresh(
        env,
        agent_conf,
        algorithm_cls,
        config,
        metrics_handler,
        val_env=validation_env,
        apply_resume_resets=apply_resume_resets,
        preserve_checkpoint_target=using_local_auto_resume,
        runtime_state_target=runtime_state_target,
    )

    rngs = compute_training_rngs(config)
    try:
        run_training(
            training,
            rngs,
            config=config,
            trajectory_data_shards=trajectory_data_shards,
            validation_trajectory_data_shards=validation_trajectory_data_shards,
            resume_agent_state=resume_agent_state,
            logging_callback=logging_callback,
            logging_interval=int(config.experiment.get("online_logging_interval", 1)),
        )
    finally:
        exiting_with_error = sys.exc_info()[0] is not None
        if video_monitor is not None:
            video_monitor.stop()
        flush_validation_videos(video_recorder, experiment_logger, wait=not exiting_with_error)
        if wandb is not None and wandb_run is not None:
            wandb.finish()
