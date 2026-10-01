"""FlashSAC experiment orchestration on the data-parallel runtime."""

from __future__ import annotations

import os
from typing import Any

from omegaconf import open_dict

from musclemimic.algorithms.common.distributed import device_count
from musclemimic.algorithms.common.trajectory_sharding import (
    prepare_trajectory_data_shards,
    prepare_validation_trajectory_data_shards,
)
from musclemimic.algorithms.flashsac import FlashSACJax
from musclemimic.runner.environment_factory import instantiate_env, instantiate_validation_env
from musclemimic.runner.flashsac_runtime import run_flashsac_training
from musclemimic.runner.trajectory_sharing import (
    attach_shared_validation_trajectory,
    should_share_validation_trajectory,
)
from musclemimic.utils.logging import setup_logger

from .checkpointing import (
    _canonicalize_resume_path,
    config_hash,
    find_latest_checkpoint,
    resolve_checkpoint_dir,
    validate_checkpoint_compatibility,
    write_manifest,
)

logger = setup_logger(__name__)


def run_flashsac_experiment(config: Any, hooks: Any) -> Any:
    """Run FlashSAC without changing the established PPO coordinator."""
    from hydra.core.hydra_config import HydraConfig

    from musclemimic.runner import engine

    engine.validate_training_backend(config)
    engine.setup_jax_cache()
    use_wandb, wandb_run = engine.setup_wandb(config)

    env = instantiate_env(config)
    share_trajectory = should_share_validation_trajectory(config) and env.th is not None
    validation_env = instantiate_validation_env(config, share_trajectory=share_trajectory)
    if share_trajectory and validation_env is not None:
        attach_shared_validation_trajectory(env, validation_env)

    agent_conf = FlashSACJax.init_agent_conf(env, config)
    n_devices = device_count(config)
    trajectory_data_shards = prepare_trajectory_data_shards(env, config, n_devices)
    validation_trajectory_data_shards = prepare_validation_trajectory_data_shards(
        validation_env,
        config,
        n_devices,
        shares_training=share_trajectory,
    )

    metrics_env = validation_env if validation_env is not None else env
    metrics_handler = engine.build_metrics_handler(config, metrics_env)
    runtime = HydraConfig.get().runtime
    recorder = hooks.build_video_recorder(result_dir=runtime.output_dir, config=config)
    hooks._video_recorder = recorder
    logging_callback = engine.build_logging_callback(
        env,
        config,
        agent_conf,
        use_wandb,
        hooks,
        algorithm_cls=FlashSACJax,
    )

    experiment_hash = config_hash(config.experiment)
    experiment_id = config.experiment.get("run_id") or experiment_hash
    auto_resume = bool(config.experiment.get("auto_resume", True))
    explicit_resume = config.experiment.get("resume_from")
    checkpoint_dir = resolve_checkpoint_dir(
        configured_ckpt_dir=config.experiment.get("checkpoint_dir", "checkpoints") or "checkpoints",
        launch_dir=runtime.cwd,
        result_dir=runtime.output_dir,
        experiment_id=experiment_id,
        auto_resume=auto_resume,
        checkpoint_root=config.experiment.get("checkpoint_root"),
    )
    if not auto_resume:
        checkpoint_dir = os.path.join(checkpoint_dir, engine._generate_run_suffix())
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open_dict(config.experiment):
        config.experiment.checkpoint_dir = checkpoint_dir

    resume_from = None
    using_local_auto_resume = False
    if auto_resume:
        resume_from = find_latest_checkpoint(checkpoint_dir)
        if resume_from is not None:
            using_local_auto_resume = True
            validate_checkpoint_compatibility(checkpoint_dir, experiment_hash)
        elif explicit_resume:
            resume_from = _canonicalize_resume_path(
                explicit_resume,
                revision=config.experiment.get("revision"),
            )
    elif explicit_resume:
        resume_from = _canonicalize_resume_path(
            explicit_resume,
            revision=config.experiment.get("revision"),
        )
    with open_dict(config.experiment):
        config.experiment.resume_from = resume_from

    write_manifest(checkpoint_dir, config.experiment, experiment_hash)
    logger.info(
        "FlashSAC topology: devices=%d global_envs=%d local_envs=%d",
        n_devices,
        int(config.experiment.num_envs),
        int(config.experiment.num_envs) // n_devices,
    )

    resume_state = None
    if resume_from is None:
        training = FlashSACJax.build_training(
            env,
            agent_conf,
            metrics_handler=metrics_handler,
            val_env=validation_env,
        )
    else:
        training, resume_state = FlashSACJax.build_resumed_training(
            env,
            agent_conf,
            resume_from,
            metrics_handler=metrics_handler,
            val_env=validation_env,
            apply_resume_resets=True,
            preserve_checkpoint_target=using_local_auto_resume,
        )

    try:
        return run_flashsac_training(
            training,
            engine.compute_training_rngs(config),
            config=config,
            trajectory_data_shards=trajectory_data_shards,
            validation_trajectory_data_shards=validation_trajectory_data_shards,
            resume_agent_state=resume_state,
            logging_callback=logging_callback,
            logging_interval=int(config.experiment.get("online_logging_interval", 1)),
        )
    finally:
        if use_wandb and wandb_run is not None:
            import wandb

            wandb.finish()


__all__ = ["run_flashsac_experiment"]
