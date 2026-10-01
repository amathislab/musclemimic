"""PPO-only experiment orchestration that preserves the legacy FlashSAC path."""

from __future__ import annotations

import os
from typing import Any

from omegaconf import open_dict

from musclemimic.algorithms.common.ppo_distributed import device_count
from musclemimic.algorithms.common.ppo_trajectory_sharding import (
    prepare_trajectory_data_shards,
    prepare_validation_trajectory_data_shards,
)
from musclemimic.algorithms.ppo.ppo import PPOJax
from musclemimic.runner.ppo_environment_factory import (
    instantiate_training_env,
    instantiate_validation_env,
)
from musclemimic.runner.training_runtime import (
    compute_training_rngs,
    release_unused_host_memory,
    run_training,
)
from musclemimic.utils.logging import setup_logger

from .checkpointing import (
    _canonicalize_resume_path,
    config_hash,
    find_latest_checkpoint,
    require_ppo_local_resume_compatibility,
    resolve_checkpoint_dir,
    write_manifest,
)

logger = setup_logger(__name__)


def _bind_ppo_wandb_run_id(config: Any) -> str | None:
    """Use the durable training run ID for W&B restart continuity."""
    run_id = str(config.experiment.get("wandb_run_id") or config.experiment.get("run_id") or "").strip()
    if not run_id:
        return None
    with open_dict(config.experiment):
        config.experiment.wandb_run_id = run_id
    return run_id


def _require_compatible_local_resume(checkpoint_dir: str, experiment_hash: str) -> None:
    """Refuse implicit PPO resume unless durable config and source identities match."""
    try:
        require_ppo_local_resume_compatibility(checkpoint_dir, experiment_hash)
    except ValueError as error:
        raise ValueError(
            f"PPO auto-resume refused the local checkpoint: {error}. "
            "Use a new run_id, or set auto_resume=false with an explicit resume_from "
            "after reviewing the compatibility change."
        ) from error


def _require_compatible_existing_local_manifest(
    checkpoint_dir: str,
    experiment_hash: str,
) -> None:
    """Validate a manifest left behind before any finalized PPO checkpoint."""
    if os.path.lexists(os.path.join(checkpoint_dir, "manifest.json")):
        _require_compatible_local_resume(checkpoint_dir, experiment_hash)


def run_ppo_experiment(config: Any, hooks: Any) -> Any:
    """Run PPO through the pure update graph and host-side coordinator."""
    # Import lazily to reuse TERRA's existing MetricsHandler monkey-patch and
    # logging hooks without creating an engine import cycle.
    from musclemimic.runner import engine

    engine.validate_training_backend(config)
    engine.setup_jax_cache()
    _bind_ppo_wandb_run_id(config)
    use_wandb, wandb_run = engine.setup_wandb(config)

    env = instantiate_training_env(config)
    validation_env = instantiate_validation_env(config, training_env=env)
    agent_conf = PPOJax.init_agent_conf(env, config)
    n_devices = device_count(config)

    trajectory_data_shards = prepare_trajectory_data_shards(env, config, n_devices)
    validation_shares_training = validation_env is None or validation_env.th.traj is env.th.traj
    if validation_env is not None and validation_shares_training:
        logger.info("PPO validation shares immutable training trajectory arrays.")
    validation_trajectory_data_shards = prepare_validation_trajectory_data_shards(
        validation_env,
        config,
        n_devices,
        shares_training=validation_shares_training,
    )
    release_unused_host_memory()

    metrics_env = validation_env if validation_env is not None else env
    metrics_handler = engine.build_metrics_handler(config, metrics_env)

    from hydra.core.hydra_config import HydraConfig

    hydra_runtime = HydraConfig.get().runtime
    result_dir = hydra_runtime.output_dir
    recorder = hooks.build_video_recorder(result_dir=result_dir, config=config)
    hooks._video_recorder = recorder
    logging_callback = engine.build_logging_callback(
        env,
        config,
        agent_conf,
        use_wandb,
        hooks,
        algorithm_cls=PPOJax,
    )

    experiment_hash = config_hash(config.experiment)
    experiment_id = config.experiment.get("run_id") or experiment_hash
    auto_resume = bool(config.experiment.get("auto_resume", True))
    explicit_resume = config.experiment.get("resume_from")
    checkpoint_dir = resolve_checkpoint_dir(
        configured_ckpt_dir=config.experiment.get("checkpoint_dir", "checkpoints") or "checkpoints",
        launch_dir=hydra_runtime.cwd,
        result_dir=result_dir,
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
            _require_compatible_local_resume(checkpoint_dir, experiment_hash)
        else:
            _require_compatible_existing_local_manifest(checkpoint_dir, experiment_hash)
            if explicit_resume:
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

    write_manifest(
        checkpoint_dir,
        config.experiment,
        experiment_hash,
        require_full_git_sha=True,
    )
    logger.info("PPO checkpoint directory: %s", checkpoint_dir)
    logger.info(
        "PPO topology: devices=%d global_envs=%d local_envs=%d",
        n_devices,
        int(config.experiment.num_envs),
        int(config.experiment.num_envs) // n_devices,
    )

    resume_state = None
    if resume_from is None:
        training = PPOJax.build_training(
            env,
            agent_conf,
            metrics_handler=metrics_handler,
            val_env=validation_env,
        )
    else:
        exact_resume = bool(config.experiment.get("exact_resume", False))
        runtime_state_target = None
        if exact_resume:
            runtime_state_target = PPOJax.build_runtime_state_restore_target(
                compute_training_rngs(config),
                env,
                agent_conf,
                local_num_envs=int(config.experiment.num_envs) // n_devices,
            )
        training, resume_state = PPOJax.build_resumed_training(
            env,
            agent_conf,
            resume_from,
            metrics_handler=metrics_handler,
            val_env=validation_env,
            apply_resume_resets=not exact_resume,
            preserve_checkpoint_target=using_local_auto_resume,
            runtime_state_target=runtime_state_target,
        )

    try:
        return run_training(
            training,
            compute_training_rngs(config),
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


__all__ = ["run_ppo_experiment"]
