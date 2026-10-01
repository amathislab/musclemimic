from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf

from musclemimic.utils.logging import setup_logger

from .checkpoint_manager import (
    CheckpointMetadata,
    create_checkpoint_manager,
)

logger = setup_logger(__name__)


@dataclass
class JaxCheckpointHookConfig:
    checkpoint_dir: str
    checkpoint_format: str = "auto"
    max_checkpoints_to_keep: int = 5
    num_steps: int = 1
    num_envs: int = 1
    async_checkpointing: bool = True


def _extract_hook_config(exp_cfg) -> JaxCheckpointHookConfig:
    return JaxCheckpointHookConfig(
        checkpoint_dir=getattr(exp_cfg, "checkpoint_dir", "checkpoints") or "checkpoints",
        checkpoint_format=getattr(exp_cfg, "checkpoint_format", "auto"),
        max_checkpoints_to_keep=int(getattr(exp_cfg, "max_checkpoints_to_keep", 5) or 5),
        num_steps=int(getattr(exp_cfg, "num_steps", 1) or 1),
        num_envs=int(getattr(exp_cfg, "num_envs", 1) or 1),
        async_checkpointing=bool(getattr(exp_cfg, "async_checkpointing", True)),
    )


def create_jax_checkpoint_host_callback(
    algorithm_cls: Any,
    agent_conf: Any,
    exp_cfg: Any,
    env: Any,
    base_global_timestep: int = 0,
    base_completed_updates: int = 0,
    target_global_timestep: int = 0,
) -> tuple[Callable, Any]:
    """Create a host callback suitable for jax.experimental.io_callback.

    Parameters
    ----------
    algorithm_cls : Algorithm class providing _agent_state dataclass.
    agent_conf : Algorithm agent configuration instance.
    exp_cfg : experiment config (OmegaConf node).
    env : environment instance (for metadata: backend/env_name).

    Returns
    -------
    (callback_fn, checkpoint_manager)
      callback_fn has signature
      (params, run_stats, opt_state, step, updates_done, rng_key, current_lr) -> int32(0)
      checkpoint_manager must be closed by caller after training.
    """

    # exp_cfg may either be the root config (with .experiment) or already the experiment node
    if hasattr(exp_cfg, "experiment") and hasattr(exp_cfg.experiment, "checkpoint_dir"):
        exp_node = exp_cfg.experiment
    else:
        exp_node = exp_cfg
    hook_cfg = _extract_hook_config(exp_node)
    manager = create_checkpoint_manager(
        hook_cfg.checkpoint_dir,
        format=hook_cfg.checkpoint_format,
        max_to_keep=hook_cfg.max_checkpoints_to_keep,
        async_save=hook_cfg.async_checkpointing,
    )

    # Pre-resolve config snapshot once (cheap) for pickle; Orbax will store JSON separately.
    # (Config snapshot kept for potential future extension; not needed directly here)
    try:
        OmegaConf.to_container(agent_conf.config, resolve=True, throw_on_missing=False)
    except Exception:
        pass

    backend = getattr(env, "mjx_backend", "jax")
    env_name = getattr(env, "__class__", type(env)).__name__

    base_global_timestep = int(base_global_timestep)
    base_completed_updates = int(base_completed_updates)
    target_global_timestep = int(target_global_timestep)

    def _host_cb(ts_params, ts_run_stats, ts_opt_state, ts_step, updates_done, rng_key, current_lr):
        """Host-side checkpoint save (invoked via io_callback)."""
        from musclemimic.algorithms import TrainState  # local import to avoid cycles

        from .checkpoint_utils import TrainingConfig, compute_checkpoint_metadata

        # Rebuild TrainState
        train_state = TrainState(
            apply_fn=agent_conf.network,
            params=ts_params,
            tx=agent_conf.tx,
            opt_state=ts_opt_state,
            step=int(ts_step),
            run_stats=ts_run_stats,
        )
        agent_state = algorithm_cls._agent_state(train_state=train_state)  # type: ignore

        # Extract training config
        try:
            config = TrainingConfig.from_experiment_config(agent_conf.config.experiment)
        except Exception as e:
            raise RuntimeError(f"Failed to extract training config for checkpoint: {e}") from e

        # Compute metadata using shared utility.
        # For resume runs (especially when LR schedule step counter is reset),
        # metadata update_number must come from `updates_done`. global_timestep is
        # reconstructed here on host from resume baselines to avoid JAX int64 issues.
        effective_update_number = max(0, int(updates_done))
        updates_since_resume = max(0, effective_update_number - base_completed_updates)
        effective_global_timestep = (
            base_global_timestep + updates_since_resume * config.num_steps * config.num_envs
        )
        metadata_dict = compute_checkpoint_metadata(
            optimizer_step=int(ts_step), config=config, learning_rate=float(current_lr)
        )
        metadata_dict["update_number"] = effective_update_number
        metadata_dict["global_timestep"] = effective_global_timestep

        # Create metadata object
        md = CheckpointMetadata(
            **metadata_dict,
            target_global_timestep=target_global_timestep,
            backend=backend,
            env_name=env_name,
        )

        # Save checkpoint (use update_number as directory name)
        path = manager.save_checkpoint(md.update_number, agent_conf, agent_state, md)
        mode = "async" if hook_cfg.async_checkpointing else "sync"
        logger.info(f"Saved checkpoint ({mode}) at update {md.update_number}: {path}")
        logger.info(f"  Global timestep: {md.global_timestep:,}")
        return 0  # dummy return to satisfy io_callback contract

    return _host_cb, manager


def create_jax_pytree_checkpoint_host_callback(
    algorithm_cls: Any,
    agent_conf: Any,
    exp_cfg: Any,
    env: Any,
    base_global_timestep: int = 0,
    base_completed_updates: int = 0,
    target_global_timestep: int = 0,
) -> tuple[Callable, Any]:
    """Create a checkpoint callback for algorithms with custom train-state pytrees.

    PPO uses :class:`TrainState`, while off-policy algorithms own several
    optimizers, target networks, and a replay buffer. This callback preserves
    that complete pytree without translating it into PPO fields.
    """
    if hasattr(exp_cfg, "experiment") and hasattr(exp_cfg.experiment, "checkpoint_dir"):
        exp_node = exp_cfg.experiment
    else:
        exp_node = exp_cfg
    hook_cfg = _extract_hook_config(exp_node)
    manager = create_checkpoint_manager(
        hook_cfg.checkpoint_dir,
        format=hook_cfg.checkpoint_format,
        max_to_keep=hook_cfg.max_checkpoints_to_keep,
        async_save=hook_cfg.async_checkpointing,
    )

    backend = getattr(env, "mjx_backend", "jax")
    env_name = getattr(env, "__class__", type(env)).__name__
    base_global_timestep = int(base_global_timestep)
    base_completed_updates = int(base_completed_updates)
    target_global_timestep = int(target_global_timestep)

    def _host_cb(train_state, optimizer_step, updates_done, _rng_key, current_lr):
        from .checkpoint_utils import TrainingConfig, compute_checkpoint_metadata

        agent_state = algorithm_cls._agent_state(train_state=train_state)
        config = TrainingConfig.from_experiment_config(agent_conf.config.experiment)
        effective_update_number = max(0, int(updates_done))
        updates_since_resume = max(0, effective_update_number - base_completed_updates)
        global_timestep = base_global_timestep + updates_since_resume * config.num_steps * config.num_envs
        metadata_dict = compute_checkpoint_metadata(
            optimizer_step=int(optimizer_step),
            config=config,
            learning_rate=float(current_lr),
        )
        metadata_dict["update_number"] = effective_update_number
        metadata_dict["global_timestep"] = global_timestep
        metadata = CheckpointMetadata(
            **metadata_dict,
            target_global_timestep=target_global_timestep,
            algo_version=f"{algorithm_cls.__name__}_v1",
            backend=backend,
            env_name=env_name,
        )
        path = manager.save_checkpoint(effective_update_number, agent_conf, agent_state, metadata)
        mode = "async" if hook_cfg.async_checkpointing else "sync"
        logger.info(f"Saved checkpoint ({mode}) at update {effective_update_number}: {path}")
        logger.info(f"  Global timestep: {global_timestep:,}")
        return 0

    return _host_cb, manager
