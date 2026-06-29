from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf, open_dict

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms.common.checkpoint_hooks import create_jax_checkpoint_host_callback
from musclemimic.utils.logging import setup_logger
from musclemimic.utils.metrics import MetricsHandler

from .checkpointing import (
    config_hash,
    find_latest_checkpoint,
    resolve_checkpoint_dir,
    resume_or_fresh,
    validate_checkpoint_compatibility,
    write_manifest,
)
from .logging import ExperimentHooks

logger = setup_logger(__name__)


def setup_jax_cache() -> None:
    cache_dir = os.path.join(Path.home(), ".musclemimic", ".jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    logger.info(f"JAX compilation cache enabled at: {cache_dir}")


def setup_wandb(config) -> tuple[bool, Any]:
    import wandb

    use_wandb = config.wandb.get("mode", "online") != "disabled"
    if not use_wandb:
        logger.info("Wandb logging disabled")
        return False, None
    wandb.login()
    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    params = {"project": config.wandb.project, "config": config_dict}
    if "tags" in config.wandb and config.wandb.tags:
        params["tags"] = config.wandb.tags
    run = wandb.init(**params)
    return True, run


# Dataset config keys that may appear in validation config
_VAL_DATASET_CONF_KEYS = ("amass_dataset_conf", "c3d_dataset_conf", "lafan1_dataset_conf", "custom_dataset_conf")


def _has_validation_dataset_override(val_cfg) -> bool:
    """Check if validation config has its own dataset configuration."""
    if val_cfg is None:
        return False
    for key in _VAL_DATASET_CONF_KEYS:
        if val_cfg.get(key, None) is not None:
            return True
    # Also check nested validation.task_factory.params
    val_task = val_cfg.get("task_factory", None)
    if val_task is None:
        return False
    if OmegaConf.is_config(val_task):
        val_task = OmegaConf.to_container(val_task, resolve=True)
    if isinstance(val_task, dict):
        params = val_task.get("params")
        if OmegaConf.is_config(params):
            params = OmegaConf.to_container(params, resolve=True)
        if isinstance(params, dict):
            for key in _VAL_DATASET_CONF_KEYS:
                if params.get(key, None) is not None:
                    return True
    return False


def _can_share_trajectory_handler(config) -> bool:
    """Check if validation env can share trajectory with training env."""
    val_cfg = config.experiment.get("validation", None)
    if not val_cfg:
        return False
    if not val_cfg.get("share_trajectory_handler", True):
        return False
    if _has_validation_dataset_override(val_cfg):
        return False
    return True


def _maybe_share_validation_trajectory(env, val_env, config) -> None:
    """Share trajectory data from training env to validation env if possible."""
    if not _can_share_trajectory_handler(config):
        if val_env is not None and getattr(val_env, "th", None) is not None:
            logger.info("Using separate validation trajectory (share disabled or dataset override).")
        return
    if env is None or val_env is None:
        return
    if getattr(env, "th", None) is None:
        return

    # val_env.th is None when we skipped loading - create handler using shared trajectory
    if getattr(val_env, "th", None) is None:
        val_env.load_trajectory(traj=env.th.traj, warn=False)
        # Convert to JAX if training env's trajectory is in JAX
        if getattr(env, "mjx_enabled", False) and not env.th.is_numpy and val_env.th.is_numpy:
            val_env.th.to_jax()
        logger.info("Created validation trajectory handler using shared trajectory data.")
        return

    # val_env already has th - just share the trajectory data reference
    if env.th.traj is val_env.th.traj:
        logger.info("Validation trajectory data already shared.")
        return
    val_env.th.traj = env.th.traj
    val_env.th._is_numpy = env.th.is_numpy
    if hasattr(val_env, "_finalize_traj_load"):
        val_env._finalize_traj_load()
    logger.info("Sharing validation trajectory data with training env.")


def instantiate_env(config) -> Any:
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

    # Convert trajectory to JAX before JIT compilation. This must happen here
    # because mjx_reset() runs inside JIT-traced functions. Calling to_jax()
    # inside JIT triggers TracerBoolConversionError due to Python control flow
    # in TrajectoryInfo.__post_init__.
    if getattr(env, "mjx_enabled", False) and getattr(env, "th", None) is not None and env.th.is_numpy:
        env.th.to_jax()

    return env


def instantiate_validation_env(config, share_trajectory: bool = False) -> Any:
    """Create a separate validation environment with validation-specific terminal state handler.

    Args:
        config: Experiment configuration.
        share_trajectory: If True, skip trajectory loading (will share from training env later).
    """
    if not config.experiment.get("validation", {}).get("active", False):
        return None

    # Get validation terminal state type, default to NoTerminalStateHandler
    val_terminal_state_type = config.experiment.validation.get("terminal_state_type", "NoTerminalStateHandler")

    # Create a copy of env_params with validation-specific settings
    val_env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)

    # Override terminal state handler for validation
    val_env_params["terminal_state_type"] = val_terminal_state_type
    val_env_params["terminal_state_params"] = config.experiment.validation.get("terminal_state_params", {})

    # Optional validation-specific initial state handler.
    val_init_state_type = config.experiment.validation.get("init_state_type", None)
    val_init_state_params = config.experiment.validation.get("init_state_params", None)
    if val_init_state_type is not None:
        val_env_params["init_state_type"] = val_init_state_type
        val_env_params["init_state_params"] = (
            OmegaConf.to_container(val_init_state_params, resolve=True) if val_init_state_params is not None else {}
        )

    # Start from beginning of each trajectory (random trajectory, but step 0)
    if config.experiment.validation.get("start_from_beginning", False):
        if "th_params" not in val_env_params:
            val_env_params["th_params"] = {}
        val_env_params["th_params"]["start_from_random_step"] = False

    # Use validation num_envs if specified
    val_env_params["num_envs"] = config.experiment.validation.get("num_envs", config.experiment.env_params.num_envs)

    # Override th_params for validation (e.g., random_start: false for deterministic eval)
    val_th_params = config.experiment.validation.get("th_params", None)
    if val_th_params is not None:
        base_th_params = val_env_params.get("th_params", {}) or {}
        val_env_params["th_params"] = {**base_th_params, **OmegaConf.to_container(val_th_params, resolve=True)}

    # Prepare task_factory params, optionally skipping trajectory loading if sharing
    task_factory_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    if share_trajectory:
        for key in _VAL_DATASET_CONF_KEYS:
            task_factory_params.pop(key, None)
        logger.info("Skipping validation trajectory load (will share from training env).")
    else:
        # Apply validation-specific dataset overrides if present
        val_cfg = config.experiment.validation
        for key in _VAL_DATASET_CONF_KEYS:
            val_dataset = val_cfg.get(key, None)
            if val_dataset is not None:
                task_factory_params[key] = (
                    OmegaConf.to_container(val_dataset, resolve=True)
                    if OmegaConf.is_config(val_dataset)
                    else val_dataset
                )
                logger.info(f"Using validation-specific {key}.")

    # Create validation environment
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    val_env = factory.make(**val_env_params, **task_factory_params)

    # Convert trajectory to JAX before JIT (see instantiate_env for details)
    if getattr(val_env, "mjx_enabled", False) and getattr(val_env, "th", None) is not None and val_env.th.is_numpy:
        val_env.th.to_jax()

    return val_env


def pick_algorithm(config) -> Any:
    from musclemimic.algorithms import PPOJax

    algorithm_name = config.experiment.get("algorithm", "PPOJax")
    if algorithm_name == "PPOJax":
        return PPOJax
    raise ValueError(f"Unknown algorithm: {algorithm_name}")


def build_agent_conf(algorithm_cls, env, config):
    return algorithm_cls.init_agent_conf(env, config)


def build_metrics_handler(config, env):
    active = getattr(config.experiment, "validation", {}).get("active", False)
    return MetricsHandler(config, env) if active else None


def build_logging_callback(env, config, agent_conf, use_wandb, hooks: ExperimentHooks):
    import wandb as _wandb

    algorithm_name = config.experiment.get("algorithm", "PPOJax")

    def _cb(metrics_dict: dict[str, Any]):
        if metrics_dict.get("_log_only", False):
            log_dict: dict[str, Any] = {}
            for k, v in metrics_dict.items():
                if k.startswith("model/"):
                    label = k.split("/", 1)[1]
                    log_dict[f"Model/{label}"] = v
            if log_dict:
                if use_wandb:
                    _wandb.log(log_dict, step=0)
                else:
                    logger.info(log_dict)
            return

        # Determine which timestep to use for logging
        # reset_logging_timestep=True: Start from 0
        # reset_logging_timestep=False: Continue from checkpoint (resume the
        # timestep logging as well)
        reset_logging = config.experiment.get("reset_logging_timestep", False)

        if reset_logging:
            # Use timestep since resume start (jax_raw_timestep)
            current_timestep = int(metrics_dict.get("jax_raw_timestep", 0.0))
        else:
            # Use global timestep including checkpoint offset (max_timestep)
            current_timestep = int(metrics_dict.get("max_timestep", metrics_dict.get("jax_raw_timestep", 0.0)))

        jax_timestep = metrics_dict.get("jax_raw_timestep", metrics_dict.get("max_timestep", 0.0))
        log_dict: dict[str, Any] = {
            "Mean Episode Return": metrics_dict.get("mean_episode_return", 0.0),
            "Mean Episode Length": metrics_dict.get("mean_episode_length", 0.0),
            "Learning Rate": metrics_dict.get("learning_rate", 0.0),
            "Current Timestep": current_timestep,
            "Raw JAX Timestep": int(jax_timestep),
        }

        # Pass through all prefixed metrics (ppo/, reward/, adaptive/, etc.)
        # Define display names for metric prefixes
        prefix_display = {"ppo": "PPO"}
        for k, v in metrics_dict.items():
            if "/" in k:
                prefix, suffix = k.split("/", 1)
                display_prefix = prefix_display.get(prefix, prefix)
                log_dict[f"{display_prefix}/{suffix}"] = v

        # Forward validation metrics from ValidationSummary.
        if metrics_dict.get("has_validation_update", False):
            for k, v in metrics_dict.items():
                if not k.startswith("val_"):
                    continue
                key_body = k[len("val_") :]
                # Summary metrics.
                if key_body == "mean_episode_return":
                    log_dict["Validation/Mean Episode Return"] = v
                    continue
                if key_body == "mean_episode_length":
                    log_dict["Validation/Mean Episode Length"] = v
                    continue
                # Detailed validation measures.
                log_dict[f"Validation Measures/{key_body}"] = v

            # Sweep metric for HPO: sum Euclidean site measures
            s_rpos = metrics_dict.get("val_euclidean_distance_site_rpos", None)
            s_rrot = metrics_dict.get("val_euclidean_distance_site_rrotvec", None)
            s_rvel = metrics_dict.get("val_euclidean_distance_site_rvel", None)
            if s_rpos is not None and s_rrot is not None and s_rvel is not None:
                sweep_metric = float(s_rpos) + float(s_rrot) + float(s_rvel)
                log_dict["Metric for Sweep"] = sweep_metric
                logger.info(f"Combined sweep metric: {sweep_metric:.4f}")

        # Experiment-specific log enrichment.
        hooks.enrich_log(log_dict, metrics_dict, env)

        if use_wandb:
            _wandb.log(log_dict, step=current_timestep)

        # Trigger validation video recording.
        if (
            algorithm_name == "PPOJax"
            and metrics_dict.get("has_validation_update", False)
            and "_train_params" in metrics_dict
        ):
            # Build a temporary agent state from the current params.
            from musclemimic.algorithms import PPOJax, TrainState

            cur = metrics_dict["_train_params"]
            temp_state = TrainState(
                apply_fn=agent_conf.network.apply,
                tx=agent_conf.tx,
                params=cur["params"],
                run_stats=cur["run_stats"],
                opt_state=None,
                step=0,
            )
            temp_agent_state = PPOJax._agent_state(train_state=temp_state)
            recorder = getattr(hooks, "_video_recorder", None)
            if recorder is not None:
                try:
                    video_path = recorder.record_episode(
                        agent_conf=agent_conf,
                        agent_state=temp_agent_state,
                        validation_number=getattr(hooks, "_validation_counter", 0) + 1,
                        timestep=current_timestep,
                    )
                    hooks._validation_counter = getattr(hooks, "_validation_counter", 0) + 1
                    if video_path:
                        logger.info(f"Validation video recorded: {video_path}")
                    hooks.on_validation_video(use_wandb, _wandb, video_path, current_timestep)
                except Exception as e:
                    # Video failures should not interrupt training.
                    logger.warning(f"Video recording failed: {e}")

    return _cb


def compute_training_rngs(config):
    n_seeds = int(config.experiment.get("n_seeds", 1))
    if "seeds" in config.experiment and config.experiment.seeds is not None:
        seeds = list(config.experiment.seeds)
        if len(seeds) != n_seeds:
            raise ValueError(f"Length of seeds ({len(seeds)}) must match n_seeds ({n_seeds})")
    else:
        seeds = list(range(n_seeds))
    keys = [jax.random.PRNGKey(int(s)) for s in seeds]
    return jnp.squeeze(jnp.vstack(keys)) if len(keys) > 1 else keys[0]


def build_train_fn(algorithm_cls, env, agent_conf, mh, logging_cb, logging_interval=1, val_env=None):
    return algorithm_cls.build_train_fn(
        env,
        agent_conf,
        mh=mh,
        online_logging_callback=logging_cb,
        logging_interval=logging_interval,
        val_env=val_env,
    )


def run_training(train_fn, rngs):
    # vmap if multiple seeds
    if hasattr(rngs, "ndim") and rngs.ndim > 1:  # jnp array
        train_fn = jax.jit(jax.vmap(train_fn))
    else:
        train_fn = jax.jit(train_fn)
    logger.info("Starting training...")
    return train_fn(rngs)


def _generate_run_suffix() -> str:
    """Return a per-run unique suffix for checkpoint directories."""
    ts = datetime.now(UTC).strftime("%y%m%dT%H%M%S")
    pid = os.getpid()
    job_num = os.environ.get("HYDRA_JOB_NUM")
    job_part = f"-job{job_num}" if job_num is not None else ""
    return f"{ts}-pid{pid}{job_part}-{uuid.uuid4().hex[:6]}"


def run_experiment(config, hooks: ExperimentHooks):
    # XLA flags
    os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

    setup_jax_cache()

    # Wandb
    use_wandb, run = setup_wandb(config)

    # Env and algo - share trajectory if possible to save memory
    env = instantiate_env(config)
    can_share = _can_share_trajectory_handler(config) and getattr(env, "th", None) is not None
    val_env = instantiate_validation_env(config, share_trajectory=can_share)
    _maybe_share_validation_trajectory(env, val_env, config)
    # NOTE: Wrapping is now handled entirely by algorithm._wrap_env methods
    # to avoid conflicts and ensure correct wrapper ordering
    algorithm_cls = pick_algorithm(config)
    agent_conf = build_agent_conf(algorithm_cls, env, config)

    # Report total motion duration (post-concatenation), before training starts
    for label, _env in [("Train", env), ("Validation", val_env)]:
        if (
            _env is not None
            and hasattr(_env, "th")
            and _env.th is not None
            and hasattr(_env.th, "traj")
            and _env.th.traj is not None
            and hasattr(_env.th.traj, "info")
            and hasattr(_env.th.traj, "data")
            and hasattr(_env.th.traj.info, "frequency")
            and hasattr(_env.th.traj.data, "qpos")
        ):
            freq = float(_env.th.traj.info.frequency)
            frames = int(_env.th.traj.data.qpos.shape[0])
            if freq > 0.0 and frames > 0:
                duration_sec = (frames - 1) / freq
                duration_min = duration_sec / 60.0
                logger.info(
                    f"[{label}] total concatenated motion duration: {duration_min:.2f} minutes "
                    f"({duration_sec:.1f} s, {frames} frames @ {freq:.1f} Hz)"
                )

    # Metrics + hooks + video
    metrics_env = val_env if val_env is not None else env
    mh = build_metrics_handler(config, metrics_env)
    # Build recorder with Hydra output dir
    from hydra.core.hydra_config import HydraConfig

    result_dir = HydraConfig.get().runtime.output_dir
    recorder = hooks.build_video_recorder(result_dir=result_dir, config=config)
    hooks._video_recorder = recorder

    # Logging callback
    logging_cb = build_logging_callback(env, config, agent_conf, use_wandb, hooks)

    # Checkpoint resume or fresh with auto-resume support
    explicit_resume = getattr(config.experiment, "resume_from", None)
    auto_resume = getattr(config.experiment, "auto_resume", True)
    run_id = getattr(config.experiment, "run_id", None)
    checkpoint_root = getattr(config.experiment, "checkpoint_root", None)

    exp_config_hash = config_hash(config.experiment)
    experiment_id = run_id or exp_config_hash

    # Resolve checkpoint directory path
    configured_ckpt_dir = getattr(config.experiment, "checkpoint_dir", "checkpoints") or "checkpoints"
    launch_dir = HydraConfig.get().runtime.cwd

    resolved_ckpt_dir = resolve_checkpoint_dir(
        configured_ckpt_dir=configured_ckpt_dir,
        launch_dir=launch_dir,
        result_dir=result_dir,
        experiment_id=experiment_id,
        auto_resume=auto_resume,
        checkpoint_root=checkpoint_root,
    )
    if not auto_resume:
        resolved_ckpt_dir = os.path.join(resolved_ckpt_dir, _generate_run_suffix())

    with open_dict(config.experiment):
        config.experiment.checkpoint_dir = resolved_ckpt_dir
    os.makedirs(resolved_ckpt_dir, exist_ok=True)

    # Determine resume path (auto-detect > explicit > fresh when auto_resume=true)
    resume_from = None
    apply_resume_resets = True
    if auto_resume:
        latest = find_latest_checkpoint(resolved_ckpt_dir)
        if latest:
            logger.info(f"Auto-resume: found checkpoint: {latest}")
            resume_from = latest
            apply_resume_resets = False
            validate_checkpoint_compatibility(resolved_ckpt_dir, exp_config_hash)
        elif explicit_resume:
            logger.info(f"Auto-resume: no local checkpoint, using explicit: {explicit_resume}")
            resume_from = explicit_resume
        else:
            logger.info(f"Auto-resume: no checkpoint in {resolved_ckpt_dir}, starting fresh")
    elif explicit_resume:
        logger.info(f"Resuming from explicit path: {explicit_resume}")
        resume_from = explicit_resume

    # Write manifest on first run (idempotent)
    write_manifest(resolved_ckpt_dir, config.experiment, exp_config_hash)

    # Update config with detected resume path for resume_or_fresh
    with open_dict(config.experiment):
        config.experiment.resume_from = resume_from

    logger.info(f"Checkpoint directory: {resolved_ckpt_dir}")
    logger.info(f"Experiment ID (config hash): {exp_config_hash}")

    train_fn = resume_or_fresh(
        env,
        agent_conf,
        algorithm_cls,
        config,
        mh,
        logging_cb,
        logging_interval=config.experiment.get("online_logging_interval", 1),
        val_env=val_env,
        apply_resume_resets=apply_resume_resets,
    )

    # Seeds and training
    rngs = compute_training_rngs(config)
    run_training(train_fn, rngs)

    # Close any cached checkpoint manager created during training (host-side cleanup)
    cache_entry = getattr(create_jax_checkpoint_host_callback, "__cached_instance__", None)
    if cache_entry is not None:
        ckpt_manager = cache_entry[2]  # (cache_key, ckpt_cb, ckpt_mgr)
        ckpt_manager.close()
        delattr(create_jax_checkpoint_host_callback, "__cached_instance__")

    if use_wandb and run is not None:
        import wandb as _wandb

        _wandb.finish()
