"""Run JAX training and host-side logging and checkpoints."""

from __future__ import annotations

import ctypes
import gc
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.algorithms.common.checkpoint_utils import optimizer_step_to_update
from musclemimic.algorithms.common.distributed import (
    DATA_AXIS_NAME,
    assert_warp_multi_gpu_compatible,
    device_count,
    make_data_mesh,
)
from musclemimic.algorithms.flashsac.train import FlashSACTraining
from musclemimic.algorithms.ppo.train import PPOTraining
from musclemimic.runner.checkpoint_saving import PPOCheckpointSaver
from musclemimic.runner.data_parallel import (
    DATA,
    REPLICATED,
    compile_sharded,
    data_specs,
    eval_shard_shape,
    global_data_shape,
    put_tree,
    replicated_specs,
    require_explicit_trajectory_data,
    trajectory_specs,
)
from musclemimic.runner.ppo_logging import format_ppo_log
from musclemimic.runner.ppo_sharding import (
    add_device_key_axis,
    localize_resume_state,
    remove_device_key_axis,
    resume_specs,
    runtime_specs,
    validation_reset_specs,
)
from musclemimic.runner.validation_reduction import (
    ValidationProcessExchange,
    add_validation_shard_axis,
    finish_validation,
)
from musclemimic.runner.validation_video_recorder import validation_video_due
from musclemimic.runner.warp_graph_cache import (
    clear_warp_graph_cache,
    disable_warp_graph_eviction,
)
from musclemimic.utils.logging import setup_logger
from musclemimic.utils.model import (
    count_actor_critic_params,
    log_network_architecture,
)

logger = setup_logger(__name__)


class _LogBatch(NamedTuple):
    """Metrics and update positions copied after one compiled chunk."""

    metrics: tuple[Any, Any, Any]
    validation_metrics: Any | None
    offsets: tuple[int, ...]
    base_global_timestep: int
    base_raw_timestep: int
    steps_per_rollout: int
    first_update: int
    validation_offset: int | None
    checkpoint_offset: int | None
    checkpoint_update: int


def release_unused_host_memory() -> None:
    """Release unreachable Python objects and return free glibc pages when available."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        return


def setup_jax_cache() -> None:
    """Enable JAX's persistent compilation cache."""
    cache_dir = Path.home() / ".musclemimic" / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    logger.info(f"JAX compilation cache enabled at: {cache_dir}")


def compute_training_rngs(config: Any) -> jax.Array:
    """Build the configured training PRNG key or key batch."""
    n_seeds = int(config.experiment.get("n_seeds", 1))
    configured_seeds = config.experiment.get("seeds")
    seeds = list(configured_seeds) if configured_seeds is not None else list(range(n_seeds))
    if len(seeds) != n_seeds:
        raise ValueError(f"Length of seeds ({len(seeds)}) must match n_seeds ({n_seeds})")

    keys = [jax.random.PRNGKey(int(seed)) for seed in seeds]
    return jnp.squeeze(jnp.vstack(keys)) if len(keys) > 1 else keys[0]


def _validate_exact_resume_layout(
    resume_state: Any | None,
    *,
    global_num_envs: int,
    num_devices: int,
) -> None:
    """Reject replica-shaped runtime checkpoints from the retired layout."""
    if not isinstance(resume_state, dict) or "runtime_state" not in resume_state:
        return
    runtime_state = resume_state["runtime_state"]
    last_obs = np.asarray(runtime_state["last_obs"])
    rng = np.asarray(runtime_state["rng"])
    if last_obs.ndim == 0 or last_obs.shape[0] != global_num_envs:
        raise ValueError(
            "Exact-resume state must use the global environment batch; "
            f"expected {global_num_envs}, got shape {last_obs.shape}."
        )
    if rng.ndim == 0 or rng.shape[0] != num_devices:
        raise ValueError(
            f"Exact-resume PRNG state must contain one key per device; expected {num_devices}, got shape {rng.shape}."
        )


def _model_only_resume_state(resume_state: Any | None) -> Any | None:
    if not isinstance(resume_state, dict) or "runtime_state" not in resume_state:
        return resume_state
    return {key: value for key, value in resume_state.items() if key != "runtime_state"}


def _checkpoint_due(counter: int, config: Any) -> bool:
    """Return whether this update should be saved."""
    interval = int(config.get("checkpoint_interval", 0) or 0)
    periodic = interval > 0 and counter > 0 and counter % interval == 0
    validation = bool(config.get("checkpoints_on_validation", True)) and (
        counter % int(config.validation_interval) == 0
    )
    validation_number = counter // int(config.validation_interval)
    validation_config = config.get("validation", {})
    video_frequency = int(validation_config.get("video_frequency", 0) or 0)
    video_enabled = (
        bool(validation_config.get("active", False)) and int(validation_config.get("video_length", 0) or 0) > 0
    )
    video = (
        video_enabled
        and counter % int(config.validation_interval) == 0
        and validation_video_due(
            validation_number,
            video_frequency,
        )
    )
    return periodic or validation or video


def _copy_log_batch(
    chunk_result: Any,
    validation_metrics: Any | None,
    *,
    offsets: tuple[int, ...],
    base_global_timestep: int,
    base_raw_timestep: int,
    steps_per_rollout: int,
    first_update: int,
    validation_offset: int | None,
    checkpoint_offset: int | None,
    checkpoint_update: int,
) -> _LogBatch:
    """Start copying one chunk's small metric histories to the host."""
    metrics = (
        chunk_result.training_metrics,
        chunk_result.ppo_metrics,
        chunk_result.diagnostics,
    )
    jax.copy_to_host_async((metrics, validation_metrics))
    return _LogBatch(
        metrics=metrics,
        validation_metrics=validation_metrics,
        offsets=offsets,
        base_global_timestep=base_global_timestep,
        base_raw_timestep=base_raw_timestep,
        steps_per_rollout=steps_per_rollout,
        first_update=first_update,
        validation_offset=validation_offset,
        checkpoint_offset=checkpoint_offset,
        checkpoint_update=checkpoint_update,
    )


def _write_logs(
    log_batch: _LogBatch,
    training: PPOTraining,
    logging_callback: Any,
) -> None:
    """Wait for copied metrics, then format and log selected updates."""
    metric_histories = jax.device_get(
        log_batch.metrics,
    )
    validation_metrics = None if log_batch.validation_metrics is None else jax.device_get(log_batch.validation_metrics)
    for offset in log_batch.offsets:
        metrics = jax.tree.map(
            lambda values, index=offset: values[index],
            metric_histories,
        )
        is_validation = offset == log_batch.validation_offset
        is_checkpoint = offset == log_batch.checkpoint_offset
        log = format_ppo_log(
            *metrics,
            training.config,
            global_timestep=(log_batch.base_global_timestep + (offset + 1) * log_batch.steps_per_rollout),
            raw_timestep=(log_batch.base_raw_timestep + (offset + 1) * log_batch.steps_per_rollout),
            validation_metrics=validation_metrics if is_validation else None,
            checkpoint_update=log_batch.checkpoint_update if is_checkpoint else -1,
        )
        update = log_batch.first_update + offset
        try:
            logging_callback(log)
        except Exception:
            logger.exception(
                "Online logging callback failed at update %d",
                update,
            )


def _next_chunk_size(counter: int, remaining_updates: int, config: Any) -> int:
    """Return the distance to the next validation, checkpoint, or final update."""
    validation_interval = int(config.validation_interval)
    if validation_interval < 1:
        raise ValueError("validation_interval must be positive")
    distances = [validation_interval - counter % validation_interval]
    checkpoint_interval = int(config.get("checkpoint_interval", 0) or 0)
    if checkpoint_interval > 0:
        distances.append(checkpoint_interval - counter % checkpoint_interval)
    return min(remaining_updates, *distances)


def _chunk_value(tree: Any, offset: int) -> Any:
    """Return one update from a metric history."""
    return jax.tree.map(
        lambda value: None if value is None else value[offset],
        tree,
        is_leaf=lambda value: value is None,
    )


def run_training(
    training: PPOTraining | FlashSACTraining,
    rngs: jax.Array,
    *,
    config: Any | None = None,
    trajectory_data_shards: Any | None = None,
    validation_trajectory_data_shards: Any | None = None,
    resume_agent_state: Any | None = None,
    logging_callback: Any | None = None,
    logging_interval: int = 1,
    validation_process_exchange: ValidationProcessExchange | None = None,
) -> Any:
    """Run the host coordinator for the configured JAX algorithm."""
    if isinstance(training, FlashSACTraining):
        if config is None:
            raise ValueError("FlashSAC training requires the experiment config")
        from musclemimic.runner.flashsac_runtime import run_flashsac_training

        return run_flashsac_training(
            training,
            rngs,
            config=config,
            trajectory_data_shards=trajectory_data_shards,
            validation_trajectory_data_shards=validation_trajectory_data_shards,
            resume_agent_state=resume_agent_state,
            logging_callback=logging_callback,
            logging_interval=logging_interval,
            validation_process_exchange=validation_process_exchange,
        )

    n_devices = 1 if config is None else device_count(config)
    uses_warp_backend = getattr(training.env, "mjx_backend", None) == "warp"
    if (
        training.config.get("validation", {}).get("active", False)
        and jax.process_count() > 1
        and validation_process_exchange is None
    ):
        raise ValueError("Multi-host PPO validation requires a host metric exchange.")

    if rngs.ndim > 1:
        raise NotImplementedError(
            "Multi-seed training (n_seeds > 1) is not supported. Run each seed as a separate job."
        )

    if validation_trajectory_data_shards is not None and trajectory_data_shards is None:
        raise ValueError("Validation trajectory shards require training trajectory shards.")
    require_explicit_trajectory_data(
        training.env,
        trajectory_data_shards,
        label="Training",
    )
    validation_env = getattr(training, "val_env", None)
    if validation_env is not None and not training.validation_shares_training_trajectory:
        require_explicit_trajectory_data(
            validation_env,
            validation_trajectory_data_shards,
            label="Validation",
        )
    if logging_interval < 1:
        raise ValueError("logging_interval must be positive")

    devices = jax.local_devices()[:n_devices]
    mesh = make_data_mesh(n_devices, devices=devices)
    if n_devices > 1:
        assert_warp_multi_gpu_compatible()
        if uses_warp_backend:
            disable_warp_graph_eviction()
    logger.info(
        "Starting data-parallel training: n_devices=%d axis_name=%s",
        n_devices,
        DATA_AXIS_NAME,
    )

    train_trajectory_specs = trajectory_specs(trajectory_data_shards)
    validation_trajectory_specs = trajectory_specs(
        validation_trajectory_data_shards,
    )

    global_num_envs = int(getattr(training.config, "global_num_envs", training.config.num_envs))
    _validate_exact_resume_layout(
        resume_agent_state,
        global_num_envs=global_num_envs,
        num_devices=n_devices,
    )
    init_rng = jax.device_put(
        rngs,
        jax.sharding.NamedSharding(mesh, REPLICATED),
    )

    def initialize_shard(rng, agent_state_data, trajectory_data_override):
        runtime_state = training.initialize(
            rng,
            localize_resume_state(agent_state_data),
            trajectory_data_override,
        )
        return add_device_key_axis(runtime_state)

    shape_resume_state = _model_only_resume_state(resume_agent_state)
    shape_resume_specs = replicated_specs(shape_resume_state)
    local_runtime_template = eval_shard_shape(
        initialize_shard,
        mesh=mesh,
        in_specs=(REPLICATED, shape_resume_specs, train_trajectory_specs),
        args=(
            init_rng,
            put_tree(shape_resume_state, shape_resume_specs, mesh),
            trajectory_data_shards,
        ),
    )
    state_specs = runtime_specs(local_runtime_template)
    init_resume_specs = resume_specs(resume_agent_state, state_specs)
    placed_resume_state = put_tree(
        resume_agent_state,
        init_resume_specs,
        mesh,
    )

    compiled_initialize = compile_sharded(
        initialize_shard,
        mesh=mesh,
        in_specs=(REPLICATED, init_resume_specs, train_trajectory_specs),
        out_specs=state_specs,
        donate_argnums=(1,) if resume_agent_state is not None else (),
    )

    runtime_state = compiled_initialize(
        init_rng,
        placed_resume_state,
        trajectory_data_shards,
    )
    progress = training.progress(
        runtime_state,
        resumed=resume_agent_state is not None,
    )
    _log_model_info(runtime_state, logging_callback)

    if progress.remaining_updates == 0:
        _log_completed_run(training, progress, logging_callback)
        return {
            "agent_state": training.agent_state_cls(train_state=runtime_state.train_state),
            "training_metrics": None,
            "validation_metrics": None,
        }

    saver = None
    if bool(training.config.get("save_checkpoints", False)):
        saver = PPOCheckpointSaver(
            training.agent_conf,
            training.env,
            base_global_timestep=progress.base_global_timestep,
            base_completed_updates=progress.completed_updates,
            target_global_timestep=progress.target_global_timestep,
        )

    steps_per_rollout = int(training.config.num_steps) * int(
        getattr(training.config, "global_num_envs", training.config.num_envs)
    )
    current_counter = int(
        optimizer_step_to_update(
            progress.base_optimizer_step,
            int(training.config.num_minibatches),
            int(training.config.update_epochs),
        )
    )
    logger.info(
        "Training updates: completed=%d remaining=%d counter=%d",
        progress.completed_updates,
        progress.remaining_updates,
        current_counter,
    )

    max_chunk_size = int(training.config.validation_interval)

    def update_chunk_shard(runtime_state, trajectory_data_override, num_updates):
        result = training.update_chunk(
            remove_device_key_axis(runtime_state),
            trajectory_data_override,
            num_updates,
            max_chunk_size,
        )
        return result._replace(
            runtime_state=add_device_key_axis(result.runtime_state),
        )

    update_in_specs = (state_specs, train_trajectory_specs, REPLICATED)
    one_update = jax.device_put(
        jnp.asarray(1, dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, REPLICATED),
    )
    local_chunk_template = eval_shard_shape(
        update_chunk_shard,
        mesh=mesh,
        in_specs=update_in_specs,
        args=(runtime_state, trajectory_data_shards, one_update),
    )
    reset_specs = validation_reset_specs(training)
    local_reset_template = local_chunk_template.validation_reset_weights
    if reset_specs is not None and local_reset_template is None:
        shard_spec = training.validation_shard_spec or training.shard_spec
        if shard_spec is None:
            raise ValueError("validation reset weights require a trajectory shard")
        local_reset_template = jax.ShapeDtypeStruct(
            (shard_spec.n_global,),
            jnp.float32,
        )

    chunk_out_specs = local_chunk_template._replace(
        runtime_state=state_specs,
        training_metrics=replicated_specs(
            local_chunk_template.training_metrics,
        ),
        ppo_metrics=replicated_specs(local_chunk_template.ppo_metrics),
        diagnostics=replicated_specs(local_chunk_template.diagnostics),
        validation_reset_weights=reset_specs,
    )

    compiled_update_chunk = compile_sharded(
        update_chunk_shard,
        mesh=mesh,
        in_specs=update_in_specs,
        out_specs=chunk_out_specs,
        donate_argnums=(0,),
    )

    def validation_shard(
        train_state,
        validation_rng,
        reset_weights,
        trajectory_data_override,
        validation_trajectory_data_override,
    ):
        result = training.validate(
            train_state,
            jnp.squeeze(validation_rng, axis=0),
            reset_weights,
            trajectory_data_override,
            validation_trajectory_data_override,
        )
        return result._replace(
            metrics=add_validation_shard_axis(result.metrics),
            rng=jnp.expand_dims(result.rng, axis=0),
        )

    validation_in_specs = (
        state_specs.train_state,
        state_specs.val_rng,
        reset_specs,
        train_trajectory_specs,
        validation_trajectory_specs,
    )
    reset_shape_input = (
        global_data_shape(local_reset_template, n_devices) if reset_specs == DATA else local_reset_template
    )
    local_validation_template = eval_shard_shape(
        validation_shard,
        mesh=mesh,
        in_specs=validation_in_specs,
        args=(
            runtime_state.train_state,
            runtime_state.val_rng,
            reset_shape_input,
            trajectory_data_shards,
            validation_trajectory_data_shards,
        ),
    )
    evaluate_all = bool(training.config.get("validation", {}).get("evaluate_all", False))
    validation_out_specs = local_validation_template._replace(
        metrics=data_specs(local_validation_template.metrics),
        rng=DATA,
    )

    compiled_validation = compile_sharded(
        validation_shard,
        mesh=mesh,
        in_specs=validation_in_specs,
        out_specs=validation_out_specs,
        donate_argnums=(1,),
    )

    last_training_metrics = None
    last_validation_metrics = None
    executed_updates = 0
    remaining_updates = progress.remaining_updates
    logger.info("PPO update chunk: up to %d updates", max_chunk_size)
    log_worker = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppo-log") if logging_callback is not None else None
    )
    pending_log: Future[None] | None = None
    try:
        while remaining_updates:
            chunk_size = _next_chunk_size(
                current_counter,
                remaining_updates,
                training.config,
            )
            chunk_start_counter = current_counter
            chunk_start_executed = executed_updates
            chunk_result = compiled_update_chunk(
                runtime_state,
                trajectory_data_shards,
                jax.device_put(
                    jnp.asarray(chunk_size, dtype=jnp.int32),
                    jax.sharding.NamedSharding(mesh, REPLICATED),
                ),
            )
            runtime_state = chunk_result.runtime_state

            # JAX dispatch is asynchronous. Submit the next chunk before
            # waiting for the preceding host log so device execution overlaps
            # metrics transfer, formatting, and logger I/O.
            if pending_log is not None:
                pending_log.result()
                pending_log = None

            current_counter += chunk_size
            executed_updates += chunk_size
            remaining_updates -= chunk_size
            updates_done = progress.completed_updates + executed_updates
            validation_due = current_counter % int(training.config.validation_interval) == 0

            validation_result = None
            validation_metrics = None
            if validation_due:
                local_validation_result = compiled_validation(
                    runtime_state.train_state,
                    runtime_state.val_rng,
                    chunk_result.validation_reset_weights,
                    trajectory_data_shards,
                    validation_trajectory_data_shards,
                )
                validation_number = current_counter // int(training.config.validation_interval)
                validation_result = finish_validation(
                    local_validation_result,
                    validation_process_exchange,
                    evaluate_all=evaluate_all,
                    event_name=f"validation_{validation_number:08d}",
                )
                runtime_state = runtime_state.replace(val_rng=validation_result.rng)
                validation_metrics = validation_result.metrics
                last_validation_metrics = validation_metrics

            if uses_warp_backend:
                # Captured graphs retain Warp temporary allocations. Release
                # them only after this chunk and any validation are complete.
                # Validation donates its RNG input, so chunk_result contains a
                # stale reference after validation. Wait only on live outputs.
                jax.block_until_ready((runtime_state, validation_result))
                clear_warp_graph_cache()

            log_offsets = tuple(
                offset
                for offset in range(chunk_size)
                if (chunk_start_counter + offset + 1) % logging_interval == 0
                or (validation_due and offset == chunk_size - 1)
            )
            final_offset = chunk_size - 1
            earlier_log_offsets = tuple(offset for offset in log_offsets if offset != final_offset)
            if logging_callback is not None and earlier_log_offsets:
                log_batch = _copy_log_batch(
                    chunk_result,
                    None,
                    offsets=earlier_log_offsets,
                    base_global_timestep=(progress.base_global_timestep + chunk_start_executed * steps_per_rollout),
                    base_raw_timestep=chunk_start_executed * steps_per_rollout,
                    steps_per_rollout=steps_per_rollout,
                    first_update=chunk_start_counter + 1,
                    validation_offset=None,
                    checkpoint_offset=None,
                    checkpoint_update=-1,
                )
                assert log_worker is not None
                pending_log = log_worker.submit(
                    _write_logs,
                    log_batch,
                    training,
                    logging_callback,
                )
                # Preserve the observable order of earlier update logs and the
                # checkpoint at this hard boundary.
                pending_log.result()
                pending_log = None

            should_checkpoint = _checkpoint_due(current_counter, training.config)
            checkpoint_update = -1
            if saver is not None and (should_checkpoint or remaining_updates == 0):
                saver.save(runtime_state, updates_done=updates_done)
                checkpoint_update = updates_done

            if logging_callback is not None and final_offset in log_offsets:
                log_batch = _copy_log_batch(
                    chunk_result,
                    validation_metrics,
                    offsets=(final_offset,),
                    base_global_timestep=(progress.base_global_timestep + chunk_start_executed * steps_per_rollout),
                    base_raw_timestep=chunk_start_executed * steps_per_rollout,
                    steps_per_rollout=steps_per_rollout,
                    first_update=chunk_start_counter + 1,
                    validation_offset=final_offset if validation_due else None,
                    checkpoint_offset=final_offset if checkpoint_update >= 0 else None,
                    checkpoint_update=checkpoint_update,
                )
                assert log_worker is not None
                pending_log = log_worker.submit(
                    _write_logs,
                    log_batch,
                    training,
                    logging_callback,
                )

                # Validation logs may trigger videos. Finish those logs before
                # continuing. Also finish the last log before returning.
                if validation_due or remaining_updates == 0:
                    pending_log.result()
                    pending_log = None

            last_training_metrics = _chunk_value(
                chunk_result.training_metrics,
                final_offset,
            )

        runtime_state = jax.block_until_ready(runtime_state)
    finally:
        if log_worker is not None:
            log_worker.shutdown(wait=True)
        if saver is not None:
            saver.close()

    return {
        "agent_state": training.agent_state_cls(train_state=runtime_state.train_state),
        "training_metrics": last_training_metrics,
        "validation_metrics": last_validation_metrics,
    }


def _log_model_info(
    runtime_state: Any,
    logging_callback: Any | None,
) -> None:
    params = runtime_state.train_state.params
    counts = count_actor_critic_params(params)
    logger.info(
        "Trainable parameters: %s (actor: %s, critic: %s, shared: %s)",
        f"{counts['total']:,}",
        f"{counts['actor']:,}",
        f"{counts['critic']:,}",
        f"{counts['shared']:,}",
    )
    log_network_architecture(params)
    if logging_callback is None:
        return
    try:
        logging_callback(
            {
                "_log_only": True,
                "model/param_count": counts["total"],
                "model/actor_param_count": counts["actor"],
                "model/critic_param_count": counts["critic"],
                "model/shared_param_count": counts["shared"],
                "max_timestep": 0,
                "jax_raw_timestep": 0,
            }
        )
    except Exception:
        logger.exception("Online logging callback failed for model info")


def _log_completed_run(
    training: PPOTraining,
    progress: Any,
    logging_callback: Any | None,
) -> None:
    if logging_callback is None:
        return
    logging_callback(
        {
            "mean_episode_return": 0.0,
            "mean_episode_length": 0.0,
            "max_timestep": progress.base_global_timestep,
            "learning_rate": training.base_learning_rate,
        }
    )
