"""Run JAX training and host-side logging and checkpoints."""

from __future__ import annotations

import ctypes
import gc
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from time import monotonic
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.algorithms.common.checkpoint_utils import optimizer_step_to_update
from musclemimic.algorithms.common.ppo_distributed import (
    PMAP_AXIS_NAME,
    assert_warp_multi_gpu_compatible,
    device_count,
    replicate_for_pmap,
)
from musclemimic.algorithms.ppo.train import PPOTraining
from musclemimic.runner.checkpoint_saving import PPOCheckpointSaver
from musclemimic.runner.hardware_gate import HardwareGateEvidenceRecorder
from musclemimic.runner.ppo_logging import format_ppo_log
from musclemimic.runner.warp_graph_cache import (
    clear_warp_graph_cache,
    require_safe_warp_graph_mode,
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


def release_jax_compilation_memory() -> None:
    """Drop quiescent compiled executables before switching GPU workloads."""
    jax.clear_caches()
    release_unused_host_memory()


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


def _shard_runtime_state(runtime_state: Any, devices: list[Any]) -> Any:
    """Place a saved ``[device, ...]`` runtime state without replicating it."""
    mesh = jax.sharding.Mesh(devices, (PMAP_AXIS_NAME,))
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(PMAP_AXIS_NAME),
    )

    def place(value):
        if value is None:
            return None
        host_value = np.asarray(jax.device_get(value))
        if host_value.ndim == 0 or host_value.shape[0] != len(devices):
            raise ValueError(
                "Exact-resume runtime leaves must have one leading entry per device; "
                f"got shape {host_value.shape} for {len(devices)} devices."
            )
        return jax.device_put(host_value, sharding)

    return jax.tree_util.tree_map(
        place,
        runtime_state,
        is_leaf=lambda value: value is None,
    )


def _place_resume_state(resume_state: Any, devices: list[Any]) -> Any:
    """Replicate model state while sharding an exact-resume runtime state."""
    if not isinstance(resume_state, dict) or "runtime_state" not in resume_state:
        return replicate_for_pmap(resume_state, devices)

    replicated_state = replicate_for_pmap(
        {key: value for key, value in resume_state.items() if key != "runtime_state"},
        devices,
    )
    return {
        **replicated_state,
        "runtime_state": _shard_runtime_state(
            resume_state["runtime_state"],
            devices,
        ),
    }


def _checkpoint_due(counter: int, config: Any) -> bool:
    """Return whether this update should be saved."""
    interval = int(config.get("checkpoint_interval", 0) or 0)
    periodic = interval > 0 and counter > 0 and counter % interval == 0
    validation = bool(config.get("checkpoints_on_validation", True)) and (
        counter % int(config.validation_interval) == 0
    )
    return periodic or validation


def checkpoint_boundary_stop_update_from_config(
    config: Any,
    *,
    completed_updates: int,
    remaining_updates: int,
) -> int | None:
    """Return the hardware gate's optional controlled checkpoint boundary.

    The setting is part of the experiment config and therefore of checkpoint
    compatibility. A resumed process that has already reached the boundary
    treats it as satisfied and continues toward the original training target.
    """
    debug_config = config.get("debug", {}) or {}
    if isinstance(debug_config, bool) or not hasattr(debug_config, "get"):
        return None
    hardware_gate_config = debug_config.get("hardware_gate", {}) or {}
    if isinstance(hardware_gate_config, bool) or not hasattr(hardware_gate_config, "get"):
        return None
    stop_value = hardware_gate_config.get("stop_after_checkpoint_update")
    if stop_value is None:
        return None
    if isinstance(stop_value, bool) or not isinstance(stop_value, int) or stop_value < 1:
        raise ValueError(
            f"debug.hardware_gate.stop_after_checkpoint_update must be a positive integer, got {stop_value!r}"
        )

    stop_update = int(stop_value)
    if not bool(hardware_gate_config.get("enabled", False)):
        raise ValueError("debug.hardware_gate.stop_after_checkpoint_update requires hardware_gate.enabled=true")
    if not bool(config.get("save_checkpoints", False)):
        raise ValueError("debug.hardware_gate.stop_after_checkpoint_update requires save_checkpoints=true")
    if bool(config.get("async_checkpointing", True)):
        raise ValueError("debug.hardware_gate.stop_after_checkpoint_update requires async_checkpointing=false")

    checkpoint_interval = int(config.get("checkpoint_interval", 0) or 0)
    if checkpoint_interval < 1 or stop_update % checkpoint_interval != 0:
        raise ValueError(
            f"debug.hardware_gate.stop_after_checkpoint_update={stop_update} must be a positive "
            f"multiple of checkpoint_interval={checkpoint_interval}"
        )

    if int(completed_updates) >= stop_update:
        return None
    final_update = int(completed_updates) + int(remaining_updates)
    if stop_update >= final_update:
        raise ValueError(
            f"debug.hardware_gate.stop_after_checkpoint_update={stop_update} must precede "
            f"final update {final_update} so resume work remains"
        )
    return stop_update


def _copy_log_batch(
    chunk_result: Any,
    validation_metrics: Any | None,
    *,
    offsets: tuple[int, ...],
    num_devices: int,
    base_global_timestep: int,
    base_raw_timestep: int,
    steps_per_rollout: int,
    first_update: int,
    validation_offset: int | None,
    checkpoint_offset: int | None,
    checkpoint_update: int,
) -> _LogBatch:
    """Start copying one chunk's small metric histories to the host."""
    metrics = _rank_zero(
        (
            chunk_result.training_metrics,
            chunk_result.ppo_metrics,
            chunk_result.diagnostics,
        ),
        num_devices,
    )
    validation_metrics = None if validation_metrics is None else _rank_zero(validation_metrics, num_devices)
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
        adaptive_config = training.config.get("adaptive_sampling", {})
        startup_log_limit = 5 * int(training.config.get("online_logging_interval", 1))
        if adaptive_config.get("enabled", False) and (update <= startup_log_limit or is_validation or is_checkpoint):
            logger.info(
                "Adaptive sampler: update=%d observed_motions=%d weight_sum=%.6f "
                "weight_min=%.8f floor=%.8f weight_max=%.8f ess=%.1f "
                "top_traj=%d top_difficulty=%.4f",
                update,
                int(log["adaptive/observed_motion_count"]),
                float(log["adaptive/weight_sum"]),
                float(log["adaptive/weight_min"]),
                float(log["adaptive/floor_weight"]),
                float(log["adaptive/weight_max"]),
                float(log["adaptive/effective_sample_size"]),
                int(log["adaptive/topk_0_traj_id"]),
                float(log["adaptive/topk_0_difficulty"]),
            )
        numerical_fault_count = float(log.get("numerics/nonfinite_transition_count", 0.0))
        if numerical_fault_count > 0:
            reason_counts = {
                reason: float(log.get(f"numerics/nonfinite_{reason}_count", 0.0))
                for reason in ("observation", "qpos", "qvel", "reward", "reward_info")
            }
            logger.warning(
                "Numerical transition fault: update=%d count=%.0f rate=%.8f "
                "offending_trajectories=%d top_traj=%d top_count=%.0f reasons=%s",
                update,
                numerical_fault_count,
                float(log.get("numerics/nonfinite_transition_rate", 0.0)),
                int(log.get("numerics/offending_trajectory_count", 0)),
                int(log.get("numerics/topk_0_traj_id", -1)),
                float(log.get("numerics/topk_0_count", 0.0)),
                reason_counts,
            )
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
    nonfinite_guard = config.get("nonfinite_guard", {}) or {}
    if bool(nonfinite_guard.get("enabled", True)):
        check_interval = int(nonfinite_guard.get("max_updates_between_checks", 0) or 0)
        if check_interval > 0:
            distances.append(check_interval - counter % check_interval)
    return min(remaining_updates, *distances)


def _first_nonfinite_metric(
    chunk_result: Any,
    *,
    chunk_size: int,
    num_devices: int,
) -> tuple[int, str] | None:
    """Return the first attempted update containing a nonfinite scalar metric."""
    metric_groups = {
        "training": chunk_result.training_metrics,
        "ppo": chunk_result.ppo_metrics,
        "diagnostics": chunk_result.diagnostics,
    }
    for group_name, tree in metric_groups.items():
        ranked = _rank_zero(tree, num_devices)
        paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(ranked)
        for path, leaf in paths_and_leaves:
            values = np.asarray(jax.device_get(leaf))
            if not np.issubdtype(values.dtype, np.inexact):
                continue
            values = values[:chunk_size]
            finite_by_update = np.isfinite(values)
            if finite_by_update.ndim > 1:
                finite_by_update = np.all(
                    finite_by_update,
                    axis=tuple(range(1, finite_by_update.ndim)),
                )
            failures = np.flatnonzero(~finite_by_update)
            if failures.size:
                path_text = jax.tree_util.keystr(path) or "<scalar>"
                return int(failures[0]), f"{group_name}{path_text}"
    return None


def _numerical_fault_log_offsets(
    chunk_result: Any,
    *,
    chunk_size: int,
    num_devices: int,
) -> tuple[int, ...]:
    """Return update offsets that must bypass the regular logging cadence."""
    counts = getattr(
        chunk_result.diagnostics,
        "numerics_invalid_transition_count",
        None,
    )
    if counts is None:
        return ()
    ranked_counts = _rank_zero(counts, num_devices)
    host_counts = np.atleast_1d(np.asarray(jax.device_get(ranked_counts)))
    return tuple(int(offset) for offset in np.flatnonzero(host_counts[:chunk_size] > 0))


def _chunk_value(tree: Any, offset: int, num_devices: int) -> Any:
    """Return the final update from a metric history without dropping replicas."""
    index = (slice(None), offset) if num_devices > 1 else offset
    return jax.tree.map(
        lambda value: None if value is None else value[index],
        tree,
        is_leaf=lambda value: value is None,
    )


def run_training(
    training: PPOTraining,
    rngs: jax.Array,
    *,
    config: Any | None = None,
    trajectory_data_shards: Any | None = None,
    validation_trajectory_data_shards: Any | None = None,
    resume_agent_state: Any | None = None,
    logging_callback: Any | None = None,
    logging_interval: int = 1,
) -> Any:
    """Run PPO with a host coordinator around pure compiled update chunks."""
    n_devices = 1 if config is None else device_count(config)
    devices = jax.local_devices()[:n_devices]
    validation_env = getattr(training, "val_env", None)
    if validation_env is None:
        validation_env = training.env
    training_uses_warp = getattr(training.env, "mjx_backend", None) == "warp"
    validation_uses_warp = getattr(validation_env, "mjx_backend", None) == "warp"
    multi_device_training_warp = n_devices > 1 and training_uses_warp
    multi_device_validation_warp = n_devices > 1 and validation_uses_warp
    multi_device_uses_warp = multi_device_training_warp or multi_device_validation_warp
    validation_config = training.config.get("validation", {}) or {}
    release_validation_compilation_memory = (
        multi_device_uses_warp
        and not isinstance(validation_config, bool)
        and hasattr(validation_config, "get")
        and bool(validation_config.get("release_compilation_caches", False))
    )

    if rngs.ndim > 1:
        raise NotImplementedError(
            "Multi-seed training (n_seeds > 1) is not supported. Run each seed as a separate job."
        )

    if validation_trajectory_data_shards is not None and trajectory_data_shards is None:
        raise ValueError("Validation trajectory shards require training trajectory shards.")
    if logging_interval < 1:
        raise ValueError("logging_interval must be positive")

    if n_devices == 1:
        init_rng = rngs
        placed_resume_state = resume_agent_state
        compile_fn = jax.jit
        init_in_axes = None
        update_in_axes = None
        validation_in_axes = None
        logger.info("Starting training...")
    else:
        if multi_device_uses_warp:
            assert_warp_multi_gpu_compatible()
        if multi_device_training_warp:
            require_safe_warp_graph_mode(training.env)
        if multi_device_validation_warp and validation_env is not training.env:
            require_safe_warp_graph_mode(validation_env)
        init_rng = jnp.broadcast_to(rngs[None], (n_devices, *rngs.shape))
        placed_resume_state = None if resume_agent_state is None else _place_resume_state(resume_agent_state, devices)

        def compile_fn(function, **kwargs):
            return jax.pmap(
                function,
                axis_name=PMAP_AXIS_NAME,
                devices=devices,
                **kwargs,
            )

        init_in_axes = (
            0,
            0 if placed_resume_state is not None else None,
            0 if trajectory_data_shards is not None else None,
        )
        update_in_axes = (
            0,
            0 if trajectory_data_shards is not None else None,
            None,
        )
        validation_in_axes = (
            0,
            0,
            0
            if (
                training.validation_shard_spec is not None
                or training.shard_spec is not None
                or (
                    training.validation_shares_training_trajectory
                    and training.config.get("adaptive_sampling", {}).get("enabled", False)
                )
            )
            else None,
            0 if trajectory_data_shards is not None else None,
            0 if validation_trajectory_data_shards is not None else None,
        )
        logger.info(f"Starting multi-GPU training: n_devices={n_devices} axis_name={PMAP_AXIS_NAME}")

    hardware_gate = HardwareGateEvidenceRecorder.from_config(
        training.config,
        devices,
        training.env,
        resumed=resume_agent_state is not None,
    )

    # Buffer donation can deadlock when pmap drives the MJX Warp FFI:
    # https://github.com/google-deepmind/mujoco/issues/2980
    nonfinite_guard = training.config.get("nonfinite_guard", {}) or {}
    nonfinite_guard_enabled = bool(nonfinite_guard.get("enabled", True))
    # Retain the pre-chunk state when the guard is active so a failed chunk can
    # never overwrite the state used for its recovery checkpoint.
    allow_buffer_donation = n_devices == 1 and not nonfinite_guard_enabled

    init_options: dict[str, Any] = {}
    if resume_agent_state is not None and allow_buffer_donation:
        init_options["donate_argnums"] = (1,)
    if init_in_axes is not None:
        init_options["in_axes"] = init_in_axes
    compiled_initialize = compile_fn(training.initialize, **init_options)

    validation_options: dict[str, Any] = {}
    if allow_buffer_donation:
        validation_options["donate_argnums"] = (1,)
    if validation_in_axes is not None:
        validation_options["in_axes"] = validation_in_axes
    compiled_validation = compile_fn(training.validate, **validation_options)

    runtime_state = compiled_initialize(
        init_rng,
        placed_resume_state,
        trajectory_data_shards,
    )
    if training_uses_warp:
        # Plain-Warp captures retain temporary allocations even on one device.
        # Wait for every replica before destroying initialization captures so
        # no callback can observe a graph whose CUDA work is still in flight.
        runtime_state = jax.block_until_ready(runtime_state)
        clear_warp_graph_cache()
    progress_state = _rank_zero(runtime_state, n_devices)
    progress = training.progress(
        progress_state,
        resumed=resume_agent_state is not None,
    )
    checkpoint_boundary_stop_update = checkpoint_boundary_stop_update_from_config(
        training.config,
        completed_updates=progress.completed_updates,
        remaining_updates=progress.remaining_updates,
    )
    if hardware_gate is not None:
        hardware_gate.record_state(
            runtime_state,
            phase="resume_initialize" if resume_agent_state is not None else "initialize",
            update_number=progress.completed_updates,
            global_timestep=progress.base_global_timestep,
        )
    _log_model_info(runtime_state, n_devices, logging_callback)

    if progress.remaining_updates == 0:
        _log_completed_run(training, progress, logging_callback)
        return {
            "agent_state": training.agent_state_cls(train_state=progress_state.train_state),
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
            num_devices=n_devices,
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

    update_options: dict[str, Any] = {}
    if allow_buffer_donation:
        update_options["donate_argnums"] = (0,)
    if update_in_axes is not None:
        update_options["in_axes"] = update_in_axes
    max_chunk_size = int(training.config.validation_interval)
    max_updates_between_checks = int(nonfinite_guard.get("max_updates_between_checks", 0) or 0)
    if nonfinite_guard_enabled and max_updates_between_checks > 0:
        max_chunk_size = min(max_chunk_size, max_updates_between_checks)

    def update_chunk(runtime_state, trajectory_data_override, num_updates):
        return training.update_chunk(
            runtime_state,
            trajectory_data_override,
            num_updates,
            max_chunk_size,
        )

    compiled_update_chunk = compile_fn(update_chunk, **update_options)

    last_training_metrics = None
    last_validation_metrics = None
    executed_updates = 0
    remaining_updates = progress.remaining_updates
    stopped_after_checkpoint_update = None
    logger.info("PPO update chunk: up to %d updates", max_chunk_size)
    log_worker = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppo-log") if logging_callback is not None else None
    )
    pending_log: Future[None] | None = None
    try:
        while remaining_updates:
            if multi_device_uses_warp and pending_log is not None:
                # Do not dispatch a new Warp graph until host work from the
                # preceding quiescent boundary has succeeded. Otherwise a log
                # failure could unwind with fresh CUDA work still in flight.
                pending_log.result()
                pending_log = None

            chunk_size = _next_chunk_size(
                current_counter,
                remaining_updates,
                training.config,
            )
            chunk_start_counter = current_counter
            chunk_start_executed = executed_updates
            safe_runtime_state = runtime_state
            chunk_started_at = monotonic()
            chunk_result = compiled_update_chunk(
                runtime_state,
                trajectory_data_shards,
                jnp.asarray(chunk_size, dtype=jnp.int32),
            )
            if training_uses_warp:
                # Each top-level compiled call owns a bounded set of
                # pointer-specific captures. Retire it before any host work or
                # a different compiled callable can fail. This is required on
                # one device too: otherwise consecutive PPO chunks retain
                # enough Warp temporaries to exhaust an 80 GiB accelerator.
                chunk_result = jax.block_until_ready(chunk_result)
                clear_warp_graph_cache()

            nonfinite = (
                _first_nonfinite_metric(
                    chunk_result,
                    chunk_size=chunk_size,
                    num_devices=n_devices,
                )
                if nonfinite_guard_enabled
                else None
            )
            if nonfinite is not None:
                failure_offset, metric_name = nonfinite
                failed_update = progress.completed_updates + chunk_start_executed + failure_offset + 1
                safe_update = progress.completed_updates + chunk_start_executed
                recovery_path = None
                if saver is not None:
                    recovery_path = saver.save(
                        safe_runtime_state,
                        updates_done=safe_update,
                    )
                recovery_message = (
                    f" Recovery checkpoint: {recovery_path}."
                    if recovery_path is not None
                    else " No checkpoint saver is configured."
                )
                raise FloatingPointError(
                    "PPO nonfinite guard stopped training before the failing chunk "
                    f"could be committed: update={failed_update}, metric={metric_name}, "
                    f"last_safe_update={safe_update}.{recovery_message}"
                )

            runtime_state = chunk_result.runtime_state

            # Other backends can overlap device execution with the preceding
            # host log. Multi-device Warp resolves it before dispatch above so
            # exceptions cannot strand an in-flight graph during unwinding.
            if not multi_device_uses_warp and pending_log is not None:
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
                if release_validation_compilation_memory:
                    if not multi_device_training_warp:
                        # Compilation caches are process-global. Ensure an
                        # asynchronous non-Warp update has also quiesced before
                        # dropping its executable at the phase boundary.
                        chunk_result = jax.block_until_ready(chunk_result)
                    release_jax_compilation_memory()
                    logger.info(
                        "Released JAX compilation caches before multi-device validation at update=%d",
                        updates_done,
                    )
                validation_result = compiled_validation(
                    runtime_state.train_state,
                    runtime_state.val_rng,
                    chunk_result.validation_reset_weights,
                    trajectory_data_shards,
                    validation_trajectory_data_shards,
                )
                if validation_uses_warp:
                    validation_result = jax.block_until_ready(validation_result)
                    clear_warp_graph_cache()
                elif release_validation_compilation_memory:
                    validation_result = jax.block_until_ready(validation_result)
                if release_validation_compilation_memory:
                    # Large validation and training executables can each consume
                    # tens of GiB. Discard the validation executable before the
                    # next training pmap is compiled or loaded; otherwise both
                    # remain resident and the transition can OOM even though
                    # each phase fits alone.
                    release_jax_compilation_memory()
                    logger.info(
                        "Released JAX compilation caches after multi-device validation at update=%d",
                        updates_done,
                    )
                runtime_state = runtime_state.replace(val_rng=validation_result.rng)
                validation_metrics = validation_result.metrics
                last_validation_metrics = validation_metrics

            if hardware_gate is not None:
                # Warp calls have already reached their individual quiescent
                # boundaries. Complete only work that has not already crossed
                # one so gate timing still covers every device computation.
                pending_device_work = []
                if not multi_device_training_warp:
                    pending_device_work.append(chunk_result)
                if validation_result is not None and not multi_device_validation_warp:
                    pending_device_work.append(validation_result)
                if pending_device_work:
                    jax.block_until_ready(tuple(pending_device_work))
            chunk_elapsed_seconds = monotonic() - chunk_started_at

            final_offset = chunk_size - 1
            if hardware_gate is not None:
                global_timestep = progress.base_global_timestep + executed_updates * steps_per_rollout
                hardware_gate.record_chunk(
                    chunk_updates=chunk_size,
                    environment_steps=chunk_size * steps_per_rollout,
                    elapsed_seconds=chunk_elapsed_seconds,
                    update_number=updates_done,
                    global_timestep=global_timestep,
                    validation_included=validation_due,
                    training_metrics=_chunk_value(
                        chunk_result.training_metrics,
                        final_offset,
                        n_devices,
                    ),
                    ppo_metrics=_chunk_value(
                        chunk_result.ppo_metrics,
                        final_offset,
                        n_devices,
                    ),
                    diagnostics=_chunk_value(
                        chunk_result.diagnostics,
                        final_offset,
                        n_devices,
                    ),
                    validation_metrics=validation_metrics,
                )
                hardware_gate.record_post_update_once(
                    runtime_state,
                    update_number=updates_done,
                    global_timestep=global_timestep,
                )

            numerical_fault_offsets = _numerical_fault_log_offsets(
                chunk_result,
                chunk_size=chunk_size,
                num_devices=n_devices,
            )
            numerical_fault_offset_set = frozenset(numerical_fault_offsets)
            log_offsets = tuple(
                offset
                for offset in range(chunk_size)
                if (chunk_start_counter + offset + 1) % logging_interval == 0
                or (validation_due and offset == chunk_size - 1)
                or offset in numerical_fault_offset_set
            )
            earlier_log_offsets = tuple(offset for offset in log_offsets if offset != final_offset)
            if logging_callback is not None and earlier_log_offsets:
                log_batch = _copy_log_batch(
                    chunk_result,
                    None,
                    offsets=earlier_log_offsets,
                    num_devices=n_devices,
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
                    num_devices=n_devices,
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
                n_devices,
            )

            if checkpoint_update == checkpoint_boundary_stop_update:
                if pending_log is not None:
                    pending_log.result()
                    pending_log = None
                stopped_after_checkpoint_update = checkpoint_update
                break

        runtime_state = jax.block_until_ready(runtime_state)
    finally:
        if log_worker is not None:
            log_worker.shutdown(wait=True)
        if saver is not None:
            saver.close()

    if stopped_after_checkpoint_update is not None:
        logger.info(
            "PPO checkpoint boundary stop complete: update=%d; checkpoint finalized, "
            "logs flushed, no later update dispatched",
            stopped_after_checkpoint_update,
        )

    final_state = _rank_zero(runtime_state, n_devices)
    return {
        "agent_state": training.agent_state_cls(train_state=final_state.train_state),
        "training_metrics": last_training_metrics,
        "validation_metrics": last_validation_metrics,
    }


def _rank_zero(tree: Any, num_devices: int) -> Any:
    """Return replica zero from a pmapped pytree, or the original value."""
    if num_devices == 1:
        return tree
    return jax.tree.map(
        lambda value: None if value is None else value[0],
        tree,
        is_leaf=lambda value: value is None,
    )


def _log_model_info(
    runtime_state: Any,
    num_devices: int,
    logging_callback: Any | None,
) -> None:
    params = _rank_zero(runtime_state.train_state.params, num_devices)
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
