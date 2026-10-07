"""Host coordinator for FlashSAC training."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.distributed import (
    DATA_AXIS_NAME,
    assert_warp_multi_gpu_compatible,
    device_count,
    make_data_mesh,
)
from musclemimic.algorithms.common.evaluation_rng import evaluation_rollout_keys
from musclemimic.algorithms.flashsac.state import (
    FlashSACAgentState,
    FlashSACPolicyState,
    FlashSACRuntimeState,
)
from musclemimic.algorithms.flashsac.train import FlashSACTraining
from musclemimic.utils.logging import setup_logger

from .data_parallel import (
    DATA,
    REPLICATED,
    compile_sharded,
    data_specs,
    eval_shard_shape,
    put_tree,
    replicated_specs,
    require_explicit_trajectory_data,
    trajectory_specs,
)
from .flashsac_checkpoint_saving import FlashSACCheckpointSaver
from .flashsac_logging import FlashSACLogWindow, format_flashsac_log
from .flashsac_sharding import (
    add_device_key_axis,
    mark_runtime_data_varying,
    remove_device_key_axis,
    runtime_specs,
    train_state_specs,
)
from .validation_reduction import (
    ValidationProcessExchange,
    add_validation_shard_axis,
    finish_validation,
)
from .validation_video_recorder import validation_video_due
from .warp_graph_cache import clear_warp_graph_cache, disable_warp_graph_eviction

logger = setup_logger(__name__)


class _LogBatch(NamedTuple):
    collection_metrics: Any
    update_metrics: Any
    reward_ema: Any
    replay_size: Any
    learning_rate: Any
    validation_metrics: Any | None
    num_interactions: int
    offsets: tuple[int, ...]
    base_global_timestep: int
    base_raw_timestep: int
    num_envs: int
    replay_size_scale: int
    first_interaction: int
    validation_offset: int | None
    checkpoint_offset: int | None
    checkpoint_update: int


def _chunk_window(config: Any, logging_interval: int) -> int:
    """Return a static scan window bounded by configured lifecycle events."""
    del logging_interval
    intervals = [int(config.get("update_chunk_size", 10))]
    if bool(config.get("validation", {}).get("active", False)):
        intervals.append(int(config.validation_interval))
    if bool(config.get("save_checkpoints", False)):
        checkpoint_interval = int(config.get("checkpoint_interval", 0) or 0)
        if checkpoint_interval > 0:
            intervals.append(checkpoint_interval)
    return min(intervals)


def _next_chunk_size(
    interaction: int,
    remaining: int,
    config: Any,
    max_chunk_size: int,
) -> int:
    """Stop at the next validation, checkpoint, final, or scan-window boundary."""
    distances = [remaining, max_chunk_size]
    if bool(config.get("validation", {}).get("active", False)):
        interval = int(config.validation_interval)
        distances.append(interval - interaction % interval)
    if bool(config.get("save_checkpoints", False)):
        interval = int(config.get("checkpoint_interval", 0) or 0)
        if interval > 0:
            distances.append(interval - interaction % interval)
    return min(distances)


def run_flashsac_training(
    training: FlashSACTraining,
    rng: jax.Array,
    *,
    config: Any,
    trajectory_data_shards: Any | None = None,
    validation_trajectory_data_shards: Any | None = None,
    resume_agent_state: Any | None = None,
    logging_callback: Any | None = None,
    logging_interval: int = 1,
    validation_process_exchange: ValidationProcessExchange | None = None,
) -> dict[str, Any]:
    """Run pure FlashSAC computations and host-side lifecycle work."""
    if jax.random.key_data(rng).shape != (2,):
        raise ValueError("FlashSAC expects one PRNG key")
    if validation_trajectory_data_shards is not None and trajectory_data_shards is None:
        raise ValueError("Validation trajectory shards require training trajectory shards")
    require_explicit_trajectory_data(
        training.env,
        trajectory_data_shards,
        label="Training",
    )
    if training.val_env is not None and not training.validation_shares_training_trajectory:
        require_explicit_trajectory_data(
            training.val_env,
            validation_trajectory_data_shards,
            label="Validation",
        )
    if logging_interval < 1:
        raise ValueError("logging_interval must be positive")

    n_devices = device_count(config)
    devices = jax.local_devices()[:n_devices]
    mesh = make_data_mesh(n_devices, devices=devices)
    uses_warp_backend = getattr(training.env, "mjx_backend", None) == "warp"
    if n_devices > 1:
        assert_warp_multi_gpu_compatible()
        if uses_warp_backend:
            disable_warp_graph_eviction()
    logger.info(
        "Starting data-parallel FlashSAC: n_devices=%d axis_name=%s local_envs=%d global_envs=%d",
        n_devices,
        DATA_AXIS_NAME,
        int(training.config.num_envs),
        int(training.config.global_num_envs),
    )

    train_trajectory_specs = trajectory_specs(trajectory_data_shards)
    validation_trajectory_specs = trajectory_specs(validation_trajectory_data_shards)
    init_rng = jax.device_put(
        rng,
        jax.sharding.NamedSharding(mesh, REPLICATED),
    )
    resume_specs = train_state_specs(resume_agent_state)
    placed_resume_state = put_tree(resume_agent_state, resume_specs, mesh)

    def initialize_shard(rng, agent_state_data, trajectory_data_override):
        runtime_state = training.initialize(
            rng,
            agent_state_data,
            trajectory_data_override,
        )
        runtime_state = mark_runtime_data_varying(
            runtime_state,
            training.axis_name,
        )
        return add_device_key_axis(runtime_state)

    local_runtime_template = eval_shard_shape(
        initialize_shard,
        mesh=mesh,
        in_specs=(REPLICATED, resume_specs, train_trajectory_specs),
        args=(init_rng, placed_resume_state, trajectory_data_shards),
    )
    state_specs = runtime_specs(local_runtime_template)
    compiled_initialize = compile_sharded(
        initialize_shard,
        mesh=mesh,
        in_specs=(REPLICATED, resume_specs, train_trajectory_specs),
        out_specs=state_specs,
        donate_argnums=(1,) if resume_agent_state is not None else (),
    )

    runtime_state = compiled_initialize(
        init_rng,
        placed_resume_state,
        trajectory_data_shards,
    )
    progress = training.progress()
    _log_model_info(runtime_state, n_devices, logging_callback)
    logger.info(
        "FlashSAC interactions: completed=%d remaining=%d",
        progress.completed_interactions,
        progress.remaining_interactions,
    )

    if progress.remaining_interactions == 0:
        return {
            "agent_state": FlashSACAgentState(runtime_state.agent),
            "training_metrics": None,
            "validation_metrics": None,
        }

    max_chunk_size = _chunk_window(training.config, logging_interval)

    def update_chunk_shard(runtime_state, trajectory_data_override, num_interactions):
        chunk = training.update_chunk(
            remove_device_key_axis(runtime_state),
            trajectory_data_override,
            num_interactions,
            max_chunk_size,
        )
        return chunk._replace(
            runtime_state=add_device_key_axis(chunk.runtime_state),
        )

    update_in_specs = (state_specs, train_trajectory_specs, REPLICATED)
    one_interaction = jax.device_put(
        jnp.asarray(1, dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, REPLICATED),
    )
    local_chunk_template = eval_shard_shape(
        update_chunk_shard,
        mesh=mesh,
        in_specs=update_in_specs,
        args=(runtime_state, trajectory_data_shards, one_interaction),
    )
    chunk_out_specs = local_chunk_template._replace(
        runtime_state=state_specs,
        collection_metrics=replicated_specs(local_chunk_template.collection_metrics),
        update_metrics=replicated_specs(local_chunk_template.update_metrics),
        reward_ema=REPLICATED,
        replay_size=REPLICATED,
        learning_rate=REPLICATED,
    )
    compiled_update_chunk = compile_sharded(
        update_chunk_shard,
        mesh=mesh,
        in_specs=update_in_specs,
        out_specs=chunk_out_specs,
        donate_argnums=(0,),
    )

    validation_active = bool(training.config.get("validation", {}).get("active", False))
    if validation_active and jax.process_count() > 1 and validation_process_exchange is None:
        raise ValueError("Multi-host FlashSAC validation requires a host metric exchange.")
    compiled_validation = None
    if validation_active:

        def validation_shard(
            policy_state,
            validation_rng,
            trajectory_data_override,
            validation_trajectory_data_override,
        ):
            result = training.validate(
                policy_state,
                jnp.squeeze(validation_rng, axis=0),
                trajectory_data_override,
                validation_trajectory_data_override,
            )
            return result._replace(
                metrics=add_validation_shard_axis(result.metrics),
                rng=jnp.expand_dims(result.rng, axis=0),
            )

        policy_template = FlashSACPolicyState(
            actor_params=runtime_state.agent.actor_params,
            actor_batch_stats=runtime_state.agent.actor_batch_stats,
        )
        policy_specs = replicated_specs(policy_template)
        validation_in_specs = (
            policy_specs,
            state_specs.validation_rng,
            train_trajectory_specs,
            validation_trajectory_specs,
        )
        local_validation_template = eval_shard_shape(
            validation_shard,
            mesh=mesh,
            in_specs=validation_in_specs,
            args=(
                policy_template,
                runtime_state.validation_rng,
                trajectory_data_shards,
                validation_trajectory_data_shards,
            ),
        )
        evaluate_all = bool(training.config.validation.get("evaluate_all", False))
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

    logger.info("FlashSAC interaction chunk: up to %d interactions", max_chunk_size)

    saver = None
    if bool(training.config.get("save_checkpoints", False)):
        saver = FlashSACCheckpointSaver(
            training.agent_conf,
            training.env,
            base_global_timestep=progress.base_global_timestep,
            base_completed_interactions=progress.completed_interactions,
            target_global_timestep=progress.target_global_timestep,
        )

    pending_log: Future[None] | None = None
    log_worker = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="flashsac-log") if logging_callback is not None else None
    )
    last_collection_metrics = None
    last_validation_metrics = None
    metric_window = FlashSACLogWindow()
    completed_this_run = 0
    interaction = progress.completed_interactions
    remaining = progress.remaining_interactions
    try:
        while remaining:
            chunk_size = _next_chunk_size(
                interaction,
                remaining,
                training.config,
                max_chunk_size,
            )
            chunk_start_interaction = interaction
            chunk_start_completed = completed_this_run
            chunk = compiled_update_chunk(
                runtime_state,
                trajectory_data_shards,
                jax.device_put(
                    jnp.asarray(chunk_size, dtype=jnp.int32),
                    jax.sharding.NamedSharding(mesh, REPLICATED),
                ),
            )
            runtime_state = chunk.runtime_state

            if pending_log is not None:
                pending_log.result()
                pending_log = None

            completed_this_run += chunk_size
            interaction += chunk_size
            remaining -= chunk_size
            validation_due = validation_active and interaction % int(training.config.validation_interval) == 0

            validation_result = None
            validation_metrics = None
            if validation_due:
                assert compiled_validation is not None
                validation_number = interaction // int(training.config.validation_interval)
                scheduled_validation_rng = jax.device_put(
                    evaluation_rollout_keys(
                        int(training.config.validation.get("eval_seed", 0)),
                        validation_number,
                        n_devices,
                    ),
                    jax.sharding.NamedSharding(mesh, DATA),
                )
                policy_state = FlashSACPolicyState(
                    actor_params=runtime_state.agent.actor_params,
                    actor_batch_stats=runtime_state.agent.actor_batch_stats,
                )
                local_validation_result = compiled_validation(
                    policy_state,
                    scheduled_validation_rng,
                    trajectory_data_shards,
                    validation_trajectory_data_shards,
                )
                validation_result = finish_validation(
                    local_validation_result,
                    validation_process_exchange,
                    evaluate_all=evaluate_all,
                    event_name=f"validation_{validation_number:08d}",
                )
                validation_metrics = validation_result.metrics
                last_validation_metrics = validation_metrics

            if uses_warp_backend:
                # Captured graphs retain Warp temporary allocations. Release
                # them only after this chunk and any validation are complete.
                # Validation donates its RNG input, so chunk contains a stale
                # reference after validation. Wait only on live outputs.
                jax.block_until_ready((runtime_state, validation_result))
                clear_warp_graph_cache()

            checkpoint_update = -1
            if saver is not None and (_checkpoint_due(interaction, training.config) or remaining == 0):
                saver.save(runtime_state.agent, interactions_done=interaction)
                checkpoint_update = interaction

            log_offsets = tuple(
                offset
                for offset in range(chunk_size)
                if (chunk_start_interaction + offset + 1) % logging_interval == 0
                or (validation_due and offset == chunk_size - 1)
                or (remaining == 0 and offset == chunk_size - 1)
            )
            final_offset = chunk_size - 1
            if logging_callback is not None:
                log_batch = _copy_log_batch(
                    chunk,
                    validation_metrics,
                    num_interactions=chunk_size,
                    offsets=log_offsets,
                    base_global_timestep=(
                        progress.base_global_timestep + chunk_start_completed * int(training.config.global_num_envs)
                    ),
                    base_raw_timestep=chunk_start_completed * int(training.config.global_num_envs),
                    num_envs=int(training.config.global_num_envs),
                    replay_size_scale=n_devices,
                    first_interaction=chunk_start_interaction + 1,
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
                    metric_window,
                )

                # The callback enqueues validation video work from the freshly
                # saved checkpoint. Finish that handoff before the next chunk.
                if validation_due or remaining == 0:
                    pending_log.result()
                    pending_log = None

            last_collection_metrics = jax.tree.map(
                lambda values, index=final_offset: values[index],
                chunk.collection_metrics,
            )
        runtime_state = jax.block_until_ready(runtime_state)
    finally:
        try:
            if pending_log is not None:
                pending_log.result()
        finally:
            if log_worker is not None:
                log_worker.shutdown(wait=True)
            if saver is not None:
                saver.close()

    return {
        "agent_state": FlashSACAgentState(runtime_state.agent),
        "training_metrics": last_collection_metrics,
        "validation_metrics": last_validation_metrics,
    }


def _copy_log_batch(
    chunk: Any,
    validation_metrics: Any | None,
    *,
    num_interactions: int,
    offsets: tuple[int, ...],
    base_global_timestep: int,
    base_raw_timestep: int,
    num_envs: int,
    replay_size_scale: int,
    first_interaction: int,
    validation_offset: int | None,
    checkpoint_offset: int | None,
    checkpoint_update: int,
) -> _LogBatch:
    """Start copying one chunk's small metric histories to the host."""
    values = (
        chunk.collection_metrics,
        chunk.update_metrics,
        chunk.reward_ema,
        chunk.replay_size,
        chunk.learning_rate,
        validation_metrics,
    )
    jax.copy_to_host_async(values)
    return _LogBatch(
        collection_metrics=values[0],
        update_metrics=values[1],
        reward_ema=values[2],
        replay_size=values[3],
        learning_rate=values[4],
        validation_metrics=values[5],
        num_interactions=num_interactions,
        offsets=offsets,
        base_global_timestep=base_global_timestep,
        base_raw_timestep=base_raw_timestep,
        num_envs=num_envs,
        replay_size_scale=replay_size_scale,
        first_interaction=first_interaction,
        validation_offset=validation_offset,
        checkpoint_offset=checkpoint_offset,
        checkpoint_update=checkpoint_update,
    )


def _write_logs(
    log_batch: _LogBatch,
    training: FlashSACTraining,
    logging_callback: Any,
    metric_window: FlashSACLogWindow,
) -> None:
    """Accumulate every interaction and emit complete metric windows on host."""
    host_values = jax.device_get(
        (
            log_batch.collection_metrics,
            log_batch.update_metrics,
            log_batch.reward_ema,
            log_batch.replay_size,
            log_batch.learning_rate,
            log_batch.validation_metrics,
        )
    )
    for offset in range(log_batch.num_interactions):
        metric_window.add(
            jax.tree.map(lambda values, index=offset: values[index], host_values[0]),
            jax.tree.map(lambda values, index=offset: values[index], host_values[1]),
        )
        if offset not in log_batch.offsets:
            continue
        window_metrics = metric_window.pop()
        interaction = log_batch.first_interaction + offset
        validation_metrics = host_values[5] if offset == log_batch.validation_offset else None
        checkpoint_update = log_batch.checkpoint_update if offset == log_batch.checkpoint_offset else -1
        log = format_flashsac_log(
            window_metrics.collection,
            window_metrics.updates,
            reward_ema=host_values[2][offset],
            replay_size=host_values[3][offset] * log_batch.replay_size_scale,
            learning_rate=host_values[4][offset],
            validation_metrics=validation_metrics,
            global_timestep=log_batch.base_global_timestep + (offset + 1) * log_batch.num_envs,
            raw_timestep=log_batch.base_raw_timestep + (offset + 1) * log_batch.num_envs,
            checkpoint_update=checkpoint_update,
            window_interactions=window_metrics.num_interactions,
            validation_number=(
                interaction // int(training.config.validation_interval) if validation_metrics is not None else None
            ),
            config=training.config,
        )
        try:
            logging_callback(log)
        except Exception:
            logger.exception("Online logging callback failed at interaction %d", interaction)


def _checkpoint_due(interaction: int, config: Any) -> bool:
    interval = int(config.get("checkpoint_interval", 0) or 0)
    periodic = interval > 0 and interaction % interval == 0
    validation_due = (
        bool(config.get("validation", {}).get("active", False)) and interaction % int(config.validation_interval) == 0
    )
    validation_checkpoint = bool(config.get("checkpoints_on_validation", True)) and validation_due
    video_checkpoint = validation_due and _video_due(interaction, config)
    return periodic or validation_checkpoint or video_checkpoint


def _video_due(interaction: int, config: Any) -> bool:
    validation_config = config.get("validation", {})
    frequency = int(validation_config.get("video_frequency", 0) or 0)
    length = int(validation_config.get("video_length", 0) or 0)
    if not validation_config.get("active", False) or frequency <= 0 or length <= 0:
        return False
    validation_number = interaction // int(config.validation_interval)
    return validation_video_due(validation_number, frequency)


def _log_model_info(
    runtime_state: FlashSACRuntimeState,
    n_devices: int,
    logging_callback: Any | None,
) -> None:
    actor_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(runtime_state.agent.actor_params))
    critic_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(runtime_state.agent.critic_params))
    replay_bytes = sum(
        array.size * array.dtype.itemsize
        for array in (
            runtime_state.agent.replay_buffer.observations,
            runtime_state.agent.replay_buffer.next_observations,
            runtime_state.agent.replay_buffer.actions,
            runtime_state.agent.replay_buffer.rewards,
            runtime_state.agent.replay_buffer.terminated,
        )
    )
    replay_gib = replay_bytes / 1024**3
    logger.info(
        "FlashSAC model: actor_params=%s critic_params=%s replay_global=%.2f GiB replay_per_device=%.2f GiB",
        f"{actor_count:,}",
        f"{critic_count:,}",
        replay_gib,
        replay_gib / n_devices,
    )
    if logging_callback is not None:
        logging_callback(
            {
                "_log_only": True,
                "model/actor_param_count": actor_count,
                "model/critic_param_count": critic_count,
                "model/param_count": actor_count + critic_count,
            }
        )
