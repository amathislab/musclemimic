"""Host-side reduction for validation workloads with runtime skew."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.algorithms.common.trajectory_evaluation import (
    ExactTrajectorySetStatistics,
)
from musclemimic.utils.metrics import ValidationSummary

ValidationProcessExchange = Callable[[str, Any], tuple[Any, ...]]

_RMSE_FIELDS = {
    "err_root_xyz",
    "err_joint_pos",
    "err_joint_vel",
    "err_rpos",
}


def add_validation_shard_axis(metrics: Any) -> Any:
    """Expose one leading data-axis slot for every shard-local metric leaf."""
    return jax.tree.map(lambda value: jnp.expand_dims(value, axis=0), metrics)


def addressable_validation_metrics(metrics: Any) -> Any:
    """Copy only this process's completed validation shards to host memory."""

    def copy_addressable(value: Any) -> np.ndarray:
        if not isinstance(value, jax.Array):
            return np.asarray(value)
        shards = value.addressable_shards
        if not shards:
            raise ValueError("Validation metric has no addressable device shards.")
        if value.sharding.is_fully_replicated:
            return np.asarray(shards[0].data)
        return np.concatenate(
            tuple(np.asarray(shard.data) for shard in shards),
            axis=0,
        )

    return jax.tree.map(copy_addressable, metrics)


def _concatenate_process_metrics(process_metrics: Sequence[Any]) -> Any:
    if not process_metrics:
        raise ValueError("At least one validation metric payload is required.")
    leaves, tree_definition = jax.tree_util.tree_flatten(process_metrics[0])
    process_leaves = [leaves]
    for metrics in process_metrics[1:]:
        other_leaves, other_definition = jax.tree_util.tree_flatten(metrics)
        if other_definition != tree_definition:
            raise ValueError("Validation metric payloads have incompatible tree structures.")
        process_leaves.append(other_leaves)

    combined_leaves = []
    for leaf_index, values in enumerate(zip(*process_leaves, strict=True)):
        arrays = tuple(np.asarray(value) for value in values)
        if any(array.ndim == 0 for array in arrays):
            raise ValueError(f"Validation metric leaf {leaf_index} is missing its leading device axis.")
        combined_leaves.append(np.concatenate(arrays, axis=0))
    return tree_definition.unflatten(combined_leaves)


def _summarize_exact_statistics(
    statistics: ExactTrajectorySetStatistics,
) -> ValidationSummary:
    episode_count = statistics.episode_count
    safe_episode_count = np.maximum(episode_count, 1.0)
    safe_step_count = np.maximum(statistics.step_count, 1.0)
    step_means = {key: value / safe_step_count for key, value in statistics.step_sums.items()}

    def step_mean(name: str) -> np.ndarray:
        return step_means.get(name, np.asarray(0.0, dtype=np.float32))

    return ValidationSummary(
        episode_count=episode_count,
        mean_episode_return=statistics.episode_return_sum / safe_episode_count,
        mean_episode_length=statistics.episode_length_sum / safe_episode_count,
        early_termination_count=statistics.early_termination_count,
        early_termination_rate=(statistics.early_termination_count / safe_episode_count),
        horizon_timeout_count=statistics.horizon_timeout_count,
        horizon_timeout_rate=statistics.horizon_timeout_count / safe_episode_count,
        frame_coverage=(statistics.covered_frame_sum / np.maximum(statistics.total_frame, 1.0)),
        total_frame=statistics.total_frame,
        reward_total=step_mean("reward_total"),
        reward_qpos=step_mean("reward_qpos"),
        reward_qvel=step_mean("reward_qvel"),
        reward_root_pos=step_mean("reward_root_pos"),
        reward_root_quat=step_mean("reward_root_quat"),
        reward_rpos=step_mean("reward_rpos"),
        reward_rquat=step_mean("reward_rquat"),
        reward_rvel_rot=step_mean("reward_rvel_rot"),
        reward_rvel_lin=step_mean("reward_rvel_lin"),
        reward_root_vel=step_mean("reward_root_vel"),
        penalty_total=step_mean("penalty_total"),
        penalty_activation_energy=step_mean("penalty_activation_energy"),
        err_root_xyz=step_mean("err_root_xyz"),
        err_root_quat=step_mean("err_root_quat"),
        err_root_yaw=step_mean("err_root_yaw"),
        err_joint_pos=step_mean("err_joint_pos"),
        err_joint_vel=step_mean("err_joint_vel"),
        err_site_abs=step_mean("err_site_abs"),
        err_rpos=step_mean("err_rpos"),
    )


def _reduce_sampled_summaries(summary: ValidationSummary) -> ValidationSummary:
    def reduce(path: tuple[Any, ...], value: Any) -> np.ndarray:
        array = np.asarray(value)
        name = path[-1].name
        if name.endswith("_count"):
            return np.sum(array, axis=0)
        if name == "max_timestep":
            return np.max(array, axis=0)
        if name in _RMSE_FIELDS:
            return np.sqrt(np.mean(np.square(array), axis=0))
        return np.mean(array, axis=0)

    reduced = jax.tree_util.tree_map_with_path(reduce, summary)
    global_count = np.maximum(reduced.episode_count, 1.0)
    local_count = np.asarray(summary.episode_count)
    return reduced.replace(
        mean_episode_return=(np.sum(np.asarray(summary.mean_episode_return) * local_count, axis=0) / global_count),
        mean_episode_length=(np.sum(np.asarray(summary.mean_episode_length) * local_count, axis=0) / global_count),
        early_termination_rate=(reduced.early_termination_count / global_count),
        horizon_timeout_rate=reduced.horizon_timeout_count / global_count,
        reward_termination_rate=(reduced.reward_termination_count / global_count),
    )


def reduce_validation_metrics_host(
    process_metrics: Sequence[Any],
    *,
    evaluate_all: bool,
) -> ValidationSummary:
    """Aggregate validation metrics on host across devices and processes."""
    combined = _concatenate_process_metrics(process_metrics)
    if evaluate_all:
        statistics = jax.tree.map(
            lambda value: np.sum(np.asarray(value), axis=0),
            combined,
        )
        return _summarize_exact_statistics(statistics)
    return _reduce_sampled_summaries(combined)


def finish_validation(
    local_result: Any,
    process_exchange: ValidationProcessExchange | None,
    *,
    evaluate_all: bool,
    event_name: str,
) -> Any:
    """Wait for local rollouts, exchange host payloads, and reduce on host."""
    jax.block_until_ready(local_result)
    local_metrics = addressable_validation_metrics(local_result.metrics)
    if process_exchange is not None:
        process_metrics = process_exchange(event_name, local_metrics)
    else:
        if jax.process_count() > 1:
            raise ValueError("Multi-host validation requires a host metric exchange.")
        process_metrics = (local_metrics,)
    return local_result._replace(
        metrics=reduce_validation_metrics_host(
            process_metrics,
            evaluate_all=evaluate_all,
        ),
    )
