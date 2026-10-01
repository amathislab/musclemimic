"""Numerical-transition diagnostics for PPO rollouts."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# These keys are emitted for every environment transition by the MJX boundary.
# Reason counts may overlap; ``numerics_nonfinite_transition`` is their union.
NUMERICAL_REASON_INFO_KEYS = (
    ("observation", "numerics_nonfinite_observation"),
    ("qpos", "numerics_nonfinite_qpos"),
    ("qvel", "numerics_nonfinite_qvel"),
    ("reward", "numerics_nonfinite_reward"),
    ("reward_info", "numerics_nonfinite_reward_info"),
)
NUMERICAL_TRANSITION_INFO_KEY = "numerics_nonfinite_transition"
NUMERICAL_TOPK_TRAJECTORIES = 3


class NumericalTransitionDiagnostics(NamedTuple):
    """Fixed-shape update diagnostics suitable for JIT, pmap, and host logs."""

    invalid_transition_count: jax.Array
    invalid_transition_rate: jax.Array
    reason_counts: jax.Array
    offending_trajectory_count: jax.Array
    topk_trajectory_ids: jax.Array
    topk_trajectory_counts: jax.Array


def numerical_transition_mask(info: dict[str, Any], reference: jax.Array) -> jax.Array:
    """Return the union of all numerical-fault flags in one rollout batch."""

    reference = jnp.asarray(reference)
    zeros = jnp.zeros(reference.shape, dtype=bool)
    invalid = jnp.asarray(info.get(NUMERICAL_TRANSITION_INFO_KEY, zeros), dtype=bool)
    invalid = jnp.broadcast_to(invalid, reference.shape)
    for _reason, key in NUMERICAL_REASON_INFO_KEYS:
        reason = jnp.asarray(info.get(key, zeros), dtype=bool)
        invalid = jnp.logical_or(invalid, jnp.broadcast_to(reason, reference.shape))
    return invalid


def summarize_numerical_transitions(
    trajectory_batch: Any,
    config: Any,
    *,
    axis_name: str | None = None,
) -> NumericalTransitionDiagnostics:
    """Aggregate numerical faults and their motion identities across replicas."""

    info = trajectory_batch.info
    reference = jnp.asarray(trajectory_batch.reward)
    zeros = jnp.zeros(reference.shape, dtype=bool)
    invalid = numerical_transition_mask(info, reference)

    reason_flags = tuple(
        jnp.broadcast_to(
            jnp.asarray(info.get(key, zeros), dtype=bool),
            reference.shape,
        )
        for _reason, key in NUMERICAL_REASON_INFO_KEYS
    )
    local_reason_counts = jnp.stack([jnp.sum(flag, dtype=jnp.float32) for flag in reason_flags])
    local_invalid_count = jnp.sum(invalid, dtype=jnp.float32)
    local_transition_count = jnp.asarray(invalid.size, dtype=jnp.float32)

    if axis_name is None:
        reason_counts = local_reason_counts
        invalid_count = local_invalid_count
        transition_count = local_transition_count
    else:
        reason_counts = jax.lax.psum(local_reason_counts, axis_name=axis_name)
        invalid_count = jax.lax.psum(local_invalid_count, axis_name=axis_name)
        transition_count = jax.lax.psum(local_transition_count, axis_name=axis_name)

    num_trajectories = max(int(config.get("n_trajectories", 1)), 1)
    fallback_ids = jnp.full(reference.shape, -1, dtype=jnp.int32)
    trajectory_ids = jnp.broadcast_to(
        jnp.asarray(info.get("final_traj_no", fallback_ids), dtype=jnp.int32),
        reference.shape,
    ).reshape(-1)
    invalid_flat = invalid.reshape(-1)
    valid_identity = (trajectory_ids >= 0) & (trajectory_ids < num_trajectories)
    safe_ids = jnp.clip(trajectory_ids, 0, num_trajectories - 1)
    local_trajectory_counts = jax.ops.segment_sum(
        (invalid_flat & valid_identity).astype(jnp.float32),
        safe_ids,
        num_segments=num_trajectories,
    )
    trajectory_counts = (
        local_trajectory_counts if axis_name is None else jax.lax.psum(local_trajectory_counts, axis_name=axis_name)
    )

    topk_size = min(NUMERICAL_TOPK_TRAJECTORIES, num_trajectories)
    topk_counts, topk_ids = jax.lax.top_k(trajectory_counts, topk_size)
    if topk_size < NUMERICAL_TOPK_TRAJECTORIES:
        padding = NUMERICAL_TOPK_TRAJECTORIES - topk_size
        topk_counts = jnp.pad(topk_counts, (0, padding))
        topk_ids = jnp.pad(topk_ids, (0, padding), constant_values=-1)
    topk_ids = jnp.where(topk_counts > 0, topk_ids, -1)

    return NumericalTransitionDiagnostics(
        invalid_transition_count=invalid_count,
        invalid_transition_rate=invalid_count / jnp.maximum(transition_count, 1.0),
        reason_counts=reason_counts,
        offending_trajectory_count=jnp.sum(trajectory_counts > 0, dtype=jnp.int32),
        topk_trajectory_ids=topk_ids,
        topk_trajectory_counts=topk_counts,
    )


__all__ = [
    "NUMERICAL_REASON_INFO_KEYS",
    "NUMERICAL_TOPK_TRAJECTORIES",
    "NUMERICAL_TRANSITION_INFO_KEY",
    "NumericalTransitionDiagnostics",
    "numerical_transition_mask",
    "summarize_numerical_transitions",
]
