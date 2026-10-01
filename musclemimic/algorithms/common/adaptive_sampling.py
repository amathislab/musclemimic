"""Adaptive trajectory sampling utilities for PPO training.

The sampler uses completed-episode evidence to focus training on difficult
trajectories while retaining an explicit uniform exploration floor. A failed
episode is weighted by how early it failed, so falling immediately is stronger
evidence than falling one frame before the end of a motion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_adaptive_weights(
    done_counts: jnp.ndarray,
    early_counts: jnp.ndarray,
    ema_done: jnp.ndarray,
    ema_early: jnp.ndarray,
    beta: float = 0.2,
    alpha: float = 1.0,
    floor_mix: float = 0.1,
    eps_div: float = 1e-6,
    eps_pow: float = 1e-6,
    group_ids: jnp.ndarray | None = None,
    num_groups: int | None = None,
    difficulty_mix: float | None = None,
    max_weight_ratio: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute trajectory sampling weights from early termination rates using count-EMA.

    Higher early termination rate -> higher sampling probability (prioritize harder trajectories).

    Pure 1D interface - runner is responsible for broadcasting to (num_envs, n_traj).

    Args:
        done_counts: (n_traj,) done counts from current rollout
        early_counts: (n_traj,) early termination counts from current rollout
        ema_done: (n_traj,) EMA state for done counts
        ema_early: (n_traj,) EMA state for early termination counts
        beta: EMA decay rate (higher = faster adaptation)
        alpha: Prioritization exponent (higher = more aggressive)
        floor_mix: Uniform mixture floor, i.e., 1 - lambda (ensures all trajectories get some samples)
        eps_div: Small constant for division stability
        eps_pow: Small constant for power stability

    Returns:
        weights_1d: (n_traj,) sampling weights
        new_ema_done: (n_traj,) updated EMA done counts
        new_ema_early: (n_traj,) updated EMA early counts
        rate_hat: (n_traj,) estimated early termination rates (for logging)
    """
    # EMA update
    new_ema_done = (1 - beta) * ema_done + beta * done_counts
    new_ema_early = (1 - beta) * ema_early + beta * early_counts

    # Rate estimation (eps_div prevents division by zero)
    rate_hat = new_ema_early / (new_ema_done + eps_div)

    # Priority weights (eps_pow prevents zero priorities, higher rate = higher weight)
    priorities = jnp.power(rate_hat + eps_pow, alpha)

    # NaN/Inf safeguard - replace non-finite values with 0
    priorities = jnp.where(jnp.isfinite(priorities), priorities, 0.0)

    # NaN/zero fallback - if sum is 0, NaN, or Inf, use uniform
    n_traj = done_counts.shape[0]
    uniform = jnp.ones(n_traj) / n_traj
    sum_p = jnp.sum(priorities)
    use_priorities = jnp.isfinite(sum_p) & (sum_p > 0)
    normalized = jnp.where(use_priorities, priorities / sum_p, uniform)

    if difficulty_mix is None:
        difficulty_mix = 1.0 - floor_mix
    if not 0.0 <= difficulty_mix <= 1.0:
        raise ValueError("difficulty_mix must be between 0 and 1")
    if max_weight_ratio is not None and max_weight_ratio < 1.0:
        raise ValueError("max_weight_ratio must be at least 1")

    if group_ids is None:
        # Backwards-compatible single-pool behavior.
        weights_1d = difficulty_mix * normalized + (1.0 - difficulty_mix) * uniform
        if max_weight_ratio is not None:
            cap = jnp.asarray(max_weight_ratio / n_traj, dtype=weights_1d.dtype)
            capped = jnp.minimum(weights_1d, cap)
            capacity = jnp.maximum(cap - capped, 0.0)
            deficit = 1.0 - jnp.sum(capped)
            weights_1d = capped + capacity * deficit / jnp.maximum(jnp.sum(capacity), eps_div)
    else:
        group_ids = jnp.asarray(group_ids, dtype=jnp.int32)
        if group_ids.shape != (n_traj,):
            raise ValueError(
                f"group_ids must have shape {(n_traj,)}, got {group_ids.shape}"
            )
        if num_groups is None:
            raise ValueError("num_groups is required when group_ids are provided")
        if num_groups < 1:
            raise ValueError("num_groups must be positive")

        group_counts = jax.ops.segment_sum(
            jnp.ones((n_traj,), dtype=priorities.dtype),
            group_ids,
            num_segments=num_groups,
        )
        group_priority = jax.ops.segment_sum(
            priorities,
            group_ids,
            num_segments=num_groups,
        )
        group_count_per_motion = group_counts[group_ids]
        group_priority_per_motion = group_priority[group_ids]
        uniform_within_group = 1.0 / jnp.maximum(group_count_per_motion, 1.0)
        difficulty_within_group = jnp.where(
            group_priority_per_motion > 0.0,
            priorities / jnp.maximum(group_priority_per_motion, eps_div),
            uniform_within_group,
        )
        group_mass = jnp.asarray(1.0 / num_groups, dtype=priorities.dtype)
        weights_1d = group_mass * (
            (1.0 - difficulty_mix) * uniform_within_group
            + difficulty_mix * difficulty_within_group
        )

        if max_weight_ratio is not None:
            balanced_uniform = group_mass * uniform_within_group
            cap = max_weight_ratio * balanced_uniform
            capped = jnp.minimum(weights_1d, cap)
            capped_group_mass = jax.ops.segment_sum(
                capped,
                group_ids,
                num_segments=num_groups,
            )
            capacity = jnp.maximum(cap - capped, 0.0)
            group_capacity = jax.ops.segment_sum(
                capacity,
                group_ids,
                num_segments=num_groups,
            )
            deficit = group_mass - capped_group_mass
            weights_1d = capped + capacity * (
                deficit[group_ids]
                / jnp.maximum(group_capacity[group_ids], eps_div)
            )

    return weights_1d, new_ema_done, new_ema_early, rate_hat


def compute_topk_weights(weights_1d: jnp.ndarray, k: int = 10) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get top-k highest weighted trajectories.

    Args:
        weights_1d: 1D array of sampling weights
        k: Number of top trajectories to return

    Returns:
        topk_vals: Top-k weight values
        topk_ids: Trajectory indices of top-k weights
    """
    topk_vals, topk_ids = jax.lax.top_k(weights_1d, k)
    return topk_vals, topk_ids


def compute_per_traj_termination_stats(
    traj_batch_info: dict,
    done: jnp.ndarray,
    absorbing: jnp.ndarray,
    n_trajectories: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute per-trajectory termination stats from a rollout batch.

    Args:
        traj_batch_info: Info dict from traj_batch with 'final_traj_no'
        done: Done flags from rollout, shape (num_steps, num_envs)
        absorbing: Absorbing state flags (early termination), shape (num_steps, num_envs)
        n_trajectories: Total number of trajectories in dataset

    Returns:
        term_rate: per-trajectory early-termination rate (early / done)
        done_counts: per-trajectory done counts
        early_counts: per-trajectory early-termination counts
    """
    final_traj_no = traj_batch_info["final_traj_no"]
    traj_ids = final_traj_no.flatten().astype(jnp.int32)
    done_flat = done.flatten().astype(jnp.float32)
    early_flat = jnp.logical_and(done, absorbing).flatten().astype(jnp.float32)

    done_counts = jax.ops.segment_sum(done_flat, traj_ids, num_segments=n_trajectories)
    early_counts = jax.ops.segment_sum(early_flat, traj_ids, num_segments=n_trajectories)
    term_rate = early_counts / jnp.maximum(done_counts, 1.0)
    return term_rate, done_counts, early_counts


def compute_per_traj_difficulty_stats(
    traj_state,
    done: jnp.ndarray,
    absorbing: jnp.ndarray,
    trajectory_end: jnp.ndarray,
    trajectory_split_points: jnp.ndarray,
    n_trajectories: int,
    late_failure_floor: float = 0.25,
    excluded_transitions: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute progress-aware difficulty evidence for each trajectory.

    Only genuine motion outcomes contribute evidence: an absorbing transition
    is a failure and reaching the end of a trajectory is a success. PPO horizon
    timeouts are deliberately ignored rather than treated as motion successes.
    Failure evidence ranges from 1.0 at the sampled start frame to
    ``late_failure_floor`` at the final frame. Simulator-fault transitions can
    be excluded explicitly so numerical resets are not mistaken for policy
    failures and upweighted by the sampler.

    Returns:
        A tuple of per-trajectory difficulty rate, completed-outcome count, and
        summed progress-aware failure evidence.
    """
    if not 0.0 <= late_failure_floor <= 1.0:
        raise ValueError("late_failure_floor must be between 0 and 1")

    traj_ids = traj_state.traj_no.flatten().astype(jnp.int32)
    current_steps = traj_state.subtraj_step_no.flatten().astype(jnp.float32)
    initial_steps = traj_state.subtraj_step_no_init.flatten().astype(jnp.float32)

    split_points = jnp.asarray(trajectory_split_points)
    trajectory_lengths = split_points[1:] - split_points[:-1]
    final_steps = jnp.maximum(trajectory_lengths[traj_ids] - 1, 0).astype(jnp.float32)
    available_steps = jnp.maximum(final_steps - initial_steps, 1.0)
    progress = jnp.clip((current_steps - initial_steps) / available_steps, 0.0, 1.0)

    done_flat = done.flatten()
    absorbing_flat = absorbing.flatten()
    trajectory_end_flat = trajectory_end.flatten()
    if excluded_transitions is None:
        valid_evidence = jnp.ones_like(done_flat, dtype=bool)
    else:
        valid_evidence = jnp.logical_not(excluded_transitions.flatten())
    failures = done_flat & absorbing_flat & valid_evidence
    outcomes = done_flat & (absorbing_flat | trajectory_end_flat) & valid_evidence
    failure_scores = failures.astype(jnp.float32) * (
        1.0 - (1.0 - late_failure_floor) * progress
    )

    outcome_counts = jax.ops.segment_sum(
        outcomes.astype(jnp.float32), traj_ids, num_segments=n_trajectories
    )
    difficulty_sums = jax.ops.segment_sum(
        failure_scores, traj_ids, num_segments=n_trajectories
    )
    difficulty_rate = difficulty_sums / jnp.maximum(outcome_counts, 1.0)
    return difficulty_rate, outcome_counts, difficulty_sums
