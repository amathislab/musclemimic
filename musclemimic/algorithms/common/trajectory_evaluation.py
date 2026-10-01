"""Batched MJX evaluation over a complete trajectory library."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.distributed import mark_data_varying
from musclemimic.utils.metrics import VALIDATION_STEP_METRIC_KEYS, ValidationSummary

PolicyAction = Callable[[Any, jax.Array, jax.Array], tuple[jax.Array, Any]]


class ExactTrajectorySetResult(NamedTuple):
    """Summary and advanced PRNG state for one fixed trajectory set."""

    metrics: ValidationSummary
    rng: jax.Array


class ExactTrajectorySetStatistics(NamedTuple):
    """Additive shard-local statistics for exact trajectory evaluation."""

    episode_count: jax.Array
    episode_return_sum: jax.Array
    episode_length_sum: jax.Array
    early_termination_count: jax.Array
    horizon_timeout_count: jax.Array
    covered_frame_sum: jax.Array
    total_frame: jax.Array
    step_count: jax.Array
    step_sums: dict[str, jax.Array]


class ExactTrajectorySetLocalResult(NamedTuple):
    """Shard-local exact statistics and advanced PRNG state."""

    metrics: ExactTrajectorySetStatistics
    rng: jax.Array


def _freeze_completed_environments(completed, old_tree, new_tree):
    """Keep completed environments unchanged inside a batched PyTree."""

    def select(old_value, new_value):
        if (
            hasattr(new_value, "shape")
            and new_value.shape
            and completed.ndim > 0
            and new_value.shape[0] == completed.shape[0]
        ):
            mask = completed
            while mask.ndim < new_value.ndim:
                mask = mask[..., None]
            return jnp.where(mask, old_value, new_value)
        return new_value

    return jax.tree.map(select, old_tree, new_tree)


@functools.partial(jax.jit, static_argnums=(0,))
def _reset_trajectory_batch(env, reset_keys, trajectory_ids):
    return env.reset_to(reset_keys, trajectory_ids)


def _rollout_trajectory_batch(
    policy_action: PolicyAction,
    policy_state: Any,
    env: Any,
    observation: jax.Array,
    env_state: Any,
    horizon: int,
    rng: jax.Array,
    *,
    axis_name: str | None = None,
):
    num_envs = observation.shape[0]
    completed = jnp.zeros(num_envs, dtype=bool)
    observation, env_state, completed = mark_data_varying(
        (observation, env_state, completed),
        axis_name,
    )

    def step(carry, _unused):
        observation, env_state, completed, step_rng = carry
        was_completed = completed
        step_rng, action_rng = jax.random.split(step_rng)
        action, _ = policy_action(policy_state, observation, action_rng)
        action = jnp.where(was_completed[:, None], 0.0, action)

        next_observation, reward, absorbing, done, info, next_env_state = env.step(
            env_state,
            action,
        )
        valid = ~was_completed
        completed = was_completed | done
        next_env_state = _freeze_completed_environments(
            was_completed,
            env_state,
            next_env_state,
        )
        next_observation = jnp.where(
            was_completed[:, None],
            observation,
            next_observation,
        )
        reward = jnp.where(valid, reward, 0.0)
        metric_info = {key: info[key] for key in VALIDATION_STEP_METRIC_KEYS if key in info}
        carry = (next_observation, next_env_state, completed, step_rng)
        step_data = {
            "reward": reward,
            "done": done,
            "valid_mask": valid,
            "absorbing": absorbing,
            "info": metric_info,
        }
        return carry, step_data

    initial_carry = (observation, env_state, completed, rng)
    _, rollout = jax.lax.scan(step, initial_carry, None, horizon)
    return rollout


def _build_trajectory_rollout(policy_action, env, horizon):
    @jax.jit
    def rollout(policy_state, observation, env_state, rng):
        return _rollout_trajectory_batch(
            policy_action,
            policy_state,
            env,
            observation,
            env_state,
            horizon,
            rng,
        )

    return rollout


def _summarize_trajectory_batch(rollout, trajectory_lengths):
    rewards = rollout["reward"]
    dones = rollout["done"]
    valid = rollout["valid_mask"]
    absorbing = rollout["absorbing"]

    episode_returns = jnp.sum(rewards, axis=0)
    any_done = jnp.any(dones, axis=0)
    first_done_step = jnp.argmax(dones, axis=0) + 1
    episode_lengths = jnp.where(any_done, first_done_step, dones.shape[0])
    terminal_step = jnp.clip(first_done_step - 1, 0, dones.shape[0] - 1)
    terminal_absorbing = absorbing[terminal_step, jnp.arange(absorbing.shape[1])]
    early_terminated = any_done & terminal_absorbing & (episode_lengths < trajectory_lengths)
    frame_coverage = episode_lengths / jnp.maximum(trajectory_lengths, 1)

    valid_steps = jnp.sum(valid, axis=0)
    step_metrics = {}
    for key, value in rollout["info"].items():
        if hasattr(value, "shape") and value.ndim >= 2 and value.shape[1] == trajectory_lengths.shape[0]:
            masked_sum = jnp.sum(jnp.where(valid, value, jnp.zeros_like(value)), axis=0)
            step_metrics[key] = masked_sum / jnp.maximum(valid_steps, 1)

    return {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "completed": any_done,
        "early_terminated": early_terminated,
        "frame_coverage": frame_coverage,
        "step_metrics": step_metrics,
    }


def _summarize_exact_trajectory_set_local(
    rollout: Any,
    trajectory_lengths: jax.Array,
    trajectory_mask: jax.Array,
) -> ExactTrajectorySetStatistics:
    """Build sufficient statistics without any cross-device operations."""
    batch = _summarize_trajectory_batch(rollout, trajectory_lengths)
    trajectory_mask = trajectory_mask.astype(jnp.float32)
    covered_frames = jnp.minimum(
        batch["episode_lengths"],
        trajectory_lengths,
    ).astype(jnp.float32)
    step_is_valid = rollout["valid_mask"] & trajectory_mask.astype(bool)[None, :]
    step_mask = step_is_valid.astype(jnp.float32)
    return ExactTrajectorySetStatistics(
        episode_count=jnp.sum(trajectory_mask),
        episode_return_sum=jnp.sum(batch["episode_returns"] * trajectory_mask),
        episode_length_sum=jnp.sum(batch["episode_lengths"] * trajectory_mask),
        early_termination_count=jnp.sum(batch["early_terminated"].astype(jnp.float32) * trajectory_mask),
        horizon_timeout_count=jnp.sum((~batch["completed"]).astype(jnp.float32) * trajectory_mask),
        covered_frame_sum=jnp.sum(covered_frames * trajectory_mask),
        total_frame=jnp.sum(trajectory_lengths.astype(jnp.float32) * trajectory_mask),
        step_count=jnp.sum(step_mask),
        step_sums={
            key: jnp.sum(jnp.where(step_is_valid, value, jnp.zeros_like(value)))
            for key, value in rollout["info"].items()
        },
    )


def reduce_exact_trajectory_set_statistics(
    statistics: ExactTrajectorySetStatistics,
    *,
    axis_name: str | None,
) -> ValidationSummary:
    """Reduce exact-validation statistics after every shard has finished."""

    def sum_across_devices(value: jax.Array) -> jax.Array:
        if axis_name is None:
            return value
        return jax.lax.psum(value, axis_name=axis_name)

    episode_count = sum_across_devices(statistics.episode_count)
    episode_return_sum = sum_across_devices(statistics.episode_return_sum)
    episode_length_sum = sum_across_devices(statistics.episode_length_sum)
    early_count = sum_across_devices(statistics.early_termination_count)
    horizon_count = sum_across_devices(statistics.horizon_timeout_count)
    covered_frame_sum = sum_across_devices(statistics.covered_frame_sum)
    total_frame = sum_across_devices(statistics.total_frame)
    step_count = sum_across_devices(statistics.step_count)
    step_sums = jax.tree.map(sum_across_devices, statistics.step_sums)

    safe_episode_count = jnp.maximum(episode_count, 1.0)
    safe_step_count = jnp.maximum(step_count, 1.0)
    step_means = {key: value / safe_step_count for key, value in step_sums.items()}

    def step_mean(name: str) -> jax.Array:
        return step_means.get(name, jnp.asarray(0.0, dtype=jnp.float32))

    return ValidationSummary(
        episode_count=episode_count,
        mean_episode_return=episode_return_sum / safe_episode_count,
        mean_episode_length=episode_length_sum / safe_episode_count,
        early_termination_count=early_count,
        early_termination_rate=early_count / safe_episode_count,
        horizon_timeout_count=horizon_count,
        horizon_timeout_rate=horizon_count / safe_episode_count,
        frame_coverage=covered_frame_sum / jnp.maximum(total_frame, 1.0),
        total_frame=total_frame,
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


def evaluate_exact_trajectory_set_local(
    policy_action: Callable[[jax.Array, jax.Array], jax.Array],
    rng: jax.Array,
    env: Any,
    *,
    trajectory_ids: jax.Array,
    trajectory_lengths: jax.Array,
    trajectory_mask: jax.Array,
    num_steps: int,
    axis_name: str | None,
) -> ExactTrajectorySetLocalResult:
    """Evaluate one trajectory shard without cross-device collectives."""

    def stateful_policy(_policy_state, observation, action_rng):
        return policy_action(observation, action_rng), None

    next_rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_rng, trajectory_ids.shape[0])
    observation, env_state = env.reset_to(
        reset_keys,
        trajectory_ids,
    )
    rollout = _rollout_trajectory_batch(
        stateful_policy,
        None,
        env,
        observation,
        env_state,
        int(num_steps),
        rollout_rng,
        axis_name=axis_name,
    )
    metrics = _summarize_exact_trajectory_set_local(
        rollout,
        trajectory_lengths,
        trajectory_mask,
    )
    return ExactTrajectorySetLocalResult(metrics=metrics, rng=next_rng)


def evaluate_exact_trajectory_set(
    policy_action: Callable[[jax.Array, jax.Array], jax.Array],
    rng: jax.Array,
    env: Any,
    *,
    trajectory_ids: jax.Array,
    trajectory_lengths: jax.Array,
    trajectory_mask: jax.Array,
    num_steps: int,
    axis_name: str | None,
) -> ExactTrajectorySetResult:
    """Evaluate each selected trajectory once from frame zero.

    ``trajectory_mask`` excludes padding slots used to keep a common batch
    shape across devices. Trajectories that outlive ``num_steps`` are reported
    as horizon timeouts, not tracking failures.
    """

    local_result = evaluate_exact_trajectory_set_local(
        policy_action,
        rng,
        env,
        trajectory_ids=trajectory_ids,
        trajectory_lengths=trajectory_lengths,
        trajectory_mask=trajectory_mask,
        num_steps=num_steps,
        axis_name=axis_name,
    )
    metrics = reduce_exact_trajectory_set_statistics(
        local_result.metrics,
        axis_name=axis_name,
    )
    return ExactTrajectorySetResult(metrics=metrics, rng=local_result.rng)


def evaluate_trajectory_set(
    policy_action: PolicyAction,
    policy_state: Any,
    trajectory_env: Any,
    env: Any,
    num_envs: int,
    eval_seed: int = 0,
) -> dict[str, float]:
    """Evaluate every trajectory using a batched policy action function."""
    num_trajectories = int(trajectory_env.th.n_trajectories)
    rng = jax.random.key(eval_seed)
    horizon = max(int(trajectory_env.th.len_trajectory(index)) for index in range(num_trajectories))
    rollout_batch = _build_trajectory_rollout(policy_action, env, horizon)

    returns = []
    lengths = []
    early_terminations = []
    coverage = []
    trajectory_lengths = []
    step_metric_sums: dict[str, float] = {}
    total_valid_steps = 0

    for batch_start in range(0, num_trajectories, num_envs):
        batch_end = min(batch_start + num_envs, num_trajectories)
        trajectory_ids = list(range(batch_start, batch_end))
        batch_lengths = [int(trajectory_env.th.len_trajectory(index)) for index in trajectory_ids]
        batch_size = len(trajectory_ids)

        rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
        reset_keys = jax.random.split(reset_rng, batch_size)
        trajectory_ids_array = jnp.asarray(trajectory_ids, dtype=jnp.int32)
        observation, env_state = _reset_trajectory_batch(
            env,
            reset_keys,
            trajectory_ids_array,
        )
        batch_lengths_array = jnp.asarray(batch_lengths, dtype=jnp.int32)

        rollout = rollout_batch(
            policy_state,
            observation,
            env_state,
            rollout_rng,
        )
        batch_metrics = jax.device_get(
            _summarize_trajectory_batch(
                rollout,
                batch_lengths_array,
            )
        )

        for local_index, trajectory_id in enumerate(trajectory_ids):
            episode_return = float(batch_metrics["episode_returns"][local_index])
            episode_length = int(batch_metrics["episode_lengths"][local_index])
            early = bool(batch_metrics["early_terminated"][local_index])
            trajectory_length = batch_lengths[local_index]
            frame_coverage = float(batch_metrics["frame_coverage"][local_index])

            returns.append(episode_return)
            lengths.append(episode_length)
            early_terminations.append(early)
            coverage.append(frame_coverage)
            trajectory_lengths.append(trajectory_length)
            total_valid_steps += episode_length

            for key, per_environment_mean in batch_metrics["step_metrics"].items():
                weighted_value = float(per_environment_mean[local_index]) * episode_length
                step_metric_sums[key] = step_metric_sums.get(key, 0.0) + weighted_value

            suffix = f"  EARLY at {episode_length}/{trajectory_length}" if early else ""
            print(
                f"  traj {trajectory_id + 1}/{num_trajectories}: "
                f"len={episode_length}, return={episode_return:.4f}{suffix}",
                flush=True,
            )

    returns_array = jnp.asarray(returns)
    lengths_array = jnp.asarray(lengths)
    early_array = jnp.asarray(early_terminations)
    coverage_array = jnp.asarray(coverage)
    trajectory_lengths_array = jnp.asarray(trajectory_lengths)
    total_frames = float(jnp.sum(trajectory_lengths_array))

    metrics = {
        "val_mean_episode_return": float(jnp.mean(returns_array)),
        "val_mean_episode_length": float(jnp.mean(lengths_array)),
        "val_early_termination_count": float(jnp.sum(early_array)),
        "val_early_termination_rate": float(jnp.mean(early_array)),
        "val_frame_coverage": (
            float(jnp.sum(coverage_array * trajectory_lengths_array)) / total_frames if total_frames > 0 else 0.0
        ),
        "val_total_frame": total_frames,
    }
    if total_valid_steps > 0:
        for key, weighted_sum in step_metric_sums.items():
            metrics[f"val_{key}"] = weighted_sum / total_valid_steps

    print(
        f"Completed {num_trajectories} trajectories, {int(metrics['val_early_termination_count'])} early terminations"
    )
    return metrics
