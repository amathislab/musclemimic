"""Policy evaluation for PPO training and full trajectory libraries."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.dataclasses import (
    MetricHandlerTransition,
    ValidationCarry,
    ValidationData,
    ValidationDataFields,
)
from musclemimic.algorithms.common.env_state_utils import unwrap_to_mjx
from musclemimic.algorithms.common.trajectory_evaluation import (
    evaluate_trajectory_set as evaluate_trajectory_set_with_policy,
)
from musclemimic.algorithms.ppo.rollout import reset_from_trajectory_weights
from musclemimic.core.wrappers import LogEnvState
from musclemimic.utils.debug_tools import DebugFlags, maybe_profile_val_batch
from musclemimic.utils.metrics import VALIDATION_STEP_METRIC_KEYS, QuantityContainer, ValidationSummary


class ValidationStepResult(NamedTuple):
    """Metrics and PRNG state produced by one scheduled validation step."""

    metrics: Any
    rng: jax.Array


def evaluate_policy(
    train_state: Any,
    rng: jax.Array,
    env: Any,
    config: Any,
    metrics_handler: Any,
    *,
    reset_weights: jax.Array | None = None,
) -> tuple[Any, jax.Array]:
    """Evaluate the current policy for one configured validation rollout."""
    debug_flags = DebugFlags.from_config(config.get("debug", False))

    def step(carry, _unused):
        env_state, observation, step_rng = carry
        step_rng, action_rng = jax.random.split(step_rng)
        policy, _ = train_state.apply_fn(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            observation,
        )
        action = policy.mode() if config.validation.get("deterministic", False) else policy.sample(seed=action_rng)

        (
            next_observation,
            _reward,
            _absorbing,
            _done,
            _info,
            next_env_state,
            transition_state,
            _transition_observation,
        ) = env.step_with_transition(env_state, action)
        log_env_state = transition_state.find(LogEnvState)
        mjx_state, _ = unwrap_to_mjx(log_env_state.env_state)
        data = mjx_state.data
        site_ids = metrics_handler.rel_site_ids
        validation_data = ValidationData(
            metrics=log_env_state.metrics,
            data=ValidationDataFields(
                qpos=data.qpos,
                qvel=data.qvel,
                xpos=data.xpos,
                xquat=data.xquat,
                cvel=data.cvel,
                subtree_com=data.subtree_com,
                site_xpos=data.site_xpos[:, site_ids, :],
                site_xmat=data.site_xmat[:, site_ids, :, :],
            ),
            additional_carry=ValidationCarry(traj_state=mjx_state.additional_carry.traj_state),
        )
        transition = MetricHandlerTransition(val_data=validation_data)
        return (next_env_state, next_observation, step_rng), transition

    next_rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_rng, config.validation.num_envs)
    if reset_weights is None:
        observation, env_state = env.reset(reset_keys)
    else:
        observation, env_state = reset_from_trajectory_weights(
            env,
            reset_keys,
            reset_weights,
        )

    initial_carry = (env_state, observation, rollout_rng)
    _, rollout = jax.lax.scan(
        step,
        initial_carry,
        None,
        config.validation.num_steps,
    )

    num_sites = metrics_handler.rel_site_ids.shape[0]
    maybe_profile_val_batch(rollout, num_sites, debug_flags)
    local_site_ids = jnp.arange(
        num_sites,
        dtype=metrics_handler.rel_site_ids.dtype,
    )
    metrics = metrics_handler(
        rollout.val_data,
        sim_site_idx=local_site_ids,
    )
    return metrics, next_rng


def _freeze_completed_environments(completed: jax.Array, old_tree: Any, new_tree: Any) -> Any:
    """Keep completed validation lanes unchanged for the rest of the scan."""

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


def _exhaustive_lane_layout(
    *,
    num_trajectories: int,
    trajectory_lengths: tuple[int, ...],
    local_num_envs: int,
    replica_index: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Assign each global validation lane to at most one reference motion."""
    global_lane = replica_index * local_num_envs + jnp.arange(local_num_envs, dtype=jnp.int32)
    active = global_lane < num_trajectories
    trajectory_ids = jnp.minimum(global_lane, num_trajectories - 1)
    lengths = jnp.asarray(trajectory_lengths, dtype=jnp.float32)[trajectory_ids]
    return trajectory_ids, lengths, active


def _gather_exhaustive_lanes(value: jax.Array, *, num_trajectories: int, axis_name: str | None) -> jax.Array:
    """Gather device-local validation lanes in stable global motion order."""
    if axis_name is not None:
        value = jax.lax.all_gather(value, axis_name=axis_name)
        value = value.reshape((-1, *value.shape[2:]))
    return value[:num_trajectories]


def exhaustive_zero_summary(metrics_handler: Any) -> ValidationSummary:
    """Return a skip value with the same per-motion shapes as exhaustive evaluation."""
    num_trajectories = int(metrics_handler._trajectory_handler.n_trajectories)
    zeros = jnp.zeros((num_trajectories,), dtype=jnp.float32)
    empty_quantities = QuantityContainer()
    return metrics_handler.get_zero_container().replace(
        per_motion_success=zeros,
        per_motion_frame_coverage=zeros,
        per_motion_return_per_frame=zeros,
        euclidean_distance=empty_quantities,
        dynamic_time_warping=empty_quantities,
        discrete_frechet_distance=empty_quantities,
        left_arm_euclidean_distance=empty_quantities,
        right_arm_euclidean_distance=empty_quantities,
    )


def evaluate_policy_exhaustive(
    train_state: Any,
    rng: jax.Array,
    env: Any,
    config: Any,
    metrics_handler: Any,
    *,
    axis_name: str | None = None,
) -> tuple[ValidationSummary, jax.Array]:
    """Evaluate every reference motion exactly once from its first frame.

    Global validation lanes are divided across data-parallel replicas. Once a
    lane reaches its first terminal state it is frozen and masked, preventing
    autoresets from being counted as additional episodes.
    """
    trajectory_handler = metrics_handler._trajectory_handler
    num_trajectories = int(trajectory_handler.n_trajectories)
    if num_trajectories < 1:
        raise ValueError("validation.evaluate_all requires at least one trajectory")

    trajectory_lengths = tuple(metrics_handler._trajectory_lengths)
    if len(trajectory_lengths) != num_trajectories:
        raise ValueError(
            "validation trajectory lengths were not materialized before PPO tracing "
            f"({len(trajectory_lengths)} != {num_trajectories})"
        )

    local_num_envs = int(config.validation.num_envs)
    global_num_envs = int(config.validation.get("global_num_envs", local_num_envs))
    if global_num_envs < num_trajectories:
        raise ValueError(
            "global validation.num_envs must be at least the number of trajectories "
            f"when validation.evaluate_all=true ({global_num_envs} < {num_trajectories})"
        )
    if axis_name is None and local_num_envs < num_trajectories:
        raise ValueError(
            "validation.num_envs must be at least the number of trajectories for "
            f"single-device exhaustive validation ({local_num_envs} < {num_trajectories})"
        )

    replica_index = jax.lax.axis_index(axis_name) if axis_name is not None else jnp.asarray(0, dtype=jnp.int32)
    trajectory_ids, local_trajectory_lengths, active = _exhaustive_lane_layout(
        num_trajectories=num_trajectories,
        trajectory_lengths=trajectory_lengths,
        local_num_envs=local_num_envs,
        replica_index=replica_index,
    )
    horizon = int(metrics_handler._validation_horizon)
    deterministic = bool(config.validation.get("deterministic", False))

    next_rng, reset_rng, rollout_rng = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_rng, local_num_envs)
    observation, env_state = env.reset_to(reset_keys, trajectory_ids)
    completed = ~active

    def step(carry, _unused):
        env_state, observation, completed, step_rng = carry
        was_completed = completed
        step_rng, action_rng = jax.random.split(step_rng)
        policy, _ = train_state.apply_fn(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            observation,
        )
        action = policy.mode() if deterministic else policy.sample(seed=action_rng)
        action = jnp.where(was_completed[:, None], 0.0, action)

        (
            next_observation,
            reward,
            absorbing,
            done,
            info,
            next_env_state,
            _transition_state,
            _transition_observation,
        ) = env.step_with_transition(env_state, action)
        valid = ~was_completed
        completed = was_completed | done
        next_env_state = _freeze_completed_environments(was_completed, env_state, next_env_state)
        next_observation = jnp.where(was_completed[:, None], observation, next_observation)
        step_metrics = {key: info[key] for key in VALIDATION_STEP_METRIC_KEYS if key in info}
        output = {
            "reward": jnp.where(valid, reward, 0.0),
            "done": done & valid,
            "valid": valid,
            "absorbing": absorbing,
            "info": step_metrics,
        }
        return (next_env_state, next_observation, completed, step_rng), output

    _, rollout = jax.lax.scan(
        step,
        (env_state, observation, completed, rollout_rng),
        None,
        length=horizon,
    )

    valid = rollout["valid"]
    done = rollout["done"]
    local_lengths = jnp.sum(valid, axis=0).astype(jnp.float32)
    local_returns = jnp.sum(rollout["reward"], axis=0)
    local_finished = jnp.any(done, axis=0)
    local_early = jnp.any(done & rollout["absorbing"], axis=0)
    local_success = active & local_finished & ~local_early
    local_coverage = jnp.minimum(local_lengths / jnp.maximum(local_trajectory_lengths, 1.0), 1.0)
    local_return_per_frame = local_returns / jnp.maximum(local_lengths, 1.0)

    lengths = _gather_exhaustive_lanes(local_lengths, num_trajectories=num_trajectories, axis_name=axis_name)
    returns = _gather_exhaustive_lanes(local_returns, num_trajectories=num_trajectories, axis_name=axis_name)
    finished = _gather_exhaustive_lanes(local_finished, num_trajectories=num_trajectories, axis_name=axis_name)
    early = _gather_exhaustive_lanes(local_early, num_trajectories=num_trajectories, axis_name=axis_name)
    success = _gather_exhaustive_lanes(local_success, num_trajectories=num_trajectories, axis_name=axis_name)
    coverage = _gather_exhaustive_lanes(local_coverage, num_trajectories=num_trajectories, axis_name=axis_name)
    return_per_frame = _gather_exhaustive_lanes(
        local_return_per_frame,
        num_trajectories=num_trajectories,
        axis_name=axis_name,
    )

    total_valid_steps = jnp.sum(valid.astype(jnp.float32))
    if axis_name is not None:
        total_valid_steps = jax.lax.psum(total_valid_steps, axis_name=axis_name)
    total_valid_steps = jnp.maximum(total_valid_steps, 1.0)

    step_means = {}
    for key, values in rollout["info"].items():
        mask = valid
        while mask.ndim < values.ndim:
            mask = mask[..., None]
        numerator = jnp.sum(jnp.where(mask, values, 0.0))
        if axis_name is not None:
            numerator = jax.lax.psum(numerator, axis_name=axis_name)
        elements_per_step = max(1, math.prod(values.shape[2:]))
        step_means[key] = numerator / (total_valid_steps * elements_per_step)

    summary_values = {
        key: step_means.get(key, jnp.asarray(0.0, dtype=jnp.float32))
        for key in VALIDATION_STEP_METRIC_KEYS
        if key in ValidationSummary.__dataclass_fields__
    }
    quartile_size = max(1, (num_trajectories + 3) // 4)
    empty_quantities = QuantityContainer()
    return (
        ValidationSummary(
            episode_count=jnp.asarray(num_trajectories, dtype=jnp.float32),
            mean_episode_return=jnp.mean(returns),
            mean_episode_length=jnp.mean(lengths),
            max_timestep=jnp.max(lengths).astype(jnp.int32),
            early_termination_count=jnp.sum(early.astype(jnp.float32)),
            early_termination_rate=jnp.mean(early.astype(jnp.float32)),
            horizon_timeout_count=jnp.sum((~finished).astype(jnp.float32)),
            horizon_timeout_rate=jnp.mean((~finished).astype(jnp.float32)),
            motion_success_rate=jnp.mean(success.astype(jnp.float32)),
            mean_frame_coverage=jnp.mean(coverage),
            worst_quartile_frame_coverage=jnp.mean(jnp.sort(coverage)[:quartile_size]),
            mean_return_per_frame=jnp.mean(return_per_frame),
            motion_count=jnp.asarray(num_trajectories, dtype=jnp.float32),
            per_motion_success=success.astype(jnp.float32),
            per_motion_frame_coverage=coverage,
            per_motion_return_per_frame=return_per_frame,
            euclidean_distance=empty_quantities,
            dynamic_time_warping=empty_quantities,
            discrete_frechet_distance=empty_quantities,
            left_arm_euclidean_distance=empty_quantities,
            right_arm_euclidean_distance=empty_quantities,
            **summary_values,
        ),
        next_rng,
    )


def reduce_validation_summary(summary: Any, *, axis_name: str) -> Any:
    """Combine validation summaries across data-parallel replicas."""

    rmse_fields = {
        "err_root_xyz",
        "err_joint_pos",
        "err_joint_vel",
        "err_rpos",
    }

    def sum_across_devices(value):
        return jax.lax.psum(value, axis_name=axis_name)

    def reduce(path, value):
        name = path[-1].name
        if name.endswith("_count"):
            return sum_across_devices(value)
        if name == "max_timestep":
            return jax.lax.pmax(value, axis_name=axis_name)
        if name in rmse_fields:
            return jnp.sqrt(jax.lax.pmean(jnp.square(value), axis_name=axis_name))
        return jax.lax.pmean(value, axis_name=axis_name)

    reduced = jax.tree_util.tree_map_with_path(reduce, summary)
    local_count = summary.episode_count
    global_count = jnp.maximum(reduced.episode_count, 1.0)
    return reduced.replace(
        mean_episode_return=sum_across_devices(summary.mean_episode_return * local_count) / global_count,
        mean_episode_length=sum_across_devices(summary.mean_episode_length * local_count) / global_count,
        early_termination_rate=reduced.early_termination_count / global_count,
        horizon_timeout_rate=reduced.horizon_timeout_count / global_count,
        reward_termination_rate=reduced.reward_termination_count / global_count,
    )


def run_validation_step(
    train_state: Any,
    rng: jax.Array,
    should_validate: jax.Array,
    env: Any,
    config: Any,
    metrics_handler: Any | None,
    *,
    reset_weights: jax.Array | None = None,
    axis_name: str | None = None,
) -> ValidationStepResult:
    """Run or skip one validation update while preserving its PRNG stream."""
    evaluate_all = bool(config.validation.get("evaluate_all", False)) if config is not None else False
    if metrics_handler is None:
        metrics = ValidationSummary()
        next_rng = jax.lax.cond(
            should_validate,
            lambda key: jax.random.split(key, 3)[0],
            lambda key: key,
            rng,
        )
    elif evaluate_all:
        metrics, next_rng = jax.lax.cond(
            should_validate,
            lambda key: evaluate_policy_exhaustive(
                train_state,
                key,
                env,
                config,
                metrics_handler,
                axis_name=axis_name,
            ),
            lambda key: (exhaustive_zero_summary(metrics_handler), key),
            rng,
        )
    else:
        metrics, next_rng = jax.lax.cond(
            should_validate,
            lambda key: evaluate_policy(
                train_state,
                key,
                env,
                config,
                metrics_handler,
                reset_weights=reset_weights,
            ),
            lambda key: (metrics_handler.get_zero_container(), key),
            rng,
        )

    if axis_name is not None and not evaluate_all:
        metrics = reduce_validation_summary(metrics, axis_name=axis_name)
    return ValidationStepResult(metrics=metrics, rng=next_rng)


def evaluate_trajectory_set(
    network: Any,
    params: Any,
    run_stats: Any,
    trajectory_env: Any,
    env: Any,
    num_envs: int,
    deterministic: bool,
    eval_seed: int = 0,
) -> dict[str, float]:
    """Evaluate every trajectory using batched MJX rollouts."""

    def policy_action(policy_state, observation, rng):
        policy, _ = network.apply(
            {"params": policy_state["params"], "run_stats": policy_state["run_stats"]},
            observation,
        )
        action = policy.mode() if deterministic else policy.sample(seed=rng)
        return action, policy_state

    return evaluate_trajectory_set_with_policy(
        policy_action=policy_action,
        policy_state={"params": params, "run_stats": run_stats},
        trajectory_env=trajectory_env,
        env=env,
        num_envs=num_envs,
        eval_seed=eval_seed,
    )
