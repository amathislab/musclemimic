"""Policy evaluation for PPO training and full trajectory libraries."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from musclemimic.algorithms.common.trajectory_evaluation import (
    evaluate_exact_trajectory_set_local,
)
from musclemimic.algorithms.common.trajectory_evaluation import (
    evaluate_trajectory_set as evaluate_trajectory_set_with_policy,
)
from musclemimic.algorithms.common.validation import (
    run_validation_rollout,
)
from musclemimic.utils.debug_tools import DebugFlags, maybe_profile_val_batch
from musclemimic.utils.metrics import ValidationSummary


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
    trajectory_ids: jax.Array | None = None,
    trajectory_lengths: jax.Array | None = None,
    trajectory_mask: jax.Array | None = None,
    axis_name: str | None = None,
) -> tuple[Any, jax.Array]:
    """Evaluate the current policy for one configured validation rollout."""
    debug_flags = DebugFlags.from_config(config.get("debug", False))

    def policy_action(observation: jax.Array, action_rng: jax.Array) -> jax.Array:
        policy, _ = train_state.apply_fn(
            {"params": train_state.params, "run_stats": train_state.run_stats},
            observation,
        )
        if config.validation.get("deterministic", False):
            return policy.mode()
        return policy.sample(seed=action_rng)

    evaluate_all = bool(config.validation.get("evaluate_all", False))
    if evaluate_all:
        if trajectory_ids is None or trajectory_lengths is None or trajectory_mask is None:
            raise ValueError("Exact validation requires trajectory ids, lengths, and a padding mask.")
        exact_result = evaluate_exact_trajectory_set_local(
            policy_action,
            rng,
            env,
            trajectory_ids=trajectory_ids,
            trajectory_lengths=trajectory_lengths,
            trajectory_mask=trajectory_mask,
            num_steps=int(config.validation.num_steps),
            axis_name=axis_name,
        )
        return exact_result.metrics, exact_result.rng

    rollout_result = run_validation_rollout(
        policy_action=policy_action,
        rng=rng,
        env=env,
        config=config,
        metrics_handler=metrics_handler,
        reset_weights=reset_weights,
        axis_name=axis_name,
    )

    num_sites = metrics_handler.rel_site_ids.shape[0]
    maybe_profile_val_batch(rollout_result.rollout, num_sites, debug_flags)
    return rollout_result.metrics, rollout_result.rng


def run_validation_step(
    train_state: Any,
    rng: jax.Array,
    should_validate: jax.Array | bool,
    env: Any,
    config: Any,
    metrics_handler: Any | None,
    *,
    reset_weights: jax.Array | None = None,
    trajectory_ids: jax.Array | None = None,
    trajectory_lengths: jax.Array | None = None,
    trajectory_mask: jax.Array | None = None,
    axis_name: str | None = None,
) -> ValidationStepResult:
    """Run or skip one validation update while preserving its PRNG stream."""
    validation_config = getattr(config, "validation", {}) if config is not None else {}
    evaluate_all = bool(validation_config.get("evaluate_all", False))
    if metrics_handler is None:
        metrics = ValidationSummary()
        if isinstance(should_validate, bool):
            next_rng = jax.random.split(rng, 3)[0] if should_validate else rng
        else:
            next_rng = jax.lax.cond(
                should_validate,
                lambda key: jax.random.split(key, 3)[0],
                lambda key: key,
                rng,
            )
    else:

        def run_validation(key):
            if not evaluate_all:
                return evaluate_policy(
                    train_state,
                    key,
                    env,
                    config,
                    metrics_handler,
                    reset_weights=reset_weights,
                    axis_name=axis_name,
                )
            return evaluate_policy(
                train_state,
                key,
                env,
                config,
                metrics_handler,
                reset_weights=reset_weights,
                trajectory_ids=trajectory_ids,
                trajectory_lengths=trajectory_lengths,
                trajectory_mask=trajectory_mask,
                axis_name=axis_name,
            )

        def skip_validation(key):
            if evaluate_all:
                # A traced condition requires the same statistics tree on both branches.
                template, _ = jax.eval_shape(run_validation, key)
                return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), template), key
            return metrics_handler.get_zero_container(), key

        if isinstance(should_validate, bool):
            metrics, next_rng = run_validation(rng) if should_validate else skip_validation(rng)
        else:
            metrics, next_rng = jax.lax.cond(
                should_validate,
                run_validation,
                skip_validation,
                rng,
            )

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
