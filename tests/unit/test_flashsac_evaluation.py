"""Tests for FlashSAC validation behavior."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.common.trajectory_evaluation import (
    ExactTrajectorySetLocalResult,
    ExactTrajectorySetStatistics,
)
from musclemimic.algorithms.flashsac import evaluation
from musclemimic.algorithms.flashsac.state import FlashSACPolicyState
from musclemimic.utils.metrics import ValidationSummary


class _Actor:
    @staticmethod
    def apply(_variables, observations, *, training):
        assert training is False
        shape = (observations.shape[0], 1)
        return jnp.zeros(shape), jnp.full(shape, 0.3)


def _agent_conf():
    return SimpleNamespace(
        actor=_Actor(),
        actor_observation_indices=jnp.asarray([0, 1], dtype=jnp.int32),
    )


def _policy_state():
    return FlashSACPolicyState(actor_params={}, actor_batch_stats={})


def test_exact_validation_passes_one_slot_per_trajectory(monkeypatch):
    captured = {}

    def evaluate_exact(policy_action, rng, env, **kwargs):
        del env
        captured.update(kwargs)
        action = policy_action(jnp.ones((3, 2)), jax.random.key(8))
        assert action.shape == (3, 1)
        assert jnp.all(action == 0.0)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return ExactTrajectorySetLocalResult(
            metrics=ExactTrajectorySetStatistics(
                episode_count=jnp.sum(kwargs["trajectory_mask"]),
                episode_return_sum=zero,
                episode_length_sum=zero,
                early_termination_count=zero,
                horizon_timeout_count=zero,
                covered_frame_sum=zero,
                total_frame=zero,
                step_count=zero,
                step_sums={},
            ),
            rng=jax.random.fold_in(rng, 1),
        )

    monkeypatch.setattr(
        evaluation,
        "evaluate_exact_trajectory_set_local",
        evaluate_exact,
    )
    trajectory_ids = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    trajectory_lengths = jnp.asarray([10, 20, 30], dtype=jnp.int32)
    trajectory_mask = jnp.asarray([True, True, False])

    result = evaluation.run_validation_step(
        _agent_conf(),
        _policy_state(),
        jax.random.key(3),
        env=object(),
        config=OmegaConf.create(
            {
                "validation": {
                    "deterministic": True,
                    "evaluate_all": True,
                    "num_steps": 100,
                }
            }
        ),
        metrics_handler=object(),
        trajectory_ids=trajectory_ids,
        trajectory_lengths=trajectory_lengths,
        trajectory_mask=trajectory_mask,
    )

    assert result.metrics.episode_count == 2
    assert captured["num_steps"] == 100
    assert captured["axis_name"] is None
    assert jnp.array_equal(captured["trajectory_ids"], trajectory_ids)
    assert jnp.array_equal(captured["trajectory_lengths"], trajectory_lengths)
    assert jnp.array_equal(captured["trajectory_mask"], trajectory_mask)


def test_exact_validation_rejects_missing_trajectory_schedule():
    with pytest.raises(ValueError, match="trajectory ids"):
        evaluation.run_validation_step(
            _agent_conf(),
            _policy_state(),
            jax.random.key(3),
            env=object(),
            config=OmegaConf.create(
                {
                    "validation": {
                        "deterministic": True,
                        "evaluate_all": True,
                        "num_steps": 100,
                    }
                }
            ),
            metrics_handler=object(),
        )
