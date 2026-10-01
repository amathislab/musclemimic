"""Transformation tests for the numerical PPO update boundary."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.common.curriculum import (
    create_curriculum_state,
    create_reward_curriculum_state,
)
from musclemimic.algorithms.ppo import update as ppo_update
from musclemimic.algorithms.ppo.learning import PPOUpdateMetrics
from musclemimic.algorithms.ppo.state import PPORuntimeState
from musclemimic.core.wrappers import SummaryMetrics
from musclemimic.utils.debug_tools import DebugFlags


@struct.dataclass
class _TestTrainState:
    params: dict
    run_stats: jax.Array
    step: jax.Array


class _RolloutMetrics(NamedTuple):
    done: jax.Array


class _TrajectoryBatch(NamedTuple):
    metrics: _RolloutMetrics
    absorbing: jax.Array
    done: jax.Array
    traj_end: jax.Array
    reward: jax.Array
    value: jax.Array
    obs: jax.Array
    info: dict


class _TestNetwork:
    @staticmethod
    def apply(_variables, observation):
        return None, jnp.zeros(observation.shape[:-1], dtype=jnp.float32)


def _metrics() -> PPOUpdateMetrics:
    values = [jnp.asarray(index, dtype=jnp.float32) for index in range(16)]
    return PPOUpdateMetrics(*values)


def _state() -> PPORuntimeState:
    return PPORuntimeState(
        train_state=_TestTrainState(
            params={
                "weight": jnp.asarray(2.0, dtype=jnp.float32),
                "log_std": jnp.asarray([0.0], dtype=jnp.float32),
            },
            run_stats=jnp.asarray(0.0, dtype=jnp.float32),
            step=jnp.asarray(4, dtype=jnp.int32),
        ),
        env_state=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        last_obs=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        rng=jnp.asarray([11, 12], dtype=jnp.uint32),
        lr=jnp.asarray(3e-4, dtype=jnp.float32),
        val_rng=jnp.asarray([13, 14], dtype=jnp.uint32),
        curriculum_state=create_curriculum_state(0.3, 1.0),
        reward_curriculum_state=create_reward_curriculum_state({}),
    )


def _install_update_fakes(monkeypatch) -> None:
    monkeypatch.setattr(
        ppo_update,
        "update_carry_reward_weights_unnormalized",
        lambda env_state, qvel_weight, root_velocity_weight: env_state + qvel_weight + root_velocity_weight,
    )

    def collect_rollout(train_state, env_state, observation, rng, **_kwargs):
        next_observation = observation + 1.0
        done = jnp.asarray([[True, False]])
        trajectory = _TrajectoryBatch(
            metrics=_RolloutMetrics(done=done),
            absorbing=jnp.asarray([[True, False]]),
            done=done,
            traj_end=jnp.asarray([[False, False]]),
            reward=jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
            value=jnp.asarray([[0.25, 0.5]], dtype=jnp.float32),
            obs=observation[None],
            info={},
        )
        next_state = (train_state, env_state + 1.0, next_observation, rng + 1)
        return next_state, trajectory

    monkeypatch.setattr(ppo_update, "collect_rollout", collect_rollout)
    monkeypatch.setattr(
        ppo_update,
        "compute_gae",
        lambda reward, value, *_args, **_kwargs: (reward - value, reward + value),
    )

    def update(train_state, _batch, _advantages, _targets, rng, learning_rate, **_kwargs):
        return (
            train_state.replace(step=train_state.step + 1),
            rng + 2,
            learning_rate * 0.5,
            _metrics(),
        )

    monkeypatch.setattr(ppo_update, "update", update)
    monkeypatch.setattr(
        ppo_update,
        "update_observation_stats",
        lambda train_state, observations, **_kwargs: train_state.replace(
            run_stats=train_state.run_stats + jnp.sum(observations),
        ),
    )
    monkeypatch.setattr(
        ppo_update,
        "summarize_training_rollout",
        lambda *_args, **_kwargs: SummaryMetrics(),
    )


def _make_update(axis_name: str | None):
    config = OmegaConf.create(
        {
            "normalize_env": False,
            "num_steps": 1,
            "num_envs": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "adaptive_sampling": {"enabled": False},
            "reward_curriculum": {"enabled": False},
            "anneal_lr": False,
            "num_minibatches": 1,
            "update_epochs": 1,
            "num_updates": 10,
        }
    )
    return partial(
        ppo_update.run_ppo_update,
        env=object(),
        network=_TestNetwork(),
        config=config,
        debug_flags=DebugFlags(),
        base_learning_rate=3e-4,
        use_adaptive_learning_rate=False,
        adaptive_termination_enabled=False,
        curriculum_params=None,
        shard_spec=None,
        axis_name=axis_name,
    )


def _assert_trees_close(actual, expected) -> None:
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    assert actual_tree == expected_tree
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(actual_leaf), np.asarray(expected_leaf))


def test_ppo_update_has_equivalent_eager_and_jit_results(monkeypatch):
    _install_update_fakes(monkeypatch)
    update = _make_update(axis_name=None)

    eager = update(_state())
    compiled = jax.jit(update)(_state())

    _assert_trees_close(compiled, eager)
    assert int(eager.runtime_state.train_state.step) == 5
    np.testing.assert_allclose(float(eager.diagnostics.early_termination_rate), 1.0)


def test_ppo_update_matches_characterized_legacy_state_and_metrics(monkeypatch):
    """Lock the state/RNG/metric contract extracted from the former scan body."""
    _install_update_fakes(monkeypatch)

    result = _make_update(axis_name=None)(_state())

    assert int(result.runtime_state.train_state.step) == 5
    np.testing.assert_allclose(result.runtime_state.train_state.run_stats, 14.0)
    np.testing.assert_allclose(result.runtime_state.env_state, [1.4, 2.4])
    np.testing.assert_allclose(
        result.runtime_state.last_obs,
        [[2.0, 3.0], [4.0, 5.0]],
    )
    np.testing.assert_array_equal(result.runtime_state.rng, [14, 15])
    np.testing.assert_array_equal(result.runtime_state.val_rng, [13, 14])
    np.testing.assert_allclose(result.runtime_state.lr, 1.5e-4)
    np.testing.assert_allclose(
        np.asarray(result.ppo_metrics),
        np.arange(16, dtype=np.float32),
    )
    assert int(result.counter) == 6


def test_ppo_update_is_valid_under_pmap(monkeypatch):
    _install_update_fakes(monkeypatch)
    state = _state()
    num_devices = jax.local_device_count()
    replicated = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (num_devices, *value.shape)),
        state,
    )

    updated = jax.pmap(_make_update(axis_name="devices"), axis_name="devices")(replicated)

    assert np.asarray(updated.runtime_state.train_state.step).tolist() == [5] * num_devices
    np.testing.assert_allclose(
        np.asarray(updated.diagnostics.early_termination_rate),
        np.ones(num_devices),
    )
