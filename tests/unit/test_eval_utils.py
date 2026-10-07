"""Tests for standalone evaluation orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.flashsac.config import FlashSACAgentConf
from musclemimic.algorithms.flashsac.state import FlashSACPolicyState
from musclemimic.algorithms.ppo.config import PPOAgentConf
from musclemimic.runner import eval_utils
from musclemimic.utils.metrics import ValidationSummary


def test_validation_metrics_converts_static_trajectory_before_wrapping(monkeypatch):
    events = []

    class FakeTrajectoryHandler:
        is_numpy = True

        def to_jax(self):
            events.append("to_jax")
            self.is_numpy = False

    validation_env = SimpleNamespace(th=FakeTrajectoryHandler())
    expected_metrics_handler = object()
    validation_summary = object()

    monkeypatch.setattr(eval_utils, "instantiate_validation_env", lambda _config: validation_env)

    def wrap_env(env, _config):
        assert env.th.is_numpy is False
        events.append("wrap")
        return env

    def build_metrics_handler(_config, env):
        assert env.th.is_numpy is False
        events.append("metrics_handler")
        return expected_metrics_handler

    def evaluate_policy(_train_state, rng, env, config, metrics_handler):
        assert env.th.is_numpy is False
        assert metrics_handler is expected_metrics_handler
        events.append("evaluate")
        return validation_summary, rng

    def flatten_metrics(*_args, include_trajectory_set_metrics):
        assert include_trajectory_set_metrics is False
        return {"metric": 1.0}

    monkeypatch.setattr(eval_utils, "wrap_env", wrap_env)
    monkeypatch.setattr(eval_utils, "build_metrics_handler", build_metrics_handler)
    monkeypatch.setattr(eval_utils, "evaluate_policy", evaluate_policy)
    monkeypatch.setattr(eval_utils, "flatten_validation_metrics", flatten_metrics)

    config = OmegaConf.create(
        {
            "experiment": {
                "trajectory": {"sharding": "static"},
                "n_seeds": 1,
                "validation": {
                    "active": True,
                    "num_steps": 4,
                    "num_envs": 2,
                    "quantities": [],
                    "measures": [],
                },
            }
        }
    )
    agent_state = SimpleNamespace(train_state=object())
    agent_conf = PPOAgentConf(config=config, network=None, tx=None)

    summary, metrics = eval_utils.run_validation_metrics(
        config,
        agent_state,
        agent_conf,
        train_state_seed=None,
    )

    assert summary is validation_summary
    assert metrics == {"metric": 1.0}
    assert events == ["to_jax", "wrap", "metrics_handler", "evaluate"]


class _TrajectoryHandler:
    is_numpy = False
    n_trajectories = 3

    def len_trajectory(self, indices):
        return jnp.asarray([2, 4, 6], dtype=jnp.int32)[indices]


class _ExactEnvironment:
    th = _TrajectoryHandler()

    def reset_to(self, keys, ids):
        assert keys.shape[0] == ids.shape[0]
        return jnp.zeros((ids.size, 1)), {"step": jnp.zeros_like(ids), "id": ids}

    def step(self, state, action):
        # The middle trajectory terminates early. The last can exceed the evaluation horizon.
        lengths = jnp.asarray([2, 3, 6])[state["id"]]
        step = state["step"] + 1
        done = step >= lengths
        reward = state["id"].astype(jnp.float32) + 1.0 + action[:, 0]
        error = jnp.where(state["step"] >= lengths, jnp.nan, state["id"] + 1.0)
        return (
            step[:, None].astype(jnp.float32),
            reward,
            done,
            done,
            {"err_root_quat": error},
            {
                "step": step,
                "id": state["id"],
            },
        )


class _Policy:
    def __init__(self, observation, bias):
        self.mean = jnp.zeros_like(observation) + bias

    def mode(self):
        return self.mean

    def sample(self, seed):
        return self.mean + jax.random.normal(seed, self.mean.shape)


def _apply_policy(variables, observation):
    return _Policy(observation, variables["params"]["bias"]), jnp.zeros(observation.shape[0])


@struct.dataclass
class _PPOTrainState:
    params: Any
    run_stats: Any
    apply_fn: Any = struct.field(pytree_node=False, default=_apply_policy)


class _Actor:
    @staticmethod
    def apply(variables, observation, *, training):
        assert training is False
        return jnp.zeros_like(observation) + variables["params"]["bias"], jnp.ones_like(observation)


def _exact_evaluation_inputs(monkeypatch, algorithm):
    config = OmegaConf.create(
        {
            "experiment": {
                "n_seeds": 1,
                "validation": {
                    "active": True,
                    "evaluate_all": True,
                    "num_steps": 4,
                    "num_envs": 2,
                    "deterministic": True,
                    "quantities": [],
                    "measures": [],
                },
            }
        }
    )
    env = _ExactEnvironment()
    monkeypatch.setattr(eval_utils, "instantiate_validation_env", lambda _: env)
    monkeypatch.setattr(eval_utils, "wrap_env", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(eval_utils, "build_metrics_handler", lambda *_: SimpleNamespace())
    params = {"bias": jnp.asarray(0.0)}
    if algorithm == "ppo":
        agent_conf = PPOAgentConf(config=config, network=None, tx=None)
        state = _PPOTrainState(params=params, run_stats={})
    else:
        agent_conf = FlashSACAgentConf(
            config=config,
            actor=_Actor(),
            critic=None,
            actor_optimizer=None,
            critic_optimizer=None,
            temperature_optimizer=None,
            learning_rate_schedule=None,
            actor_observation_indices=jnp.asarray([0]),
            critic_observation_indices=jnp.asarray([0]),
        )
        state = FlashSACPolicyState(actor_params=params, actor_batch_stats={})
    return config, SimpleNamespace(train_state=state), agent_conf


@pytest.mark.parametrize("algorithm", ["ppo", "flashsac"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_exact_validation_metrics_finalize_all_trajectories(monkeypatch, algorithm, compiled, batch_size):
    config, state, agent_conf = _exact_evaluation_inputs(monkeypatch, algorithm)
    original_config = OmegaConf.to_container(config)
    with jax.disable_jit(not compiled):
        summary, metrics = eval_utils.run_validation_metrics(
            config,
            state,
            agent_conf,
            None,
            num_envs=batch_size,
        )

    assert isinstance(summary, ValidationSummary)
    assert metrics["val_episode_count"] == 3.0
    assert metrics["val_mean_episode_return"] == pytest.approx(20.0 / 3.0)
    assert metrics["val_mean_episode_length"] == 3.0
    assert metrics["val_early_termination_count"] == 1.0
    assert metrics["val_early_termination_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["val_horizon_timeout_count"] == 1.0
    assert metrics["val_horizon_timeout_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["val_frame_coverage"] == 0.75
    assert metrics["val_total_frame"] == 12.0
    assert metrics["val_err_root_quat"] == pytest.approx(20.0 / 9.0)
    assert OmegaConf.to_container(config) == original_config


@pytest.mark.parametrize("algorithm", ["ppo", "flashsac"])
def test_exact_validation_metrics_honor_step_override(monkeypatch, algorithm):
    config, state, agent_conf = _exact_evaluation_inputs(monkeypatch, algorithm)
    _, metrics = eval_utils.run_validation_metrics(config, state, agent_conf, None, num_steps=2)
    assert metrics["val_episode_count"] == 3.0
    assert metrics["val_mean_episode_length"] == 2.0
    assert metrics["val_horizon_timeout_count"] == 2.0
    assert metrics["val_frame_coverage"] == 0.5
    assert config.experiment.validation.num_steps == 4


@pytest.mark.parametrize("algorithm", ["ppo", "flashsac"])
def test_exact_validation_metrics_keep_rng_reproducible(monkeypatch, algorithm):
    config, state, agent_conf = _exact_evaluation_inputs(monkeypatch, algorithm)

    def evaluate(seed, deterministic=False):
        return eval_utils.run_validation_metrics(
            config,
            state,
            agent_conf,
            None,
            eval_seed=seed,
            deterministic_override=deterministic,
        )[1]

    first = evaluate(7)
    assert evaluate(7) == first
    assert evaluate(8)["val_mean_episode_return"] != first["val_mean_episode_return"]
    assert evaluate(7, deterministic=True)["val_mean_episode_return"] == pytest.approx(20.0 / 3.0)
    assert config.experiment.validation.deterministic is True


@pytest.mark.parametrize("algorithm", ["ppo", "flashsac"])
def test_exact_validation_metrics_select_checkpoint_seed(monkeypatch, algorithm):
    config, state, agent_conf = _exact_evaluation_inputs(monkeypatch, algorithm)
    config.experiment.n_seeds = 2
    state.train_state = jax.tree.map(lambda value: jnp.stack([value + 1.0, value]), state.train_state)

    with pytest.raises(ValueError, match="specify --train_state_seed"):
        eval_utils.run_validation_metrics(config, state, agent_conf, None)
    _, metrics = eval_utils.run_validation_metrics(config, state, agent_conf, 1)
    assert metrics["val_mean_episode_return"] == pytest.approx(20.0 / 3.0)


@pytest.mark.parametrize("override", [{"num_envs": 0}, {"num_envs": -1}, {"num_steps": 0}])
def test_exact_validation_metrics_reject_invalid_batch_dimensions(monkeypatch, override):
    config, state, agent_conf = _exact_evaluation_inputs(monkeypatch, "ppo")
    with pytest.raises(ValueError, match="positive num_envs and num_steps"):
        eval_utils.run_validation_metrics(config, state, agent_conf, None, **override)


def test_exact_validation_metrics_reject_empty_library(monkeypatch):
    config, state, agent_conf = _exact_evaluation_inputs(monkeypatch, "ppo")
    monkeypatch.setattr(_ExactEnvironment.th, "n_trajectories", 0)
    with pytest.raises(ValueError, match="at least one trajectory"):
        eval_utils.run_validation_metrics(config, state, agent_conf, None)
