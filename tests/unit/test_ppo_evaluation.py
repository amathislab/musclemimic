"""Tests for PPO policy and trajectory-set evaluation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.common import trajectory_evaluation as common_trajectory_evaluation
from musclemimic.algorithms.common import trajectory_sampling
from musclemimic.algorithms.common import validation as common_validation
from musclemimic.algorithms.common.distributed import DATA_AXIS_NAME, make_data_mesh
from musclemimic.algorithms.ppo import evaluation
from musclemimic.runner.validation_reduction import (
    add_validation_shard_axis,
    finish_validation,
)
from musclemimic.utils.metrics import ValidationSummary


class _TrajectoryState(NamedTuple):
    traj_no: jax.Array
    subtraj_step_no: jax.Array


class _Policy:
    def __init__(self, batch_shape):
        self._batch_shape = batch_shape

    def mode(self):
        return jnp.zeros((*self._batch_shape, 1), dtype=jnp.float32)

    def sample(self, seed):
        del seed
        return jnp.ones((*self._batch_shape, 1), dtype=jnp.float32)


def _apply_policy(variables, observation):
    del variables
    policy = _Policy(observation.shape[:-1])
    value = jnp.zeros(observation.shape[:-1], dtype=jnp.float32)
    return policy, value


@struct.dataclass
class _EvaluationTrainState:
    params: dict
    run_stats: jax.Array
    apply_fn: Any = struct.field(pytree_node=False)


class _PolicyEvaluationEnv:
    def reset(self, reset_keys):
        num_envs = reset_keys.shape[0]
        return (
            jnp.zeros((num_envs, 1), dtype=jnp.float32),
            jnp.zeros(num_envs, dtype=jnp.float32),
        )

    def step_with_transition(self, env_state, action):
        num_envs = env_state.shape[0]
        physical_state = env_state + action[:, 0]
        data = SimpleNamespace(
            qpos=physical_state[:, None],
            qvel=jnp.zeros((num_envs, 1)),
            xpos=jnp.zeros((num_envs, 1, 3)),
            xquat=jnp.zeros((num_envs, 1, 4)),
            cvel=jnp.zeros((num_envs, 1, 6)),
            subtree_com=jnp.zeros((num_envs, 1, 3)),
            site_xpos=jnp.zeros((num_envs, 1, 3)),
            site_xmat=jnp.zeros((num_envs, 1, 3, 3)),
        )
        mjx_state = SimpleNamespace(
            data=data,
            additional_carry=SimpleNamespace(
                traj_state=_TrajectoryState(
                    traj_no=jnp.zeros(num_envs, dtype=jnp.int32),
                    subtraj_step_no=jnp.zeros(num_envs, dtype=jnp.int32),
                )
            ),
        )
        log_state = SimpleNamespace(
            env_state=mjx_state,
            metrics={"action": action[:, 0]},
        )
        transition_state = SimpleNamespace(
            find=lambda _state_type: log_state,
        )
        done = jnp.zeros(num_envs, dtype=bool)
        next_observation = jnp.zeros((num_envs, 1), dtype=jnp.float32)
        next_env_state = jnp.zeros_like(env_state)
        return (
            next_observation,
            jnp.zeros(num_envs),
            done,
            done,
            {},
            next_env_state,
            transition_state,
            next_observation,
        )


class _MetricsHandler:
    rel_site_ids = jnp.asarray([0], dtype=jnp.int32)

    def __call__(self, validation_data, *, sim_site_idx):
        return {
            "qpos_mean": jnp.mean(validation_data.data.qpos),
            "action_mean": jnp.mean(validation_data.metrics["action"]),
            "site_id_sum": jnp.sum(sim_site_idx),
        }

    @staticmethod
    def get_zero_container():
        return {
            "qpos_mean": jnp.asarray(0.0, dtype=jnp.float32),
            "action_mean": jnp.asarray(0.0, dtype=jnp.float32),
            "site_id_sum": jnp.asarray(0, dtype=jnp.int32),
        }


@pytest.mark.parametrize(
    ("deterministic", "expected_action"),
    [(True, 0.0), (False, 1.0)],
)
def test_evaluate_policy_compiles_and_uses_transition_state(
    monkeypatch,
    deterministic,
    expected_action,
):
    monkeypatch.setattr(
        common_validation,
        "unwrap_to_mjx",
        lambda state: (state, None),
    )
    weighted_reset_calls = []
    env = _PolicyEvaluationEnv()

    def weighted_reset(env_arg, reset_keys, weights):
        weighted_reset_calls.append((weights, reset_keys.shape[0]))
        return env_arg.reset(reset_keys)

    monkeypatch.setattr(
        trajectory_sampling,
        "reset_from_trajectory_weights",
        weighted_reset,
    )
    config = OmegaConf.create(
        {
            "debug": False,
            "validation": {
                "num_envs": 2,
                "num_steps": 3,
                "deterministic": deterministic,
            },
        }
    )
    train_state = _EvaluationTrainState(
        params={},
        run_stats=jnp.asarray(0, dtype=jnp.int32),
        apply_fn=_apply_policy,
    )
    initial_rng = jax.random.key(4)
    weights = jnp.asarray([1.0], dtype=jnp.float32)

    compiled_evaluation = jax.jit(
        lambda state, rng: evaluation.evaluate_policy(
            state,
            rng,
            env,
            config,
            _MetricsHandler(),
            reset_weights=weights,
        )
    )
    metrics, next_rng = compiled_evaluation(train_state, initial_rng)

    assert len(weighted_reset_calls) == 1
    assert weighted_reset_calls[0][1] == 2
    np.testing.assert_allclose(metrics["action_mean"], expected_action)
    np.testing.assert_allclose(metrics["qpos_mean"], expected_action)
    assert int(metrics["site_id_sum"]) == 0
    expected_rng = jax.random.split(initial_rng, 3)[0]
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(next_rng)),
        np.asarray(jax.random.key_data(expected_rng)),
    )


def test_validation_step_is_independently_compilable_and_preserves_skip_semantics(
    monkeypatch,
):
    monkeypatch.setattr(
        common_validation,
        "unwrap_to_mjx",
        lambda state: (state, None),
    )
    config = OmegaConf.create(
        {
            "debug": False,
            "validation": {
                "num_envs": 2,
                "num_steps": 3,
                "deterministic": False,
            },
        }
    )
    train_state = _EvaluationTrainState(
        params={},
        run_stats=jnp.asarray(0, dtype=jnp.int32),
        apply_fn=_apply_policy,
    )
    initial_rng = jax.random.key(7)
    env = _PolicyEvaluationEnv()
    metrics_handler = _MetricsHandler()
    compiled_validation = jax.jit(
        lambda state, rng, should_validate: evaluation.run_validation_step(
            state,
            rng,
            should_validate,
            env,
            config,
            metrics_handler,
        )
    )

    evaluated = compiled_validation(train_state, initial_rng, jnp.asarray(True))
    skipped = compiled_validation(train_state, initial_rng, jnp.asarray(False))

    np.testing.assert_allclose(evaluated.metrics["action_mean"], 1.0)
    np.testing.assert_allclose(evaluated.metrics["qpos_mean"], 1.0)
    expected_rng = jax.random.split(initial_rng, 3)[0]
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(evaluated.rng)),
        np.asarray(jax.random.key_data(expected_rng)),
    )
    assert all(float(value) == 0.0 for value in skipped.metrics.values())
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(skipped.rng)),
        np.asarray(jax.random.key_data(initial_rng)),
    )


def test_validation_step_without_metrics_handler_preserves_rng_schedule():
    initial_rng = jax.random.key(9)
    compiled_validation = jax.jit(
        lambda rng, should_validate: evaluation.run_validation_step(
            train_state=None,
            rng=rng,
            should_validate=should_validate,
            env=None,
            config=None,
            metrics_handler=None,
        )
    )

    evaluated = compiled_validation(initial_rng, jnp.asarray(True))
    skipped = compiled_validation(initial_rng, jnp.asarray(False))

    expected_rng = jax.random.split(initial_rng, 3)[0]
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(evaluated.rng)),
        np.asarray(jax.random.key_data(expected_rng)),
    )
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(skipped.rng)),
        np.asarray(jax.random.key_data(initial_rng)),
    )
    assert float(evaluated.metrics.episode_count) == 0.0
    assert float(skipped.metrics.episode_count) == 0.0


def test_validation_step_shard_map_defers_reduction_and_uses_local_reset_weights(
    monkeypatch,
):
    num_devices = jax.local_device_count()

    class SummaryMetricsHandler:
        @staticmethod
        def get_zero_container():
            return ValidationSummary()

    def evaluate_policy(
        train_state,
        rng,
        _env,
        _config,
        _handler,
        *,
        reset_weights,
        axis_name,
    ):
        assert axis_name == DATA_AXIS_NAME
        replica = train_state.astype(jnp.float32)
        metrics = ValidationSummary(
            episode_count=replica + 1.0,
            mean_episode_return=replica * 10.0,
            mean_episode_length=replica + 5.0,
            err_root_yaw=reset_weights[0],
            max_timestep=replica,
        )
        return metrics, jax.random.split(rng, 3)[0]

    monkeypatch.setattr(evaluation, "evaluate_policy", evaluate_policy)
    train_states = jnp.arange(num_devices, dtype=jnp.int32)
    rngs = jax.random.split(jax.random.key(11), num_devices)
    reset_weights = jnp.stack(
        (
            jnp.arange(num_devices, dtype=jnp.float32) + 0.25,
            jnp.arange(num_devices, dtype=jnp.float32) + 0.75,
        ),
        axis=1,
    )

    mesh = make_data_mesh(num_devices)
    data = jax.sharding.PartitionSpec(DATA_AXIS_NAME)

    def validation_shard(state, rng, weights):
        result = evaluation.run_validation_step(
            jnp.squeeze(state, axis=0),
            jnp.squeeze(rng, axis=0),
            True,
            env=None,
            config=None,
            metrics_handler=SummaryMetricsHandler(),
            reset_weights=jnp.squeeze(weights, axis=0),
            axis_name=DATA_AXIS_NAME,
        )
        return result._replace(
            metrics=add_validation_shard_axis(result.metrics),
            rng=jnp.expand_dims(result.rng, axis=0),
        )

    metric_specs = jax.tree.map(
        lambda _value: data,
        ValidationSummary(),
    )
    compiled_validation = jax.jit(
        jax.shard_map(
            validation_shard,
            mesh=mesh,
            in_specs=(data, data, data),
            out_specs=evaluation.ValidationStepResult(metric_specs, data),
        )
    )
    sharding = jax.sharding.NamedSharding(mesh, data)
    validation_inputs = (
        jax.device_put(train_states, sharding),
        jax.device_put(rngs, sharding),
        jax.device_put(reset_weights, sharding),
    )
    lowered_validation = compiled_validation.lower(*validation_inputs)
    assert "all_reduce" not in lowered_validation.as_text().lower()
    local_results = lowered_validation.compile()(*validation_inputs)
    results = finish_validation(
        local_results,
        None,
        evaluate_all=False,
        event_name="validation_00000001",
    )

    counts = np.arange(1, num_devices + 1, dtype=np.float32)
    returns = np.arange(num_devices, dtype=np.float32) * 10.0
    expected_return = np.sum(counts * returns) / np.sum(counts)
    np.testing.assert_allclose(
        np.asarray(results.metrics.episode_count),
        np.sum(counts),
    )
    np.testing.assert_allclose(
        np.asarray(results.metrics.mean_episode_return),
        expected_return,
    )
    np.testing.assert_allclose(
        np.asarray(results.metrics.err_root_yaw),
        np.mean(np.asarray(reset_weights[:, 0])),
    )
    np.testing.assert_array_equal(
        np.asarray(results.metrics.max_timestep),
        num_devices - 1,
    )
    expected_rngs = jax.vmap(lambda rng: jax.random.split(rng, 3)[0])(rngs)
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(results.rng)),
        np.asarray(jax.random.key_data(expected_rngs)),
    )


class _TrajectoryLibrary:
    n_trajectories = 3
    _lengths = (2, 4, 3)

    def len_trajectory(self, index):
        return self._lengths[index]


class _TrajectoryEvaluationEnv:
    _lengths = jnp.asarray([2, 4, 3], dtype=jnp.int32)

    def reset_to(self, reset_keys, trajectory_ids):
        del reset_keys
        state = {
            "step": jnp.zeros_like(trajectory_ids),
            "trajectory_id": trajectory_ids,
        }
        observation = jnp.zeros((trajectory_ids.shape[0], 1), dtype=jnp.float32)
        return observation, state

    def step(self, state, action):
        del action
        step = state["step"] + 1
        trajectory_id = state["trajectory_id"]
        early = (trajectory_id == 1) & (step >= 2)
        done = early | (step >= self._lengths[trajectory_id])
        next_state = {"step": step, "trajectory_id": trajectory_id}
        observation = step[:, None].astype(jnp.float32)
        reward = jnp.ones_like(step, dtype=jnp.float32)
        info = {"reward_total": trajectory_id.astype(jnp.float32)}
        return observation, reward, early, done, info, next_state


class _TrajectoryNetwork:
    @staticmethod
    def apply(variables, observation):
        del variables
        policy = _Policy(observation.shape[:-1])
        value = jnp.zeros(observation.shape[:-1], dtype=jnp.float32)
        return policy, value


def test_evaluate_trajectory_set_aggregates_partial_final_batch():
    trajectory_env = SimpleNamespace(th=_TrajectoryLibrary())

    metrics = evaluation.evaluate_trajectory_set(
        _TrajectoryNetwork(),
        params={},
        run_stats=jnp.asarray(0),
        trajectory_env=trajectory_env,
        env=_TrajectoryEvaluationEnv(),
        num_envs=2,
        deterministic=True,
        eval_seed=3,
    )

    assert metrics["val_mean_episode_length"] == pytest.approx(7.0 / 3.0)
    assert metrics["val_mean_episode_return"] == pytest.approx(7.0 / 3.0)
    assert metrics["val_early_termination_count"] == 1.0
    assert metrics["val_early_termination_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["val_frame_coverage"] == pytest.approx(7.0 / 9.0)
    assert metrics["val_total_frame"] == 9.0
    assert metrics["val_reward_total"] == pytest.approx(8.0 / 7.0)


def test_exact_set_evaluates_each_unmasked_trajectory_once():
    trajectory_ids = jnp.asarray([0, 1, 2, 0], dtype=jnp.int32)
    trajectory_lengths = jnp.asarray([2, 4, 3, 2], dtype=jnp.int32)
    trajectory_mask = jnp.asarray([True, True, True, False])

    result = common_trajectory_evaluation.evaluate_exact_trajectory_set(
        lambda observation, _rng: jnp.zeros_like(observation),
        jax.random.key(3),
        _TrajectoryEvaluationEnv(),
        trajectory_ids=trajectory_ids,
        trajectory_lengths=trajectory_lengths,
        trajectory_mask=trajectory_mask,
        num_steps=4,
        axis_name=None,
    )

    assert result.metrics.episode_count == pytest.approx(3.0)
    assert result.metrics.mean_episode_length == pytest.approx(7.0 / 3.0)
    assert result.metrics.early_termination_count == pytest.approx(1.0)
    assert result.metrics.early_termination_rate == pytest.approx(1.0 / 3.0)
    assert result.metrics.horizon_timeout_count == pytest.approx(0.0)
    assert result.metrics.frame_coverage == pytest.approx(7.0 / 9.0)
    assert result.metrics.total_frame == pytest.approx(9.0)
    assert result.metrics.reward_total == pytest.approx(8.0 / 7.0)


def test_exact_set_reports_horizon_censoring_separately():
    result = common_trajectory_evaluation.evaluate_exact_trajectory_set(
        lambda observation, _rng: jnp.zeros_like(observation),
        jax.random.key(3),
        _TrajectoryEvaluationEnv(),
        trajectory_ids=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        trajectory_lengths=jnp.asarray([2, 4, 3], dtype=jnp.int32),
        trajectory_mask=jnp.asarray([True, True, True]),
        num_steps=1,
        axis_name=None,
    )

    assert result.metrics.early_termination_count == pytest.approx(0.0)
    assert result.metrics.horizon_timeout_count == pytest.approx(3.0)
    assert result.metrics.horizon_timeout_rate == pytest.approx(1.0)
    assert result.metrics.frame_coverage == pytest.approx(3.0 / 9.0)


def test_exact_set_shard_map_accepts_constant_reset_state():
    """Exact rollout stays local before a separately compiled reduction."""

    class DeviceVaryingEnv(_TrajectoryEvaluationEnv):
        def step(self, state, action):
            del action
            replica = jax.lax.axis_index(DATA_AXIS_NAME).astype(jnp.int32)
            step = state["step"] + replica + 1
            trajectory_id = state["trajectory_id"]
            done = step >= self._lengths[trajectory_id]
            next_state = {"step": step, "trajectory_id": trajectory_id}
            observation = step[:, None].astype(jnp.float32)
            reward = jnp.ones_like(step, dtype=jnp.float32)
            info = {"reward_total": trajectory_id.astype(jnp.float32)}
            return observation, reward, done, done, info, next_state

    num_devices = jax.local_device_count()
    mesh = make_data_mesh(num_devices)
    data = jax.sharding.PartitionSpec(DATA_AXIS_NAME)
    rngs = jax.random.split(jax.random.key(17), num_devices)

    def evaluate_shard(rng):
        result = common_trajectory_evaluation.evaluate_exact_trajectory_set_local(
            lambda observation, _rng: jnp.zeros_like(observation),
            jnp.squeeze(rng, axis=0),
            DeviceVaryingEnv(),
            trajectory_ids=jnp.asarray([0, 1, 2], dtype=jnp.int32),
            trajectory_lengths=jnp.asarray([2, 4, 3], dtype=jnp.int32),
            trajectory_mask=jnp.asarray([True, True, True]),
            num_steps=4,
            axis_name=DATA_AXIS_NAME,
        )
        return result._replace(
            metrics=add_validation_shard_axis(result.metrics),
            rng=jnp.expand_dims(result.rng, axis=0),
        )

    scalar = jnp.asarray(0.0, dtype=jnp.float32)
    statistics_template = common_trajectory_evaluation.ExactTrajectorySetStatistics(
        episode_count=scalar,
        episode_return_sum=scalar,
        episode_length_sum=scalar,
        early_termination_count=scalar,
        horizon_timeout_count=scalar,
        covered_frame_sum=scalar,
        total_frame=scalar,
        step_count=scalar,
        step_sums={"reward_total": scalar},
    )
    output_specs = common_trajectory_evaluation.ExactTrajectorySetLocalResult(
        metrics=jax.tree.map(lambda _value: data, statistics_template),
        rng=data,
    )
    compiled_rollout = jax.jit(
        jax.shard_map(
            evaluate_shard,
            mesh=mesh,
            in_specs=(data,),
            out_specs=output_specs,
            check_vma=True,
        )
    )
    placed_rngs = jax.device_put(
        rngs,
        jax.sharding.NamedSharding(mesh, data),
    )
    lowered_rollout = compiled_rollout.lower(placed_rngs)
    assert "all_reduce" not in lowered_rollout.as_text().lower()
    local_result = lowered_rollout.compile()(placed_rngs)
    result = finish_validation(
        local_result,
        None,
        evaluate_all=True,
        event_name="validation_00000001",
    )

    assert result.metrics.episode_count == pytest.approx(3.0 * num_devices)


class _SingleTrajectoryLibrary:
    n_trajectories = 1

    @staticmethod
    def len_trajectory(_index):
        return 2


class _BatchSizePolicy:
    def __init__(self, batch_shape, value):
        self._batch_shape = batch_shape
        self._value = value

    def mode(self):
        return jnp.broadcast_to(
            self._value,
            (*self._batch_shape, 1),
        )

    def sample(self, seed):
        del seed
        return self.mode()


class _BatchSizeSensitiveNetwork:
    @staticmethod
    def apply(variables, observation):
        del variables
        policy = _BatchSizePolicy(
            observation.shape[:-1],
            observation.shape[0],
        )
        value = jnp.zeros(observation.shape[:-1], dtype=jnp.float32)
        return policy, value


class _ActionRewardTrajectoryEnv:
    def reset_to(self, reset_keys, trajectory_ids):
        del reset_keys
        state = {
            "step": jnp.zeros_like(trajectory_ids),
            "trajectory_id": trajectory_ids,
        }
        observation = jnp.zeros((trajectory_ids.shape[0], 1), dtype=jnp.float32)
        return observation, state

    def step(self, state, action):
        step = state["step"] + 1
        done = step >= 2
        next_state = {
            "step": step,
            "trajectory_id": state["trajectory_id"],
        }
        observation = step[:, None].astype(jnp.float32)
        reward = action[:, 0]
        absorbing = jnp.zeros_like(done)
        return observation, reward, absorbing, done, {}, next_state


def test_evaluate_trajectory_set_does_not_pad_partial_batch_into_network():
    trajectory_env = SimpleNamespace(th=_SingleTrajectoryLibrary())

    def evaluate(num_envs):
        return evaluation.evaluate_trajectory_set(
            _BatchSizeSensitiveNetwork(),
            params={},
            run_stats=jnp.asarray(0.0),
            trajectory_env=trajectory_env,
            env=_ActionRewardTrajectoryEnv(),
            num_envs=num_envs,
            deterministic=True,
        )

    single_env_metrics = evaluate(num_envs=1)
    oversized_batch_metrics = evaluate(num_envs=4)

    assert single_env_metrics["val_mean_episode_return"] == pytest.approx(2.0)
    assert oversized_batch_metrics["val_mean_episode_return"] == pytest.approx(2.0)
