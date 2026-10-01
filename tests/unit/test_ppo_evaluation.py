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

from musclemimic.algorithms.ppo import evaluation
from musclemimic.utils.metrics import QuantityContainer, ValidationSummary


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
        evaluation,
        "unwrap_to_mjx",
        lambda state: (state, None),
    )
    weighted_reset_calls = []
    env = _PolicyEvaluationEnv()

    def weighted_reset(env_arg, reset_keys, weights):
        weighted_reset_calls.append((weights, reset_keys.shape[0]))
        return env_arg.reset(reset_keys)

    monkeypatch.setattr(
        evaluation,
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
        evaluation,
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


class _ExhaustiveState(NamedTuple):
    step: jax.Array
    trajectory_id: jax.Array


class _ExhaustiveTrajectoryLibrary:
    n_trajectories = 3


class _ExhaustiveMetricsHandler:
    _trajectory_handler = _ExhaustiveTrajectoryLibrary()
    _trajectory_lengths = (2, 4, 3)
    _validation_horizon = 4

    @staticmethod
    def get_zero_container():
        return ValidationSummary(
            euclidean_distance=QuantityContainer(qpos=jnp.asarray(0.0)),
        )


class _ExhaustiveEvaluationEnv:
    _lengths = jnp.asarray([2, 4, 3], dtype=jnp.int32)

    @staticmethod
    def reset_to(_reset_keys, trajectory_ids):
        num_envs = trajectory_ids.shape[0]
        return (
            jnp.zeros((num_envs, 1), dtype=jnp.float32),
            _ExhaustiveState(
                step=jnp.zeros((num_envs,), dtype=jnp.int32),
                trajectory_id=trajectory_ids,
            ),
        )

    @classmethod
    def step_with_transition(cls, state, _action):
        next_step = state.step + 1
        target_length = cls._lengths[state.trajectory_id]
        forced_failure = (state.trajectory_id == 1) & (next_step == 2)
        done = forced_failure | (next_step >= target_length)
        autoreset_step = jnp.where(done, 0, next_step)
        next_state = _ExhaustiveState(autoreset_step, state.trajectory_id)
        observation = jnp.zeros((state.step.shape[0], 1), dtype=jnp.float32)
        reward = jnp.ones_like(state.step, dtype=jnp.float32)
        info = {
            "reward_total": reward,
            "err_joint_pos": next_step.astype(jnp.float32),
        }
        return (
            observation,
            reward,
            forced_failure,
            done,
            info,
            next_state,
            next_state,
            observation,
        )


def test_exhaustive_validation_shards_motions_across_replicas_and_masks_autoresets():
    config = OmegaConf.create(
        {
            "validation": {
                "evaluate_all": True,
                "num_envs": 2,
                "global_num_envs": 4,
                "deterministic": True,
            },
        }
    )
    train_state = _EvaluationTrainState(
        params={},
        run_stats=jnp.asarray(0, dtype=jnp.int32),
        apply_fn=_apply_policy,
    )
    keys = jax.random.split(jax.random.key(13), 2)

    evaluate_replicas = jax.jit(
        jax.vmap(
            lambda key: evaluation.evaluate_policy_exhaustive(
                train_state,
                key,
                _ExhaustiveEvaluationEnv(),
                config,
                _ExhaustiveMetricsHandler(),
                axis_name="devices",
            ),
            axis_name="devices",
        )
    )
    metrics, next_rngs = evaluate_replicas(keys)

    expected_success = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    expected_coverage = np.asarray([1.0, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(metrics.per_motion_success),
        np.broadcast_to(expected_success, (2, 3)),
    )
    np.testing.assert_allclose(
        np.asarray(metrics.per_motion_frame_coverage),
        np.broadcast_to(expected_coverage, (2, 3)),
    )
    np.testing.assert_allclose(np.asarray(metrics.episode_count), [3.0, 3.0])
    np.testing.assert_allclose(np.asarray(metrics.motion_count), [3.0, 3.0])
    np.testing.assert_allclose(np.asarray(metrics.motion_success_rate), [2.0 / 3.0] * 2)
    np.testing.assert_allclose(np.asarray(metrics.early_termination_rate), [1.0 / 3.0] * 2)
    np.testing.assert_allclose(np.asarray(metrics.mean_frame_coverage), [5.0 / 6.0] * 2)
    np.testing.assert_allclose(np.asarray(metrics.worst_quartile_frame_coverage), [0.5, 0.5])
    np.testing.assert_allclose(np.asarray(metrics.mean_episode_length), [7.0 / 3.0] * 2)
    np.testing.assert_allclose(np.asarray(metrics.mean_return_per_frame), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(metrics.horizon_timeout_count), [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(metrics.reward_total), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(metrics.err_joint_pos), [12.0 / 7.0] * 2)
    expected_next_rngs = jax.vmap(lambda key: jax.random.split(key, 3)[0])(keys)
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(next_rngs)),
        np.asarray(jax.random.key_data(expected_next_rngs)),
    )


def test_validation_step_dispatches_evaluate_all_and_preserves_shaped_skip(monkeypatch):
    handler = _ExhaustiveMetricsHandler()
    config = OmegaConf.create(
        {
            "validation": {
                "evaluate_all": True,
            },
        }
    )
    train_state = _EvaluationTrainState(
        params={},
        run_stats=jnp.asarray(0, dtype=jnp.int32),
        apply_fn=_apply_policy,
    )

    def fail_pooled(*_args, **_kwargs):
        raise AssertionError("evaluate_all must not call pooled PPO validation")

    def exhaustive(_state, key, _env, _config, _handler, *, axis_name):
        assert axis_name is None
        values = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)
        return (
            ValidationSummary(
                episode_count=jnp.asarray(3.0),
                motion_count=jnp.asarray(3.0),
                per_motion_success=values,
                per_motion_frame_coverage=values,
                per_motion_return_per_frame=values,
            ),
            jax.random.split(key, 3)[0],
        )

    monkeypatch.setattr(evaluation, "evaluate_policy", fail_pooled)
    monkeypatch.setattr(evaluation, "evaluate_policy_exhaustive", exhaustive)
    initial_rng = jax.random.key(17)
    compiled = jax.jit(
        lambda should_validate: evaluation.run_validation_step(
            train_state,
            initial_rng,
            should_validate,
            env=None,
            config=config,
            metrics_handler=handler,
        )
    )

    evaluated = compiled(jnp.asarray(True))
    skipped = compiled(jnp.asarray(False))

    np.testing.assert_allclose(np.asarray(evaluated.metrics.per_motion_success), [1.0, 0.0, 1.0])
    np.testing.assert_allclose(np.asarray(skipped.metrics.per_motion_success), [0.0, 0.0, 0.0])
    assert evaluated.metrics.per_motion_success.shape == skipped.metrics.per_motion_success.shape
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(skipped.rng)),
        np.asarray(jax.random.key_data(initial_rng)),
    )


def test_validation_step_pmap_reduces_metrics_and_uses_local_reset_weights(
    monkeypatch,
):
    num_devices = jax.local_device_count()

    class SummaryMetricsHandler:
        @staticmethod
        def get_zero_container():
            return ValidationSummary()

    def evaluate_policy(train_state, rng, _env, _config, _handler, *, reset_weights):
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

    compiled_validation = jax.jit(
        lambda state, rng, weights: evaluation.run_validation_step(
            state,
            rng,
            jnp.asarray(True),
            env=None,
            config=None,
            metrics_handler=SummaryMetricsHandler(),
            reset_weights=weights,
            axis_name="devices",
        )
    )
    results = jax.pmap(
        compiled_validation,
        axis_name="devices",
    )(train_states, rngs, reset_weights)

    counts = np.arange(1, num_devices + 1, dtype=np.float32)
    returns = np.arange(num_devices, dtype=np.float32) * 10.0
    expected_return = np.sum(counts * returns) / np.sum(counts)
    np.testing.assert_allclose(
        np.asarray(results.metrics.episode_count),
        np.full(num_devices, np.sum(counts)),
    )
    np.testing.assert_allclose(
        np.asarray(results.metrics.mean_episode_return),
        np.full(num_devices, expected_return),
    )
    np.testing.assert_allclose(
        np.asarray(results.metrics.err_root_yaw),
        np.full(num_devices, np.mean(np.asarray(reset_weights[:, 0]))),
    )
    np.testing.assert_array_equal(
        np.asarray(results.metrics.max_timestep),
        np.full(num_devices, num_devices - 1),
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
