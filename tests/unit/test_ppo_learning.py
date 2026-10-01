"""Characterization tests for PPO learning and rollout modules."""

from __future__ import annotations

import os
from collections import namedtuple
from types import SimpleNamespace
from typing import NamedTuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.common.distributed import replicate_for_pmap
from musclemimic.algorithms.common.networks import ActorCritic
from musclemimic.algorithms.ppo.learning import (
    PPOUpdateMetrics,
    initialize_train_state,
    update,
    update_observation_stats,
)
from musclemimic.algorithms.ppo.rollout import collect_rollout
from musclemimic.utils.debug_tools import DebugFlags


class _LearningRollout(NamedTuple):
    observation: jax.Array
    action: jax.Array
    value: jax.Array
    log_prob: jax.Array

    @property
    def obs(self):
        return self.observation


def _learning_config():
    return OmegaConf.create(
        {
            "num_minibatches": 2,
            "update_epochs": 2,
            "clip_eps": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "use_moe": False,
            "schedule": "fixed",
            "desired_kl": None,
            "reset_lr_schedule_on_resume": False,
            "reset_std_on_resume": None,
        }
    )


def test_update_runs_real_ppo_minibatches_under_jit():
    network = ActorCritic(
        action_dim=2,
        hidden_layer_dims=(8,),
        actor_obs_ind=jnp.arange(3),
        critic_obs_ind=jnp.arange(3),
    )
    optimizer = optax.adam(1e-3)
    env = SimpleNamespace(
        info=SimpleNamespace(
            observation_space=SimpleNamespace(shape=(3,)),
        )
    )
    train_state = initialize_train_state(
        jax.random.key(0),
        env,
        network,
        optimizer,
        None,
        _learning_config(),
    )

    observations = jnp.linspace(-1.0, 1.0, 24).reshape(8, 3)
    policy, values = network.apply(
        {
            "params": train_state.params,
            "run_stats": train_state.run_stats,
        },
        observations,
    )
    actions = jnp.zeros((8, 2), dtype=jnp.float32)
    rollout = _LearningRollout(
        observation=observations.reshape(2, 4, 3),
        action=actions.reshape(2, 4, 2),
        value=values.reshape(2, 4),
        log_prob=policy.log_prob(actions).reshape(2, 4),
    )
    advantages = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    targets = rollout.value + 0.25
    initial_params = train_state.params
    initial_rng = jax.random.key(7)
    learning_rate = jnp.asarray(1e-3, dtype=jnp.float32)
    config = _learning_config()

    compiled_update = jax.jit(
        lambda state, rng: update(
            state,
            rollout,
            advantages,
            targets,
            rng,
            learning_rate,
            network=network,
            config=config,
            base_learning_rate=1e-3,
            use_adaptive_learning_rate=False,
        )
    )
    updated_state, updated_rng, updated_lr, metrics = compiled_update(
        train_state,
        initial_rng,
    )

    assert isinstance(metrics, PPOUpdateMetrics)
    assert int(updated_state.step) == 4
    assert not np.array_equal(
        np.asarray(jax.random.key_data(updated_rng)),
        np.asarray(jax.random.key_data(initial_rng)),
    )
    np.testing.assert_allclose(np.asarray(updated_lr), 1e-3)
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(metrics))
    assert any(
        not np.allclose(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree.leaves(initial_params),
            jax.tree.leaves(updated_state.params),
            strict=True,
        )
    )
    for before, after in zip(
        jax.tree.leaves(train_state.run_stats),
        jax.tree.leaves(updated_state.run_stats),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))


def test_update_observation_stats_merges_boundary_batch_once():
    network = ActorCritic(action_dim=2, hidden_layer_dims=(8,))
    env = SimpleNamespace(
        info=SimpleNamespace(
            observation_space=SimpleNamespace(shape=(3,)),
        )
    )
    train_state = initialize_train_state(
        jax.random.key(0),
        env,
        network,
        optax.adam(1e-3),
        None,
        _learning_config(),
    )
    observations = jnp.arange(24, dtype=jnp.float32).reshape(2, 4, 3)

    updated_state = jax.jit(
        lambda state: update_observation_stats(
            state,
            observations,
            network=network,
        )
    )(train_state)

    stats = updated_state.run_stats["RunningMeanStd_0"]
    initial_count = train_state.run_stats["RunningMeanStd_0"]["count"]
    np.testing.assert_allclose(np.asarray(stats["count"]), np.asarray(initial_count + 8))
    np.testing.assert_allclose(np.asarray(stats["mean"]), np.asarray(jnp.mean(observations, axis=(0, 1))), atol=1e-5)


@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="requires at least two JAX devices",
)
def test_update_keeps_parameters_and_metrics_replica_consistent_under_pmap():
    num_devices = 2
    devices = jax.local_devices()[:num_devices]
    config = _learning_config()
    network = ActorCritic(
        action_dim=2,
        hidden_layer_dims=(8,),
        actor_obs_ind=jnp.arange(3),
        critic_obs_ind=jnp.arange(3),
    )
    optimizer = optax.adam(1e-3)
    env = SimpleNamespace(
        info=SimpleNamespace(
            observation_space=SimpleNamespace(shape=(3,)),
        )
    )
    train_state = initialize_train_state(
        jax.random.key(0),
        env,
        network,
        optimizer,
        None,
        config,
    )

    observations = jnp.linspace(-1.0, 1.0, num_devices * 12).reshape(num_devices, 2, 2, 3)
    policy, values = network.apply(
        {
            "params": train_state.params,
            "run_stats": train_state.run_stats,
        },
        observations,
    )
    actions = jnp.zeros((num_devices, 2, 2, 2), dtype=jnp.float32)
    rollouts = _LearningRollout(
        observation=observations,
        action=actions,
        value=values,
        log_prob=policy.log_prob(actions),
    )
    advantages = jnp.arange(num_devices * 4, dtype=jnp.float32).reshape(num_devices, 2, 2)
    targets = values + 0.25
    rngs = jax.random.split(jax.random.key(7), num_devices)
    replicated_state = replicate_for_pmap(train_state, devices)

    pmapped_update = jax.pmap(
        lambda state, rollout, gae, target, rng: update(
            state,
            rollout,
            gae,
            target,
            rng,
            jnp.asarray(1e-3, dtype=jnp.float32),
            network=network,
            config=config,
            base_learning_rate=1e-3,
            use_adaptive_learning_rate=False,
            axis_name="devices",
        ),
        axis_name="devices",
        devices=devices,
    )
    updated_state, _rngs, _learning_rates, metrics = pmapped_update(
        replicated_state,
        rollouts,
        advantages,
        targets,
        rngs,
    )
    jax.block_until_ready((updated_state, metrics))

    expected_step = config.num_minibatches * config.update_epochs
    np.testing.assert_array_equal(
        np.asarray(updated_state.step),
        np.full(num_devices, expected_step),
    )
    for leaf in jax.tree.leaves(updated_state.params):
        for replica in range(1, num_devices):
            np.testing.assert_allclose(
                np.asarray(leaf[replica]),
                np.asarray(leaf[0]),
                rtol=1e-6,
                atol=1e-6,
            )
    for leaf in jax.tree.leaves(metrics):
        for replica in range(1, num_devices):
            np.testing.assert_allclose(
                np.asarray(leaf[replica]),
                np.asarray(leaf[0]),
                rtol=1e-6,
                atol=1e-6,
            )


@struct.dataclass
class _RolloutTrainState:
    params: dict
    run_stats: jax.Array


class _Policy:
    def __init__(self, batch_shape):
        self._batch_shape = batch_shape

    def sample(self, seed):
        del seed
        return jnp.ones((*self._batch_shape, 1), dtype=jnp.float32)

    def log_prob(self, action):
        return jnp.zeros(action.shape[:-1], dtype=jnp.float32)


class _RolloutNetwork:
    @staticmethod
    def apply(variables, observation):
        del variables
        policy = _Policy(observation.shape[:-1])
        value = observation[..., 0]
        return policy, value


_TrajectoryState = namedtuple(
    "_TrajectoryState",
    ("traj_no", "subtraj_step_no"),
)


class _TransitionState:
    def __init__(self, env_state, trajectory_state):
        self.additional_carry = SimpleNamespace(traj_state=trajectory_state)
        self._log_state = SimpleNamespace(
            env_state=env_state,
            metrics={"timestep": env_state},
        )

    def find(self, _state_type):
        return self._log_state


class _RolloutEnv:
    def __init__(self):
        trajectory_data = SimpleNamespace(split_points=jnp.asarray([0, 5], dtype=jnp.int32))
        self.th = SimpleNamespace(traj=SimpleNamespace(data=trajectory_data))

    def step_with_transition(self, env_state, action):
        next_env_state = env_state + 1
        observation = jnp.stack(
            (next_env_state, next_env_state + 10),
            axis=-1,
        )
        done = jnp.zeros_like(env_state, dtype=bool)
        trajectory_state = _TrajectoryState(
            traj_no=jnp.zeros_like(env_state, dtype=jnp.int32),
            subtraj_step_no=next_env_state.astype(jnp.int32),
        )
        transition_state = _TransitionState(
            next_env_state,
            trajectory_state,
        )
        return (
            observation,
            jnp.sum(action, axis=-1),
            done,
            done,
            {"step": next_env_state},
            next_env_state,
            transition_state,
            observation,
        )


def test_collect_rollout_keeps_only_live_state_in_scan_carry():
    num_envs = 2
    initial_rng = jax.random.key(11)
    initial_state = _RolloutTrainState(
        params={},
        run_stats=jnp.asarray(0, dtype=jnp.int32),
    )

    final_state, rollout = collect_rollout(
        initial_state,
        jnp.zeros(num_envs, dtype=jnp.float32),
        jnp.zeros((num_envs, 2), dtype=jnp.float32),
        initial_rng,
        network=_RolloutNetwork(),
        env=_RolloutEnv(),
        num_steps=3,
        debug_flags=DebugFlags(),
    )
    train_state, env_state, observation, final_rng = final_state

    assert int(train_state.run_stats) == 0
    np.testing.assert_array_equal(np.asarray(env_state), [3.0, 3.0])
    np.testing.assert_array_equal(
        np.asarray(observation),
        [[3.0, 13.0], [3.0, 13.0]],
    )
    assert rollout.reward.shape == (3, num_envs)
    assert rollout.action.shape == (3, num_envs, 1)
    assert not np.asarray(rollout.traj_end).any()

    expected_rng = initial_rng
    for _ in range(3):
        expected_rng, _ = jax.random.split(expected_rng)
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(final_rng)),
        np.asarray(jax.random.key_data(expected_rng)),
    )
