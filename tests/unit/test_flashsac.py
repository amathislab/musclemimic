"""Focused tests for FlashSAC numerical and storage invariants."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from musclemimic.algorithms.common.checkpoint_manager import (
    CheckpointMetadata,
    UnifiedCheckpointManager,
)
from musclemimic.algorithms.flashsac import train as flashsac_train
from musclemimic.algorithms.flashsac.checkpoint import (
    FLASHSAC_ALGO_VERSION,
    load_checkpoint_for_resume,
    restore_train_state,
)
from musclemimic.algorithms.flashsac.config import FlashSACAgentConf, get_flashsac_config
from musclemimic.algorithms.flashsac.losses import (
    categorical_td_target,
    sample_tanh_normal,
    select_min_q_distribution,
)
from musclemimic.algorithms.flashsac.networks import (
    FlashSACActor,
    FlashSACDoubleCritic,
    UnitBatchNorm,
    normalize_parameters,
)
from musclemimic.algorithms.flashsac.replay_buffer import create_replay_buffer
from musclemimic.algorithms.flashsac.reward_normalization import update_discounted_return
from musclemimic.algorithms.flashsac.state import (
    FlashSACAgentState,
    FlashSACLearnerState,
    FlashSACRuntimeState,
    FlashSACTrainState,
)
from musclemimic.algorithms.flashsac.train import (
    FlashSACTraining,
    build_training,
    initialize_agent_state,
    mean_reward_components,
    prepare_replay_transition,
    reset_zeta_noise_on_done,
    select_collection_actions,
)
from musclemimic.algorithms.flashsac.update import update_from_batch
from musclemimic.algorithms.inference import make_policy_action_fn
from musclemimic.core.wrappers import AutoResetWrapper, LogWrapper
from musclemimic.runner.engine import pick_algorithm
from musclemimic.runner.flashsac_sharding import train_state_specs

LEGACY_FLASHSAC_ALGO_VERSION = "FlashSACJax_v1"


def _small_agent_conf(buffer_capacity: int = 64) -> FlashSACAgentConf:
    experiment_values = {
        "initial_temperature": 0.01,
        "buffer_capacity": buffer_capacity,
        "replay_storage_dtype": "bfloat16",
        "actor_update_period": 2,
        "target_sigma": 0.15,
        "target_update_tau": 0.01,
        "gamma": 0.99,
        "critic_min_value": -5.0,
        "critic_max_value": 5.0,
        "normalize_reward": True,
        "normalized_return_max": 5.0,
        "num_envs": 4,
    }
    experiment = OmegaConf.create(experiment_values)
    schedule = optax.constant_schedule(3e-4)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    return FlashSACAgentConf(
        config=OmegaConf.create({"experiment": experiment}),
        actor=FlashSACActor(action_dim=3, hidden_dim=8, num_blocks=1),
        critic=FlashSACDoubleCritic(hidden_dim=8, num_blocks=1, num_bins=5),
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
        temperature_optimizer=optax.adam(3e-4),
        learning_rate_schedule=schedule,
        actor_observation_indices=jnp.arange(5),
        critic_observation_indices=jnp.arange(5),
    )


def test_collection_actions_are_uniform_until_replay_warmup_finishes():
    key = jax.random.key(7)
    policy_actions = jnp.full((8, 3), 0.25, dtype=jnp.float32)
    expected_uniform = jax.random.uniform(
        key,
        policy_actions.shape,
        minval=-1.0,
        maxval=1.0,
    )
    select_actions = jax.jit(
        lambda replay_size: select_collection_actions(
            replay_size,
            min_buffer_size=16,
            policy_actions=policy_actions,
            uniform_rng=key,
        )
    )

    np.testing.assert_allclose(select_actions(jnp.array(15)), expected_uniform)
    np.testing.assert_allclose(select_actions(jnp.array(16)), policy_actions)


def test_discounted_return_includes_terminal_reward_and_resets_carry_in_eager_and_jit():
    previous_return = jnp.array([1.0, 10.0], dtype=jnp.float32)
    rewards = jnp.array([2.0, 3.0], dtype=jnp.float32)
    done = jnp.array([False, True])
    expected_samples = np.array([2.5, 8.0], dtype=np.float32)
    expected_carry = np.array([2.5, 0.0], dtype=np.float32)

    def update(previous, current_rewards, current_done):
        return update_discounted_return(
            previous,
            current_rewards,
            current_done,
            gamma=0.5,
        )

    for samples, carry in (update(previous_return, rewards, done), jax.jit(update)(previous_return, rewards, done)):
        np.testing.assert_allclose(samples, expected_samples, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(carry, expected_carry, atol=0.0, rtol=0.0)


def test_replay_transition_distinguishes_terminal_motion_end_from_horizon_and_sanitizes_numerics():
    observations = jnp.arange(12, dtype=jnp.float32).reshape(6, 2)
    actions = jnp.zeros((6, 2), dtype=jnp.float32).at[3, 1].set(jnp.inf)
    next_observations = observations + 0.5
    next_observations = next_observations.at[4, 0].set(jnp.nan)
    rewards = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, jnp.inf], dtype=jnp.float32)
    absorbing = jnp.array([True, False, False, False, False, False])
    info = {
        "done_numerical": jnp.zeros(6, dtype=bool),
        "done_trajectory": jnp.array([False, True, False, False, False, False]),
        "done_horizon": jnp.array([False, False, True, False, False, False]),
    }
    expected_terminated = np.array([True, True, False, True, True, True])

    def evaluate():
        return prepare_replay_transition(
            observations,
            actions,
            rewards,
            next_observations,
            absorbing,
            info,
        )

    for transition in (evaluate(), jax.jit(evaluate)()):
        np.testing.assert_array_equal(transition.terminated, expected_terminated)
        np.testing.assert_array_equal(
            transition.trajectory_end,
            [False, True, False, False, False, False],
        )
        np.testing.assert_array_equal(
            transition.numerical,
            [False, False, False, True, True, True],
        )
        assert np.isfinite(np.asarray(transition.observations)).all()
        assert np.isfinite(np.asarray(transition.actions)).all()
        assert np.isfinite(np.asarray(transition.next_observations)).all()
        assert np.isfinite(np.asarray(transition.rewards)).all()


def test_zeta_noise_repeat_is_cleared_on_done_in_eager_and_jit():
    noise = jnp.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
    repeat_count = jnp.array([3, 4, 5], dtype=jnp.int32)
    done = jnp.array([True, False, True])
    expected_noise = np.array([[0.0, 0.0], [2.0, -2.0], [0.0, 0.0]])
    expected_repeat = np.array([0, 3, 0], dtype=np.int32)

    def update():
        return reset_zeta_noise_on_done(noise, repeat_count, done)

    for next_noise, next_repeat in (update(), jax.jit(update)()):
        np.testing.assert_array_equal(next_noise, expected_noise)
        np.testing.assert_array_equal(next_repeat, expected_repeat)


@struct.dataclass
class _BoundaryCarry:
    autoreset_rng: jax.Array | None = None


@struct.dataclass
class _BoundaryState:
    data: jax.Array
    observation: jax.Array
    reward: jax.Array
    absorbing: jax.Array
    done: jax.Array
    info: dict[str, jax.Array]
    additional_carry: _BoundaryCarry


class _BoundaryEnv:
    """Batch containing tracking failure, clip end, horizon, and a live row."""

    def reset(self, _keys):
        observation = jnp.zeros((5, 5), dtype=jnp.float32)
        flag = jnp.zeros(5, dtype=bool)
        return observation, _BoundaryState(
            data=observation,
            observation=observation,
            reward=jnp.zeros(5),
            absorbing=flag,
            done=flag,
            info={
                **{name: jnp.zeros(5) for name, _ in flashsac_train.REWARD_INFO_TO_LOG_KEY},
                "done_numerical": flag,
                "done_trajectory": flag,
                "done_horizon": flag,
                "termination_site_position": flag,
            },
            additional_carry=_BoundaryCarry(),
        )

    def step(self, state, action):
        numerical = ~jnp.all(jnp.isfinite(action), axis=1)
        tracking = jnp.arange(5) == 0
        clip_end = jnp.arange(5) == 1
        horizon = jnp.arange(5) == 2
        absorbing = tracking | numerical
        done = absorbing | clip_end | horizon
        observation = jnp.full_like(state.observation, 7.0)
        reward = jnp.where(numerical, 0.0, 1.0)
        next_state = state.replace(
            data=observation,
            observation=observation,
            reward=reward,
            absorbing=absorbing,
            done=done,
            info={
                **state.info,
                "done_numerical": numerical,
                "done_trajectory": clip_end,
                "done_horizon": horizon,
                "termination_site_position": tracking,
            },
        )
        return observation, reward, absorbing, done, next_state.info, next_state


@pytest.mark.parametrize("compiled", [False, True])
def test_collection_update_preserves_transition_observation_and_actual_autoreset(compiled):
    agent_conf = _small_agent_conf()
    config = agent_conf.config.experiment
    config.update(
        num_envs=5,
        global_num_envs=5,
        min_buffer_size=1,
        sample_batch_size=8,
        noise_zeta_exponent=2.0,
        noise_max_repeat=16,
        updates_per_interaction=1,
    )
    agent = initialize_agent_state(jax.random.key(0), 5, 3, agent_conf)
    agent = agent.replace(
        replay_buffer=agent.replay_buffer.add(
            jnp.zeros((5, 5)),
            jnp.zeros((5, 3)),
            jnp.zeros(5),
            jnp.zeros((5, 5)),
            jnp.zeros(5, dtype=bool),
        )
    )
    env = AutoResetWrapper(LogWrapper(_BoundaryEnv()))
    observation, env_state = env.reset(jax.random.split(jax.random.key(1), 5))
    runtime = FlashSACRuntimeState(
        agent=agent,
        env_state=env_state,
        observation=observation,
        exploration_noise=jnp.full((5, 3), 0.2).at[3, 0].set(jnp.nan),
        noise_steps_remaining=jnp.full(5, 5, dtype=jnp.int32),
        rng=jax.random.key(2),
        validation_rng=jax.random.key(3),
        interaction_step=jnp.asarray(0, dtype=jnp.int32),
        reward_ema=jnp.asarray(0.0),
        discounted_return=jnp.ones(5),
        last_episode_return=jnp.asarray(0.0),
        last_episode_length=jnp.asarray(0.0),
    )
    training = FlashSACTraining(
        env=env,
        val_env=None,
        agent_conf=agent_conf,
        config=config,
        metrics_handler=None,
        resume_info=None,
        preserve_checkpoint_target=False,
        axis_name=None,
        shard_spec=None,
        validation_shard_spec=None,
        validation_shares_training_trajectory=True,
    )

    def update(state):
        return training._update(state, env)

    result = jax.jit(update)(runtime) if compiled else update(runtime)
    updated = result.runtime_state
    replay = updated.agent.replay_buffer
    np.testing.assert_array_equal(replay.terminated[5:10], [True, True, False, True, False])
    np.testing.assert_array_equal(replay.next_observations[5:10], np.full((5, 5), 7.0))
    np.testing.assert_array_equal(updated.observation[:4], np.zeros((4, 5)))
    np.testing.assert_array_equal(updated.observation[4], np.full(5, 7.0))
    np.testing.assert_array_equal(updated.noise_steps_remaining, [0, 0, 0, 0, 4])
    np.testing.assert_array_equal(updated.exploration_noise[:4], np.zeros((4, 3)))
    np.testing.assert_array_equal(updated.discounted_return[:4], np.zeros(4))
    assert int(updated.agent.update_step) == 1
    assert int(result.collection_metrics.done_count) == 4
    assert int(result.collection_metrics.numerical_count) == 1
    assert int(result.collection_metrics.tracking_termination_count) == 1
    assert int(result.collection_metrics.trajectory_end_count) == 1
    assert int(result.collection_metrics.horizon_end_count) == 1
    assert all(np.isfinite(np.asarray(value, dtype=np.float32)).all() for value in jax.tree.leaves(updated.agent))


def test_reward_component_logging_matches_shared_schema():
    info_keys = (
        "reward_total",
        "reward_qpos",
        "reward_qvel",
        "reward_root_pos",
        "reward_root_quat",
        "reward_rpos",
        "reward_rquat",
        "reward_rvel_rot",
        "reward_rvel_lin",
        "reward_root_vel",
        "penalty_total",
        "penalty_activation_energy",
        "err_root_xyz",
        "err_root_quat",
        "err_root_yaw",
        "err_joint_pos",
        "err_joint_vel",
        "err_site_abs",
        "err_rpos",
    )
    info = {key: jnp.array([index, index + 2.0], dtype=jnp.float32) for index, key in enumerate(info_keys)}

    logged = jax.jit(mean_reward_components)(info)

    assert logged["reward/total"] == pytest.approx(1.0)
    assert logged["err/rpos"] == pytest.approx(19.0)


def test_replay_buffer_wraps_and_samples_only_valid_entries():
    replay = create_replay_buffer(4, observation_dim=2, action_dim=1, storage_dtype="bfloat16")
    for offset in (0.0, 2.0, 4.0):
        observations = jnp.array([[offset, offset + 1], [offset + 1, offset + 2]])
        replay = replay.add(
            observations,
            jnp.ones((2, 1)) * offset,
            jnp.array([offset, offset + 1]),
            observations + 0.5,
            jnp.array([False, True]),
        )

    assert int(replay.size) == 4
    assert int(replay.position) == 2
    batch = replay.sample(jax.random.key(0), 128)
    assert batch.observations.dtype == jnp.float32
    assert set(np.asarray(batch.rewards).tolist()).issubset({2.0, 3.0, 4.0, 5.0})


def test_categorical_projection_preserves_probability_mass():
    log_probabilities = jnp.log(
        jnp.array(
            [
                [0.1, 0.2, 0.4, 0.2, 0.1],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        + 1e-8
    )
    projected = categorical_td_target(
        target_log_probabilities=log_probabilities,
        rewards=jnp.array([0.5, -0.5]),
        terminated=jnp.array([0.0, 1.0]),
        entropy_term=jnp.zeros(2),
        gamma=0.99,
        min_value=-2.0,
        max_value=2.0,
    )

    np.testing.assert_allclose(projected.sum(axis=-1), np.ones(2), atol=1e-6)
    np.testing.assert_allclose(projected[1], [0.0, 0.5, 0.5, 0.0, 0.0], atol=1e-6)


def test_min_q_distribution_selects_per_transition():
    values = jnp.array([[1.0, 3.0], [2.0, 2.0]])
    log_probabilities = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)

    selected = select_min_q_distribution(values, log_probabilities)

    np.testing.assert_array_equal(selected, [[0, 1, 2], [9, 10, 11]])


def test_actor_and_critic_outputs_are_finite():
    actor = FlashSACActor(action_dim=3, hidden_dim=16, num_blocks=1)
    critic = FlashSACDoubleCritic(hidden_dim=16, num_blocks=1, num_bins=11)
    observations = jnp.ones((8, 5))
    actions = jnp.zeros((8, 3))
    actor_variables = actor.init(jax.random.key(0), observations, training=False)
    critic_variables = critic.init(jax.random.key(1), observations, actions, training=False)

    mean, std = actor.apply(actor_variables, observations, training=False)
    values, log_probabilities = critic.apply(
        critic_variables,
        observations,
        actions,
        training=False,
    )
    sampled_actions, sampled_log_probabilities = sample_tanh_normal(
        mean,
        std,
        jax.random.key(2),
    )

    assert mean.shape == std.shape == sampled_actions.shape == (8, 3)
    assert sampled_log_probabilities.shape == (8,)
    assert values.shape == (2, 8)
    assert log_probabilities.shape == (2, 8, 11)
    assert np.isfinite(np.asarray(values)).all()
    assert np.isfinite(np.asarray(sampled_log_probabilities)).all()


def test_batch_norm_tracks_unbiased_running_variance():
    norm = UnitBatchNorm(momentum=0.01)
    inputs = jnp.array([[1.0], [3.0]])
    variables = norm.init(jax.random.key(0), inputs, training=False)

    _, mutable = norm.apply(variables, inputs, training=True, mutable=["batch_stats"])

    np.testing.assert_allclose(mutable["batch_stats"]["var"], [1.01], atol=1e-6)


def test_parameter_projection_makes_dense_columns_unit_norm():
    actor = FlashSACActor(action_dim=3, hidden_dim=16, num_blocks=1)
    variables = actor.init(jax.random.key(0), jnp.ones((8, 5)), training=False)
    normalized = normalize_parameters(jax.tree_util.tree_map(lambda value: value * 3.0, variables["params"]))

    def check(path, value):
        if path[-1].key == "kernel":
            np.testing.assert_allclose(
                jnp.linalg.norm(value, axis=0),
                np.ones(value.shape[-1]),
                atol=1e-5,
            )
        return value

    jax.tree_util.tree_map_with_path(check, normalized)


def test_one_compiled_gradient_update_is_finite():
    agent_conf = _small_agent_conf()
    state = initialize_agent_state(jax.random.key(0), 5, 3, agent_conf)
    assert float(state.reward_return_count) == 0.0
    observations = jax.random.normal(jax.random.key(1), (32, 5))
    actions = jnp.tanh(jax.random.normal(jax.random.key(2), (32, 3)))
    replay = state.replay_buffer.add(
        observations,
        actions,
        jnp.ones(32),
        observations + 0.1,
        jnp.zeros(32, dtype=bool),
    )
    state = state.replace(replay_buffer=replay, reward_return_max=jnp.array(10.0))
    batch = replay.sample(jax.random.key(3), 16)

    next_state, metrics = jax.jit(
        lambda current: update_from_batch(
            current,
            batch,
            jax.random.key(4),
            agent_conf,
        )
    )(state)

    assert int(next_state.update_step) == 1
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics)
    assert bool(metrics.has_actor_update)
    assert bool(metrics.has_critic_update)

    _, critic_only_metrics = jax.jit(
        lambda current: update_from_batch(
            current,
            batch,
            jax.random.key(5),
            agent_conf,
        )
    )(next_state)

    assert not bool(critic_only_metrics.has_actor_update)
    assert bool(critic_only_metrics.has_critic_update)


def test_policy_adapter_supports_deterministic_and_stochastic_actions():
    agent_conf = _small_agent_conf()
    train_state = initialize_agent_state(jax.random.key(0), 5, 3, agent_conf)
    observations = jnp.ones((2, 5), dtype=jnp.float32)

    deterministic, _ = make_policy_action_fn(agent_conf, deterministic=True)(
        train_state,
        observations,
        jax.random.key(1),
    )
    stochastic, _ = make_policy_action_fn(agent_conf, deterministic=False)(
        train_state,
        observations,
        jax.random.key(2),
    )

    assert deterministic.shape == stochastic.shape == (2, 3)
    assert np.isfinite(np.asarray(deterministic)).all()
    assert not np.allclose(deterministic, stochastic)


def test_checkpoint_state_restoration_preserves_replay_and_optimizers():
    agent_conf = _small_agent_conf(buffer_capacity=16)
    train_state = initialize_agent_state(jax.random.key(0), 5, 3, agent_conf)
    replay = train_state.replay_buffer.add(
        jnp.arange(20, dtype=jnp.float32).reshape(4, 5),
        jnp.ones((4, 3)),
        jnp.arange(4, dtype=jnp.float32),
        jnp.arange(20, dtype=jnp.float32).reshape(4, 5) + 1,
        jnp.array([False, False, True, False]),
    )
    train_state = train_state.replace(
        replay_buffer=replay,
        reward_return_mean=jnp.array(2.5),
        update_step=jnp.array(7, dtype=jnp.int32),
    )

    restored = restore_train_state(FlashSACAgentState(train_state).serialize(), agent_conf)

    assert int(restored.update_step) == 7
    assert int(restored.replay_buffer.size) == 4
    assert float(restored.reward_return_mean) == pytest.approx(2.5)
    np.testing.assert_allclose(restored.replay_buffer.rewards[:4], np.arange(4))


def test_resume_without_replay_reinitializes_empty_buffer(monkeypatch):
    agent_conf = _small_agent_conf(buffer_capacity=16)
    train_state = initialize_agent_state(jax.random.key(0), 5, 3, agent_conf).replace(
        reward_return_mean=jnp.array(2.5),
        update_step=jnp.array(7, dtype=jnp.int32),
    )
    state_data = FlashSACAgentState(train_state).serialize(include_replay_buffer=False)

    assert "replay_buffer" not in state_data
    learner_state = restore_train_state(state_data, agent_conf)
    assert isinstance(learner_state, FlashSACLearnerState)
    assert not hasattr(train_state_specs(learner_state), "replay_buffer")
    assert int(learner_state.update_step) == 7
    assert float(learner_state.reward_return_mean) == pytest.approx(2.5)

    wrapped_env = SimpleNamespace(
        info=SimpleNamespace(
            observation_space=SimpleNamespace(shape=(5,)),
            action_space=SimpleNamespace(shape=(3,)),
        ),
        reset=lambda _keys: (jnp.zeros((4, 5)), jnp.zeros((4, 1))),
    )
    monkeypatch.setattr(flashsac_train, "wrap_env", lambda *_args, **_kwargs: wrapped_env)
    training = FlashSACTraining(
        env=object(),
        val_env=None,
        agent_conf=agent_conf,
        config=agent_conf.config.experiment,
        metrics_handler=None,
        resume_info={"update_number": 11},
        preserve_checkpoint_target=False,
        axis_name=None,
        shard_spec=None,
        validation_shard_spec=None,
        validation_shares_training_trajectory=True,
    )

    runtime_state = training.initialize(jax.random.key(1), learner_state)

    assert isinstance(runtime_state.agent, FlashSACTrainState)
    assert runtime_state.agent.replay_buffer.capacity == 16
    assert int(runtime_state.agent.replay_buffer.size) == 0
    assert int(runtime_state.agent.replay_buffer.position) == 0
    assert int(runtime_state.agent.update_step) == 7
    assert int(runtime_state.interaction_step) == 11


@pytest.mark.parametrize("include_replay_buffer", [False, True])
def test_orbax_checkpoint_round_trip_supports_optional_replay(
    tmp_path,
    include_replay_buffer,
):
    agent_conf = _small_agent_conf(buffer_capacity=16)
    train_state = initialize_agent_state(jax.random.key(0), 5, 3, agent_conf)
    replay = train_state.replay_buffer.add(
        jnp.arange(20, dtype=jnp.float32).reshape(4, 5),
        jnp.ones((4, 3)),
        jnp.arange(4, dtype=jnp.float32),
        jnp.arange(20, dtype=jnp.float32).reshape(4, 5) + 1,
        jnp.array([False, False, True, False]),
    )
    train_state = train_state.replace(
        replay_buffer=replay,
        reward_return_mean=jnp.array(2.5),
        update_step=jnp.array(7, dtype=jnp.int32),
    )
    metadata = CheckpointMetadata(
        step=7,
        update_number=3,
        global_timestep=12,
        target_global_timestep=100,
        learning_rate=3e-4,
        num_envs=4,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        algo_version=FLASHSAC_ALGO_VERSION,
    )
    manager = UnifiedCheckpointManager(str(tmp_path), async_save=False)
    try:
        checkpoint_path = manager.save_checkpoint(
            3,
            agent_conf,
            FlashSACAgentState(train_state).serialize(include_replay_buffer=include_replay_buffer),
            metadata,
        )
    finally:
        manager.close()

    loader = UnifiedCheckpointManager(str(tmp_path), async_save=False)
    try:
        (_, state_data), restored_metadata = loader.load_checkpoint(checkpoint_path)
    finally:
        loader.close()
    restored = restore_train_state(state_data, agent_conf)

    assert restored_metadata.algo_version == FLASHSAC_ALGO_VERSION
    assert int(restored.update_step) == 7
    assert isinstance(restored, FlashSACTrainState if include_replay_buffer else FlashSACLearnerState)
    if include_replay_buffer:
        assert int(restored.replay_buffer.size) == 4

    resumed_agent, resume_info = load_checkpoint_for_resume(checkpoint_path, agent_conf)
    assert int(resumed_agent.train_state.update_step) == 7
    assert float(resumed_agent.train_state.reward_return_mean) == pytest.approx(2.5)
    assert isinstance(
        resumed_agent.train_state,
        FlashSACTrainState if include_replay_buffer else FlashSACLearnerState,
    )
    assert resume_info["update_number"] == 3
    assert resume_info["global_timestep"] == 12


@pytest.mark.parametrize(
    "algo_version",
    [LEGACY_FLASHSAC_ALGO_VERSION, "FlashSACJax_v2", "FlashSACJax_v3", "FlashSACJax_v4"],
)
@pytest.mark.parametrize("include_replay_buffer", [False, True])
def test_resume_requires_compatible_replay_and_migrates_legacy_statistics(
    tmp_path, algo_version, include_replay_buffer
):
    agent_conf = _small_agent_conf(buffer_capacity=16)
    train_state = initialize_agent_state(jax.random.key(0), 5, 3, agent_conf).replace(
        reward_return_mean=jnp.array(2.5),
        reward_return_var=jnp.array(3.5),
        reward_return_count=jnp.array(128.0),
        reward_return_max=jnp.array(9.0),
        update_step=jnp.array(7, dtype=jnp.int32),
    )
    metadata = CheckpointMetadata(
        step=7,
        update_number=3,
        global_timestep=12,
        target_global_timestep=100,
        learning_rate=3e-4,
        num_envs=4,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        algo_version=algo_version,
    )
    manager = UnifiedCheckpointManager(str(tmp_path), async_save=False)
    try:
        checkpoint_path = manager.save_checkpoint(
            3,
            agent_conf,
            FlashSACAgentState(train_state).serialize(include_replay_buffer=include_replay_buffer),
            metadata,
        )
    finally:
        manager.close()

    if include_replay_buffer:
        with pytest.raises(ValueError, match="incompatible terminal labels"):
            load_checkpoint_for_resume(checkpoint_path, agent_conf)
        return

    resumed_agent, resume_info = load_checkpoint_for_resume(checkpoint_path, agent_conf)

    assert int(resumed_agent.train_state.update_step) == 7
    expected = (0.0, 1.0, 0.0, 0.0) if algo_version == LEGACY_FLASHSAC_ALGO_VERSION else (2.5, 3.5, 128.0, 9.0)
    assert float(resumed_agent.train_state.reward_return_mean) == pytest.approx(expected[0])
    assert float(resumed_agent.train_state.reward_return_var) == pytest.approx(expected[1])
    assert float(resumed_agent.train_state.reward_return_count) == pytest.approx(expected[2])
    assert float(resumed_agent.train_state.reward_return_max) == pytest.approx(expected[3])
    assert resume_info["update_number"] == 3


@pytest.mark.parametrize(
    ("task_directory", "config_name"),
    [
        ("fullbody", "conf_fullbody_flashsac"),
        ("bimanual", "conf_bimanual_flashsac"),
    ],
)
def test_task_configs_select_single_device_flashsac(task_directory, config_name):
    repository_root = Path(__file__).parents[2]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(repository_root / task_directory),
    ):
        config = compose(config_name=config_name)

    assert pick_algorithm(config).__name__ == "FlashSACJax"
    assert config.experiment.env_params.mjx_backend == "warp"
    assert config.experiment.distributed.num_devices == 1
    assert config.experiment.normalize_env is False
    assert config.experiment.checkpoints_on_validation is False
    assert config.experiment.validation.video_max_motions == 3
    assert "reward_normalizer_version" not in config.experiment.flashsac_config
    assert config.experiment.task_factory.params.amass_dataset_conf.write_materialized_cache is True

    if task_directory == "fullbody":
        assert config.experiment.num_envs == 8192
        assert config.experiment.total_timesteps == 402_653_184
        assert config.experiment.online_logging_interval == 5
        assert config.experiment.env_params.horizon == 23_000
        assert config.experiment.env_params.reward_params.qpos_w_sum == pytest.approx(0.18)
        assert config.experiment.env_params.reward_params.qvel_w_sum == pytest.approx(0.14)
        assert config.experiment.env_params.reward_params.root_pos_w_sum == pytest.approx(0.14)
        assert config.experiment.env_params.reward_params.root_quat_w_sum == pytest.approx(0.09)
        assert config.experiment.env_params.reward_params.root_vel_w_sum == pytest.approx(0.14)
        assert config.experiment.env_params.reward_params.rpos_w_sum == pytest.approx(0.18)
        assert config.experiment.env_params.reward_params.rquat_w_sum == pytest.approx(0.08)
        assert config.experiment.env_params.reward_params.rvel_w_sum == pytest.approx(0.08)
        assert config.experiment.trajectory.sharding == "static"
        assert config.experiment.checkpoint_include_replay_buffer is True
        assert config.experiment.checkpoint_interval == 2048
        assert config.experiment.max_checkpoints_to_keep == 3
        assert config.experiment.validation.deterministic is True
        assert config.experiment.validation.eval_seed == 0
        assert config.experiment.validation.evaluate_all is True
        assert config.experiment.validation.num_steps == 23_000
        assert config.experiment.validation.horizon == 23_000
        assert config.experiment.validation.num == 12
        assert config.experiment.validation.num_envs == 284
        assert config.experiment.validation.video_frequency == 4
        assert config.experiment.validation.video_length == 3000
        assert config.experiment.flashsac_config.buffer_capacity == 2_097_152
        assert config.experiment.flashsac_config.min_buffer_size == 1_048_576
        assert config.experiment.flashsac_config.sample_batch_size == 16_384
        assert config.experiment.flashsac_config.updates_per_interaction == 2
        assert config.experiment.flashsac_config.update_chunk_size == 1
        assert config.experiment.flashsac_config.actor_hidden_dim == 768
        assert config.experiment.flashsac_config.actor_update_period == 4
        assert config.experiment.flashsac_config.critic_hidden_dim == 512
        assert config.experiment.flashsac_config.critic_min_value == pytest.approx(-5.0)
        assert config.experiment.flashsac_config.critic_max_value == pytest.approx(5.0)
        assert config.experiment.flashsac_config.initial_temperature == pytest.approx(1e-4)
        assert config.experiment.flashsac_config.warmup_updates == 4000
        assert config.flashsac.termination.id == "relative-mean100cm-root100cm"
        assert config.experiment.env_params.terminal_state_params.mean_site_deviation_threshold == 1.0
        assert config.experiment.env_params.terminal_state_params.root_deviation_threshold == 1.0
        assert config.experiment.env_params.terminal_state_params.root_orientation_threshold == 1.0
    else:
        assert config.experiment.num_envs == 512
        assert config.experiment.trajectory.sharding == "none"
        assert config.experiment.checkpoint_include_replay_buffer is False
        assert config.experiment.validation.video_frequency == 20
        assert config.experiment.validation.video_length == 1000
        assert config.experiment.flashsac_config.buffer_capacity == 1_048_576


@pytest.mark.parametrize(
    ("profile", "expected_id", "expected_type"),
    [
        ("phc", "phc-max25cm", "MeanSiteDeviationTerminalStateHandler"),
        ("loose", "relative-mean100cm-root100cm", "MeanRelativeSiteDeviationWithRootTerminalStateHandler"),
        ("no_tracking", "no-tracking-et", "NoTerminalStateHandler"),
    ],
)
def test_fullbody_flashsac_termination_profiles(profile, expected_id, expected_type):
    repository_root = Path(__file__).parents[2]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(repository_root / "fullbody"),
    ):
        config = compose(
            config_name="conf_fullbody_flashsac",
            overrides=[f"flashsac/termination={profile}"],
        )

    assert config.flashsac.termination.id == expected_id
    assert config.experiment.env_params.terminal_state_type == expected_type
    assert config.experiment.env_params.terminal_state_params == config.flashsac.termination.params


def test_config_rejects_exact_resume():
    experiment = OmegaConf.create(
        {
            "total_timesteps": 100,
            "num_envs": 4,
            "normalize_env": False,
            "exact_resume": True,
        }
    )

    with pytest.raises(ValueError, match="does not support exact_resume"):
        get_flashsac_config(experiment)


def test_completed_interaction_budget_does_not_resume_for_a_partial_batch():
    experiment = OmegaConf.create(
        {
            "total_timesteps": 100_000_000,
            "num_envs": 512,
            "num_interactions": 195_312,
        }
    )
    final_timestep = int(experiment.num_interactions) * int(experiment.num_envs)
    fresh_training = FlashSACTraining(
        env=object(),
        val_env=None,
        agent_conf=SimpleNamespace(config=OmegaConf.create({"experiment": experiment})),
        config=OmegaConf.merge(experiment, {"global_num_envs": 512}),
        metrics_handler=None,
        resume_info=None,
        preserve_checkpoint_target=False,
        axis_name=None,
        shard_spec=None,
        validation_shard_spec=None,
        validation_shares_training_trajectory=True,
    )
    fresh_progress = fresh_training.progress()
    training = replace(
        fresh_training,
        resume_info={
            "update_number": int(experiment.num_interactions),
            "global_timestep": final_timestep,
            "target_global_timestep": int(experiment.total_timesteps),
        },
        preserve_checkpoint_target=True,
    )

    progress = training.progress()

    assert fresh_progress.target_global_timestep == final_timestep
    assert progress.remaining_interactions == 0
    assert progress.target_global_timestep == final_timestep


def test_build_training_localizes_global_data_sizes(monkeypatch):
    from musclemimic.algorithms.flashsac import train

    monkeypatch.setattr(train, "device_count", lambda _config: 4)
    experiment = OmegaConf.create(
        {
            "distributed": {"num_devices": 4},
            "trajectory": {"sharding": "none"},
            "num_envs": 2048,
            "buffer_capacity": 4_194_304,
            "min_buffer_size": 524_288,
            "sample_batch_size": 8192,
        }
    )
    training = build_training(
        object(),
        SimpleNamespace(config=OmegaConf.create({"experiment": experiment})),
    )

    assert training.config.num_envs == 512
    assert training.config.buffer_capacity == 1_048_576
    assert training.config.min_buffer_size == 131_072
    assert training.config.sample_batch_size == 2048
    assert training.config.global_num_envs == 2048
    assert training.shard_spec is None
    assert training.validation_shard_spec is None


def test_build_training_assigns_training_and_validation_motion_shards(monkeypatch):
    from musclemimic.algorithms.flashsac import train

    monkeypatch.setattr(train, "device_count", lambda _config: 4)
    training_trajectory = object()
    validation_trajectory = object()
    env = SimpleNamespace(th=SimpleNamespace(traj=training_trajectory))
    val_env = SimpleNamespace(th=SimpleNamespace(traj=validation_trajectory))
    experiment = OmegaConf.create(
        {
            "distributed": {"num_devices": 4},
            "trajectory": {"sharding": "static"},
            "n_trajectories": 10,
            "validation": {"n_trajectories": 6},
            "num_envs": 16,
            "buffer_capacity": 64,
            "min_buffer_size": 16,
            "sample_batch_size": 8,
        }
    )

    training = build_training(
        env,
        SimpleNamespace(config=OmegaConf.create({"experiment": experiment})),
        val_env=val_env,
    )

    assert training.shard_spec.counts == (3, 3, 2, 2)
    assert training.validation_shard_spec.counts == (2, 2, 1, 1)
    assert training.validation_shares_training_trajectory is False


def test_build_training_allows_single_device_static_trajectory():
    experiment = OmegaConf.create(
        {
            "distributed": {"num_devices": 1},
            "trajectory": {"sharding": "static"},
            "n_trajectories": 10,
            "validation": {},
            "num_envs": 16,
            "buffer_capacity": 64,
            "min_buffer_size": 16,
            "sample_batch_size": 8,
        }
    )

    training = build_training(
        object(),
        SimpleNamespace(config=OmegaConf.create({"experiment": experiment})),
    )

    assert training.shard_spec.counts == (10,)
    assert training.shard_spec.offsets == (0, 10)
    assert training.config.num_envs == 16
    assert training.config.global_num_envs == 16


def test_build_training_exact_validation_supports_replicated_single_device_data():
    training_trajectory = object()
    validation_trajectory = object()
    env = SimpleNamespace(th=SimpleNamespace(traj=training_trajectory))
    val_env = SimpleNamespace(
        th=SimpleNamespace(
            traj=validation_trajectory,
            n_trajectories=5,
        )
    )
    experiment = OmegaConf.create(
        {
            "distributed": {"num_devices": 1},
            "trajectory": {"sharding": "none"},
            "validation": {
                "evaluate_all": True,
                "num_envs": 5,
            },
            "num_envs": 16,
            "buffer_capacity": 64,
            "min_buffer_size": 16,
            "sample_batch_size": 8,
        }
    )

    training = build_training(
        env,
        SimpleNamespace(config=OmegaConf.create({"experiment": experiment})),
        val_env=val_env,
    )

    assert training.shard_spec is None
    assert training.validation_shard_spec.counts == (5,)


def test_build_training_exact_validation_rejects_too_few_environments():
    training_trajectory = object()
    validation_trajectory = object()
    env = SimpleNamespace(th=SimpleNamespace(traj=training_trajectory))
    val_env = SimpleNamespace(
        th=SimpleNamespace(
            traj=validation_trajectory,
            n_trajectories=5,
        )
    )
    experiment = OmegaConf.create(
        {
            "distributed": {"num_devices": 1},
            "trajectory": {"sharding": "none"},
            "validation": {
                "evaluate_all": True,
                "num_envs": 4,
            },
            "num_envs": 16,
            "buffer_capacity": 64,
            "min_buffer_size": 16,
            "sample_batch_size": 8,
        }
    )

    with pytest.raises(ValueError, match="got 4, need 5"):
        build_training(
            env,
            SimpleNamespace(config=OmegaConf.create({"experiment": experiment})),
            val_env=val_env,
        )
