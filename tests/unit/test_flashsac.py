"""Focused tests for FlashSAC's numerical and storage invariants."""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from loco_mujoco.trajectory.dataclasses import TrajectoryData
from loco_mujoco.trajectory.handler import TrajectoryHandler
from musclemimic.algorithms.common.base_algorithm import JaxTrainingInvocation
from musclemimic.algorithms.common.checkpoint_manager import (
    CheckpointMetadata,
    UnifiedCheckpointManager,
)
from musclemimic.algorithms.common.inference import (
    initialize_policy_state,
    make_policy_action_fn,
)
from musclemimic.algorithms.flashsac import flashsac as flashsac_module
from musclemimic.algorithms.flashsac.checkpoint import restore_train_state
from musclemimic.algorithms.flashsac.config import (
    FlashSACAgentConf,
    FlashSACAgentState,
)
from musclemimic.algorithms.flashsac.flashsac import FlashSACJax
from musclemimic.algorithms.flashsac.losses import (
    categorical_td_target,
    sample_tanh_normal,
    select_min_q_distribution,
)
from musclemimic.algorithms.flashsac.networks import (
    FlashSACActor,
    FlashSACDoubleCritic,
    normalize_parameters,
)
from musclemimic.algorithms.flashsac.replay_buffer import create_replay_buffer
from musclemimic.algorithms.flashsac.runner import (
    _init_agent_state,
    _mean_reward_components,
    _select_collection_actions,
)
from musclemimic.algorithms.flashsac.update import reward_normalization_scales, update_from_batch
from musclemimic.algorithms.flashsac.validation import run_validation
from musclemimic.runner.engine import pick_algorithm


def _small_agent_conf(buffer_capacity: int = 64) -> FlashSACAgentConf:
    experiment = OmegaConf.create(
        {
            "algorithm": "FlashSACJax",
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
            "fixed_reward_normalization_scale": None,
            "num_envs": 4,
            "num_steps": 1,
            "num_minibatches": 1,
            "update_epochs": 1,
        }
    )
    schedule = optax.constant_schedule(3e-4)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    return FlashSACAgentConf(
        config=OmegaConf.create({"experiment": experiment}),
        actor=FlashSACActor(action_dim=3, hidden_dim=8, num_blocks=1),
        critic=FlashSACDoubleCritic(hidden_dim=8, num_blocks=1, num_bins=5),
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
        temperature_optimizer=optax.adam(3e-4),
        actor_learning_rate_schedule=schedule,
        critic_learning_rate_schedule=schedule,
        learning_rate_schedule=schedule,
        actor_observation_indices=jnp.arange(5),
        critic_observation_indices=jnp.arange(5),
    )


def test_collection_actions_are_uniform_until_replay_warmup_finishes():
    key = jax.random.PRNGKey(7)
    policy_actions = jnp.full((8, 3), 0.25, dtype=jnp.float32)
    expected_uniform = jax.random.uniform(
        key,
        policy_actions.shape,
        minval=-1.0,
        maxval=1.0,
        dtype=policy_actions.dtype,
    )
    select_actions = jax.jit(
        lambda replay_size: _select_collection_actions(
            replay_size,
            min_buffer_size=16,
            policy_actions=policy_actions,
            uniform_rng=key,
        )
    )

    np.testing.assert_allclose(np.asarray(select_actions(jnp.array(15))), np.asarray(expected_uniform))
    np.testing.assert_allclose(np.asarray(select_actions(jnp.array(16))), np.asarray(policy_actions))


def test_reward_component_logging_matches_ppo_schema():
    expected_mapping = {
        "reward_total": "reward/total",
        "reward_qpos": "reward/qpos",
        "reward_qvel": "reward/qvel",
        "reward_root_pos": "reward/root_pos",
        "reward_root_quat": "reward/root_quat",
        "reward_rpos": "reward/rpos",
        "reward_rquat": "reward/rquat",
        "reward_rvel_rot": "reward/rvel_rot",
        "reward_rvel_lin": "reward/rvel_lin",
        "reward_root_vel": "reward/root_vel",
        "penalty_total": "reward/penalty",
        "penalty_activation_energy": "reward/penalty_activation_energy",
        "err_root_xyz": "err/root_xyz",
        "err_root_yaw": "err/root_yaw",
        "err_joint_pos": "err/joint_pos",
        "err_joint_vel": "err/joint_vel",
        "err_site_abs": "err/site_abs",
        "err_rpos": "err/rpos",
    }
    info = {
        info_key: jnp.array([index, index + 2.0], dtype=jnp.float32)
        for index, info_key in enumerate(expected_mapping)
    }

    logged = jax.jit(_mean_reward_components)(info)

    assert set(logged) == set(expected_mapping.values())
    for index, log_key in enumerate(expected_mapping.values()):
        assert float(logged[log_key]) == pytest.approx(index + 1.0)


def test_replay_buffer_wraps_and_only_samples_valid_entries():
    buffer = create_replay_buffer(4, observation_dim=2, action_dim=1)
    for offset in (0.0, 2.0, 4.0):
        observations = jnp.array([[offset, offset + 1], [offset + 1, offset + 2]])
        buffer = buffer.add(
            observations=observations,
            actions=jnp.ones((2, 1)) * offset,
            rewards=jnp.array([offset, offset + 1]),
            next_observations=observations + 0.5,
            terminated=jnp.array([False, True]),
        )

    assert int(buffer.size) == 4
    assert int(buffer.position) == 2
    batch = buffer.sample(jax.random.PRNGKey(0), 128)
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

    np.testing.assert_allclose(np.asarray(projected.sum(axis=-1)), np.ones(2), atol=1e-6)
    # A terminal reward of -0.5 lies halfway between the -1 and 0 bins.
    np.testing.assert_allclose(np.asarray(projected[1]), [0.0, 0.5, 0.5, 0.0, 0.0], atol=1e-6)


def test_min_q_distribution_selects_per_item():
    values = jnp.array([[1.0, 3.0], [2.0, 2.0]])
    log_probabilities = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)
    selected = select_min_q_distribution(values, log_probabilities)
    np.testing.assert_array_equal(np.asarray(selected), np.asarray([[0, 1, 2], [9, 10, 11]]))


def test_actor_and_critic_shapes_are_finite():
    actor = FlashSACActor(action_dim=3, hidden_dim=16, num_blocks=1)
    critic = FlashSACDoubleCritic(hidden_dim=16, num_blocks=1, num_bins=11)
    observations = jnp.ones((8, 5))
    actions = jnp.zeros((8, 3))

    actor_variables = actor.init(jax.random.PRNGKey(0), observations, training=False)
    critic_variables = critic.init(jax.random.PRNGKey(1), observations, actions, training=False)
    mean, std = actor.apply(actor_variables, observations, training=False)
    values, log_probabilities = critic.apply(critic_variables, observations, actions, training=False)
    sampled_actions, sampled_log_probabilities = sample_tanh_normal(mean, std, jax.random.PRNGKey(2))

    assert mean.shape == std.shape == sampled_actions.shape == (8, 3)
    assert sampled_log_probabilities.shape == (8,)
    assert values.shape == (2, 8)
    assert log_probabilities.shape == (2, 8, 11)
    assert np.isfinite(np.asarray(values)).all()
    assert np.isfinite(np.asarray(sampled_log_probabilities)).all()


def test_parameter_projection_makes_dense_columns_unit_norm():
    actor = FlashSACActor(action_dim=3, hidden_dim=16, num_blocks=1)
    variables = actor.init(jax.random.PRNGKey(0), jnp.ones((8, 5)), training=False)
    scaled = jax.tree_util.tree_map(lambda value: value * 3.0, variables["params"])
    normalized = normalize_parameters(scaled)

    def visit(path, value):
        if path[-1].key == "kernel":
            np.testing.assert_allclose(
                np.asarray(jnp.linalg.norm(value, axis=0)),
                np.ones(value.shape[-1]),
                atol=1e-5,
            )
        return value

    jax.tree_util.tree_map_with_path(visit, normalized)


def test_trajectory_handler_converts_mixed_backend_fields_to_jax():
    """Bimanual caches may contain JAX samples but NumPy split points."""

    @dataclass(frozen=True)
    class FakeModel:
        def to_jax(self):
            return self

    @dataclass(frozen=True)
    class FakeInfo:
        model: FakeModel

    @dataclass(frozen=True)
    class FakeTrajectory:
        data: TrajectoryData
        info: FakeInfo

    handler = object.__new__(TrajectoryHandler)
    handler.traj = FakeTrajectory(
        data=TrajectoryData(
            qpos=jnp.zeros((2, 1)),
            qvel=jnp.zeros((2, 1)),
            split_points=np.array([0, 2]),
        ),
        info=FakeInfo(model=FakeModel()),
    )
    # This was previously inferred only from qpos and caused to_jax() to skip.
    handler._is_numpy = False

    handler.to_jax()

    assert isinstance(handler.traj.data.qpos, jax.Array)
    assert isinstance(handler.traj.data.split_points, jax.Array)


def test_one_compiled_update_is_finite():
    agent_conf = _small_agent_conf()
    state = _init_agent_state(jax.random.PRNGKey(0), 5, 3, agent_conf)
    observations = jax.random.normal(jax.random.PRNGKey(1), (32, 5))
    actions = jnp.tanh(jax.random.normal(jax.random.PRNGKey(2), (32, 3)))
    replay = state.replay_buffer.add(
        observations,
        actions,
        jnp.ones(32),
        observations + 0.1,
        jnp.zeros(32, dtype=bool),
    )
    state = state.replace(replay_buffer=replay, reward_return_max=jnp.array(10.0))
    batch = replay.sample(jax.random.PRNGKey(3), 16)

    next_state, metrics = jax.jit(
        lambda current_state: update_from_batch(
            current_state,
            batch,
            jax.random.PRNGKey(4),
            agent_conf,
        )
    )(state)

    assert int(next_state.update_step) == 1
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics)


def test_fixed_reward_normalization_is_stationary_but_keeps_running_diagnostic():
    agent_conf = _small_agent_conf()
    state = _init_agent_state(jax.random.PRNGKey(0), 5, 3, agent_conf).replace(
        reward_return_var=jnp.array(64.0),
        reward_return_max=jnp.array(100.0),
    )
    running_config = agent_conf.config.experiment
    applied, running = reward_normalization_scales(state, running_config)
    assert float(applied) == pytest.approx(20.0)
    assert float(running) == pytest.approx(20.0)

    fixed_config = OmegaConf.merge(
        running_config,
        {"fixed_reward_normalization_scale": 130.0},
    )
    applied, running = reward_normalization_scales(state, fixed_config)
    assert float(applied) == pytest.approx(130.0)
    assert float(running) == pytest.approx(20.0)


def test_actor_and_critic_lr_schedules_share_environment_timestep_progress():
    experiment = OmegaConf.create(
        {
            "num_envs": 2048,
            "num_updates": 488282,
            "updates_per_interaction": 2,
            "actor_update_period": 2,
            "learning_rate": 3e-4,
            "end_learning_rate": 5e-5,
            "warmup_updates": 0,
            "learning_rate_warmup_timesteps": 2_000_000,
            "learning_rate_decay_timesteps": 300_000_000,
        }
    )
    critic_schedule = FlashSACJax._learning_rate_schedule(experiment, actor=False)
    actor_schedule = FlashSACJax._learning_rate_schedule(experiment, actor=True)
    critic_decay_updates = ((300_000_000 + 2047) // 2048) * 2
    actor_decay_updates = (critic_decay_updates + 1) // 2
    critic_mid = critic_decay_updates // 2
    actor_mid = (critic_mid + 1) // 2

    assert float(critic_schedule(critic_mid)) == pytest.approx(float(actor_schedule(actor_mid)), rel=1e-5)
    assert float(critic_schedule(critic_decay_updates)) == pytest.approx(5e-5, rel=1e-5)
    assert float(actor_schedule(actor_decay_updates)) == pytest.approx(5e-5, rel=1e-5)


def test_exhaustive_validation_evaluates_each_motion_once_and_masks_autoresets():
    class FakeState(NamedTuple):
        step: jax.Array
        trajectory_index: jax.Array

    class FakeTrajectoryHandler:
        n_trajectories = 3

        @staticmethod
        def len_trajectory(index):
            raise AssertionError("trajectory lengths must not be read while tracing validation")

    class FakeMetricsHandler:
        _trajectory_handler = FakeTrajectoryHandler()
        _trajectory_lengths = (2, 4, 3)
        _validation_horizon = 4

    class FakeEnv:
        @staticmethod
        def reset_to(_keys, trajectory_indices):
            num_envs = trajectory_indices.shape[0]
            return (
                jnp.zeros((num_envs, 5), dtype=jnp.float32),
                FakeState(
                    step=jnp.zeros((num_envs,), dtype=jnp.int32),
                    trajectory_index=trajectory_indices,
                ),
            )

        @staticmethod
        def step_with_transition(state, _actions):
            next_step = state.step + 1
            target_lengths = jnp.asarray([2, 4, 3], dtype=jnp.int32)[state.trajectory_index]
            forced_failure = (state.trajectory_index == 1) & (next_step == 2)
            done = (next_step >= target_lengths) | forced_failure
            autoreset_step = jnp.where(done, 0, next_step)
            next_state = FakeState(autoreset_step, state.trajectory_index)
            observations = jnp.zeros((state.step.shape[0], 5), dtype=jnp.float32)
            rewards = jnp.ones_like(state.step, dtype=jnp.float32)
            info = {
                "reward_total": rewards,
                "err_joint_pos": next_step.astype(jnp.float32),
            }
            return (
                observations,
                rewards,
                forced_failure,
                done,
                info,
                next_state,
                next_state,
                observations,
            )

    agent_conf = _small_agent_conf()
    train_state = _init_agent_state(jax.random.key(0), 5, 3, agent_conf)
    config = OmegaConf.create(
        {
            "validation_interval": 1,
            "validation": {
                "evaluate_all": True,
                "num_envs": 4,
                "deterministic": True,
            },
        }
    )
    metrics, _rng = jax.jit(
        lambda key: run_validation(
            agent_conf,
            train_state,
            key,
            FakeEnv(),
            config,
            FakeMetricsHandler(),
            jnp.array(1, dtype=jnp.int32),
        )
    )(jax.random.key(1))

    np.testing.assert_allclose(np.asarray(metrics.per_motion_success), [1.0, 0.0, 1.0])
    np.testing.assert_allclose(np.asarray(metrics.per_motion_frame_coverage), [1.0, 0.5, 1.0])
    assert float(metrics.motion_success_rate) == pytest.approx(2.0 / 3.0)
    assert float(metrics.early_termination_rate) == pytest.approx(1.0 / 3.0)
    assert float(metrics.mean_frame_coverage) == pytest.approx(5.0 / 6.0)
    assert float(metrics.worst_quartile_frame_coverage) == pytest.approx(0.5)
    assert float(metrics.mean_episode_length) == pytest.approx(7.0 / 3.0)
    assert float(metrics.mean_return_per_frame) == pytest.approx(1.0)
    assert float(metrics.horizon_timeout_count) == pytest.approx(0.0)


def test_policy_adapter_runs_deterministic_and_stochastic_actions():
    agent_conf = _small_agent_conf()
    train_state = _init_agent_state(jax.random.key(0), 5, 3, agent_conf)
    observations = jnp.ones((2, 5), dtype=jnp.float32)
    policy_state, reset = initialize_policy_state(agent_conf, observations)

    deterministic_fn = make_policy_action_fn(agent_conf, deterministic=True)
    stochastic_fn = make_policy_action_fn(agent_conf, deterministic=False)
    deterministic, _, _ = deterministic_fn(
        train_state,
        observations,
        jax.random.key(1),
        policy_state,
        reset,
    )
    stochastic, _, _ = stochastic_fn(
        train_state,
        observations,
        jax.random.key(2),
        policy_state,
        reset,
    )

    assert deterministic.shape == stochastic.shape == (2, 3)
    assert np.isfinite(np.asarray(deterministic)).all()
    assert not np.allclose(np.asarray(deterministic), np.asarray(stochastic))


def test_orbax_checkpoint_round_trip_preserves_complete_state(tmp_path):
    agent_conf = _small_agent_conf(buffer_capacity=16)
    train_state = _init_agent_state(jax.random.key(0), 5, 3, agent_conf)
    replay = train_state.replay_buffer.add(
        observations=jnp.arange(20, dtype=jnp.float32).reshape(4, 5),
        actions=jnp.ones((4, 3), dtype=jnp.float32),
        rewards=jnp.arange(4, dtype=jnp.float32),
        next_observations=jnp.arange(20, dtype=jnp.float32).reshape(4, 5) + 1,
        terminated=jnp.array([False, False, True, False]),
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
        algo_version="FlashSACJax_v1",
    )

    manager = UnifiedCheckpointManager(
        str(tmp_path),
        max_to_keep=2,
        async_save=False,
    )
    checkpoint_path = manager.save_checkpoint(
        3,
        agent_conf,
        FlashSACAgentState(train_state=train_state),
        metadata,
    )
    manager.close()

    loader = UnifiedCheckpointManager(
        str(tmp_path),
        max_to_keep=2,
        async_save=False,
    )
    try:
        (_, state_dict), loaded_metadata = loader.load_checkpoint(checkpoint_path)
    finally:
        loader.close()
    restored = restore_train_state(state_dict, agent_conf)

    assert loaded_metadata.algo_version == "FlashSACJax_v1"
    assert int(restored.update_step) == 7
    assert int(restored.replay_buffer.size) == 4
    assert float(restored.reward_return_mean) == pytest.approx(2.5)
    np.testing.assert_allclose(
        np.asarray(restored.replay_buffer.rewards[:4]),
        np.arange(4, dtype=np.float32),
    )


def test_resume_builder_keeps_restored_replay_out_of_jax_constants(monkeypatch):
    agent_conf = _small_agent_conf(buffer_capacity=16)
    restored_state = FlashSACAgentState(train_state=_init_agent_state(jax.random.key(0), 5, 3, agent_conf))
    resume_info = {
        "update_number": 4,
        "global_timestep": 16,
        "target_global_timestep": 100,
        "num_envs": 4,
        "num_steps": 1,
        "num_minibatches": 1,
        "update_epochs": 1,
    }
    monkeypatch.setattr(
        flashsac_module,
        "load_checkpoint_for_resume",
        lambda _path, _conf: (restored_state, resume_info),
    )

    def fake_train(
        cls,
        _rng,
        _env,
        _agent_conf,
        agent_state,
        **_kwargs,
    ):
        del cls
        return agent_state.train_state.replay_buffer.observations

    monkeypatch.setattr(FlashSACJax, "_train_fn", classmethod(fake_train))

    invocation = FlashSACJax.build_resume_train_fn_from_path(
        env=None,
        agent_conf=agent_conf,
        checkpoint_path="/unused/checkpoint_4",
    )

    assert isinstance(invocation, JaxTrainingInvocation)
    assert len(invocation.dynamic_args) == 1
    assert invocation.dynamic_args[0] is restored_state
    assert invocation.donate_argnums == (1,)
    closed_jaxpr = jax.make_jaxpr(invocation.train_fn)(
        jax.random.key(1),
        *invocation.dynamic_args,
    )
    assert not closed_jaxpr.consts


@pytest.mark.parametrize(
    ("task_directory", "config_name", "num_envs"),
    [
        ("fullbody", "conf_fullbody_flashsac", 512),
        ("bimanual", "conf_bimanual_flashsac", 512),
    ],
)
def test_task_configs_select_flashsac(task_directory, config_name, num_envs):
    repository_root = Path(__file__).parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(repository_root / task_directory)):
        config = compose(config_name=config_name)

    assert pick_algorithm(config).__name__ == "FlashSACJax"
    assert config.experiment.num_envs == num_envs
    assert config.experiment.normalize_env is False
    assert config.experiment.validation.active is True
    assert config.experiment.save_checkpoints is True
    assert config.experiment.auto_resume is True
    assert config.experiment.flashsac_config.sample_batch_size == 2048


def test_terra_single_motion_config_reads_launcher_contract(monkeypatch):
    repository_root = Path(__file__).parents[2]
    monkeypatch.setenv("TERRA_RETARGETED_MOTIONS", "/tmp/published-terra")
    monkeypatch.setenv("TERRA_MOTION", "Study/Subject/Trial")
    monkeypatch.setenv("TERRA_VALIDATION_STEPS", "237")
    with initialize_config_dir(version_base=None, config_dir=str(repository_root / "fullbody")):
        config = compose(config_name="conf_terra_flashsac_single")

    train_dataset = config.experiment.task_factory.params.amass_dataset_conf
    validation_dataset = config.experiment.validation.amass_dataset_conf
    assert train_dataset.cache_root == "/tmp/published-terra"
    assert train_dataset.rel_dataset_path == "Study/Subject/Trial"
    assert validation_dataset.rel_dataset_path == "Study/Subject/Trial"
    assert config.experiment.validation.num_steps == 237
    assert isinstance(config.experiment.validation.num_steps, int)


@pytest.mark.parametrize(
    ("task_directory", "config_name"),
    [
        ("fullbody", "conf_fullbody_flashsac"),
        ("bimanual", "conf_bimanual_flashsac"),
    ],
)
def test_a5000_task_configs_prioritize_replay_coverage(task_directory, config_name):
    repository_root = Path(__file__).parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(repository_root / task_directory)):
        config = compose(config_name=config_name)

    flashsac = config.experiment.flashsac_config
    replay_interactions = flashsac.buffer_capacity // config.experiment.num_envs
    samples_per_new_transition = (
        flashsac.sample_batch_size
        * flashsac.updates_per_interaction
        / config.experiment.num_envs
    )

    assert flashsac.buffer_capacity == 1048576
    assert flashsac.min_buffer_size == 131072
    assert replay_interactions == 2048
    assert samples_per_new_transition == pytest.approx(4.0)
    assert flashsac.actor_hidden_dim == 256
    assert flashsac.actor_num_blocks == 4
    assert flashsac.critic_hidden_dim == 512
    assert flashsac.critic_num_blocks == 4
    assert flashsac.target_sigma == pytest.approx(0.20)
    assert config.experiment.checkpoints_on_validation is False
    assert config.experiment.checkpoint_interval == 8192
    assert config.experiment.async_checkpointing is True
    if task_directory == "bimanual":
        dataset = config.experiment.task_factory.params.amass_dataset_conf
        assert dataset.max_motions == 384
        assert dataset.max_motions_seed == 0


@pytest.mark.parametrize(
    (
        "task_directory",
        "config_name",
        "num_envs",
        "buffer_capacity",
        "min_buffer_size",
        "updates_per_interaction",
        "warmup_updates",
    ),
    [
        ("fullbody", "conf_fullbody_flashsac_large_scale", 2048, 4194304, 524288, 2, 2000),
        ("bimanual", "conf_bimanual_flashsac_large_scale", 8192, 1048576, 131072, 4, 1000),
    ],
)
def test_h100_task_configs_scale_flashsac_as_a_complete_profile(
    task_directory,
    config_name,
    num_envs,
    buffer_capacity,
    min_buffer_size,
    updates_per_interaction,
    warmup_updates,
):
    repository_root = Path(__file__).parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(repository_root / task_directory)):
        config = compose(config_name=config_name)

    flashsac = config.experiment.flashsac_config
    assert pick_algorithm(config).__name__ == "FlashSACJax"
    assert config.experiment.num_envs == num_envs
    assert flashsac.buffer_capacity == buffer_capacity
    assert flashsac.min_buffer_size == min_buffer_size
    assert flashsac.sample_batch_size == 4096
    assert flashsac.updates_per_interaction == updates_per_interaction
    assert flashsac.actor_hidden_dim == 512
    assert flashsac.actor_num_blocks == 8
    assert flashsac.critic_hidden_dim == 512
    assert flashsac.critic_num_blocks == 8
    assert flashsac.warmup_updates == warmup_updates
    assert config.experiment.checkpoints_on_validation is False
    assert config.experiment.checkpoint_interval == 1024
    assert config.experiment.async_checkpointing is True
    assert config.experiment.matmul_precision == "tensorfloat32"
    assert {"h100", "large-scale"}.intersection(config.wandb.tags)
