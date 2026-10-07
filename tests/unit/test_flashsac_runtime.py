"""Host-coordinator tests for FlashSAC."""

from dataclasses import replace
from threading import get_ident
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.flashsac.state import FlashSACPolicyState
from musclemimic.algorithms.flashsac.train import (
    FlashSACChunk,
    FlashSACCollectionMetrics,
    FlashSACProgress,
    FlashSACTraining,
)
from musclemimic.algorithms.flashsac.update import FlashSACUpdateMetrics
from musclemimic.runner import flashsac_runtime
from musclemimic.utils.metrics import ValidationSummary


@struct.dataclass
class _Replay:
    observations: jax.Array
    next_observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    position: jax.Array
    size: jax.Array


@struct.dataclass
class _Agent:
    actor_params: dict
    actor_batch_stats: dict
    critic_params: dict
    replay_buffer: _Replay
    update_step: jax.Array


@struct.dataclass
class _Runtime:
    agent: _Agent
    env_state: jax.Array
    observation: jax.Array
    exploration_noise: jax.Array
    noise_steps_remaining: jax.Array
    rng: jax.Array
    validation_rng: jax.Array
    interaction_step: jax.Array
    reward_ema: jax.Array
    discounted_return: jax.Array
    last_episode_return: jax.Array
    last_episode_length: jax.Array


class _Step(NamedTuple):
    runtime_state: _Runtime
    collection_metrics: FlashSACCollectionMetrics
    update_metrics: FlashSACUpdateMetrics
    learning_rate: jax.Array


class _ValidationResult(NamedTuple):
    metrics: ValidationSummary
    rng: jax.Array


class _Training(FlashSACTraining):
    def initialize(self, _rng, _resume_state=None, _trajectory_override=None):
        replay = _Replay(
            observations=jnp.zeros((8, 2), dtype=jnp.bfloat16),
            next_observations=jnp.zeros((8, 2), dtype=jnp.bfloat16),
            actions=jnp.zeros((8, 1), dtype=jnp.bfloat16),
            rewards=jnp.zeros((8,), dtype=jnp.float32),
            terminated=jnp.zeros((8,), dtype=bool),
            position=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
        )
        agent = _Agent(
            actor_params={"weight": jnp.ones((2,))},
            actor_batch_stats={},
            critic_params={"weight": jnp.ones((3,))},
            replay_buffer=replay,
            update_step=jnp.asarray(0, dtype=jnp.int32),
        )
        return _Runtime(
            agent=agent,
            env_state=jnp.zeros((4, 1)),
            observation=jnp.zeros((4, 2)),
            exploration_noise=jnp.zeros((4, 1)),
            noise_steps_remaining=jnp.zeros((4,), dtype=jnp.int32),
            rng=jax.random.key(0),
            validation_rng=jax.random.key(1),
            interaction_step=jnp.asarray(0, dtype=jnp.int32),
            reward_ema=jnp.asarray(0.0),
            discounted_return=jnp.zeros((4,)),
            last_episode_return=jnp.asarray(0.0),
            last_episode_length=jnp.asarray(0.0),
        )

    def update(self, runtime_state, _trajectory_override=None):
        agent = runtime_state.agent.replace(
            update_step=runtime_state.agent.update_step + 1,
            replay_buffer=runtime_state.agent.replay_buffer.replace(
                size=runtime_state.agent.replay_buffer.size + 1,
            ),
        )
        next_state = runtime_state.replace(
            agent=agent,
            reward_ema=runtime_state.reward_ema + 1.0,
        )
        value = agent.update_step.astype(jnp.float32)
        return _Step(
            runtime_state=next_state,
            collection_metrics=FlashSACCollectionMetrics(
                mean_reward=value,
                done_count=jnp.asarray(0),
                absorbing_count=jnp.asarray(0),
                tracking_termination_count=jnp.asarray(0),
                horizon_end_count=jnp.asarray(0),
                trajectory_end_count=jnp.asarray(0),
                numerical_count=jnp.asarray(0),
                site_position_count=jnp.asarray(0),
                episode_return=value,
                episode_length=value,
                reward_components={},
            ),
            update_metrics=FlashSACUpdateMetrics(
                actor_loss=value,
                critic_loss=value,
                temperature_loss=value,
                temperature=value,
                entropy=value,
                q_value=value,
                has_actor_update=jnp.asarray(True),
                has_critic_update=jnp.asarray(True),
            ),
            learning_rate=jnp.asarray(3e-4),
        )

    def update_chunk(
        self,
        runtime_state,
        trajectory_data_override,
        num_interactions,
        max_interactions,
    ):
        def run_interaction(state):
            step = self.update(state, trajectory_data_override)
            metrics = (
                step.collection_metrics,
                step.update_metrics,
                step.runtime_state.reward_ema,
                step.runtime_state.agent.replay_buffer.size,
                step.learning_rate,
            )
            return step.runtime_state, metrics

        runtime_state, first_metrics = run_interaction(runtime_state)
        zero_metrics = jax.tree.map(jnp.zeros_like, first_metrics)

        def scan_interaction(state, index):
            return jax.lax.cond(
                index < num_interactions,
                run_interaction,
                lambda current: (current, zero_metrics),
                state,
            )

        runtime_state, remaining_metrics = jax.lax.scan(
            scan_interaction,
            runtime_state,
            jnp.arange(1, max_interactions),
        )
        metrics = jax.tree.map(
            lambda first, remaining: jnp.concatenate((first[None], remaining)),
            first_metrics,
            remaining_metrics,
        )
        return FlashSACChunk(runtime_state, *metrics)

    def validate(self, _policy_state, validation_rng, *_trajectory_overrides):
        return _ValidationResult(
            metrics=ValidationSummary(
                episode_count=jnp.asarray(1.0),
                mean_episode_return=jnp.asarray(4.0),
            ),
            rng=jax.random.fold_in(validation_rng, 1),
        )

    def progress(self):
        return FlashSACProgress(
            completed_interactions=0,
            remaining_interactions=int(self.config.num_interactions),
            base_global_timestep=0,
            target_global_timestep=int(self.config.total_timesteps),
        )


def _training(
    *,
    save_checkpoints=False,
    interactions=3,
    validation_active=False,
    validation_interval=100,
    num_devices=1,
):
    experiment = OmegaConf.create(
        {
            "distributed": {"num_devices": num_devices},
            "num_envs": 4,
            "global_num_envs": 4 * num_devices,
            "num_interactions": interactions,
            "total_timesteps": interactions * 4,
            "save_checkpoints": save_checkpoints,
            "checkpoint_interval": 2,
            "checkpoints_on_validation": False,
            "validation_interval": validation_interval,
            "validation": {
                "active": validation_active,
                "video_frequency": 0,
                "video_length": 0,
            },
        }
    )
    return _Training(
        env=object(),
        val_env=None,
        agent_conf=SimpleNamespace(config=OmegaConf.create({"experiment": experiment})),
        config=experiment,
        metrics_handler=None,
        resume_info=None,
        preserve_checkpoint_target=False,
        axis_name="data",
        shard_spec=None,
        validation_shard_spec=None,
        validation_shares_training_trajectory=True,
    )


def test_flashsac_host_loop_runs_on_four_data_shards():
    if jax.local_device_count() < 4:
        pytest.skip("requires four visible or simulated JAX devices")
    logs = []
    output = flashsac_runtime.run_flashsac_training(
        _training(interactions=2, num_devices=4),
        jax.random.key(0),
        config=OmegaConf.create({"experiment": {"distributed": {"num_devices": 4}}}),
        logging_callback=logs.append,
    )

    assert int(output["agent_state"].train_state.update_step) == 2
    assert [log["max_timestep"] for log in logs if "max_timestep" in log] == [16, 32]


def test_flashsac_trajectory_requires_explicit_runtime_input():
    training = replace(
        _training(),
        env=SimpleNamespace(th=SimpleNamespace(traj=object())),
    )

    with pytest.raises(ValueError, match="Training trajectory data"):
        flashsac_runtime.run_flashsac_training(
            training,
            jax.random.key(0),
            config=OmegaConf.create({"experiment": {"distributed": {"num_devices": 1}}}),
        )


def test_flashsac_multi_device_donates_runtime_buffers(monkeypatch):
    if jax.local_device_count() < 4:
        pytest.skip("requires four visible or simulated JAX devices")
    compile_options = []
    compile_sharded = flashsac_runtime.compile_sharded

    def record_compile(function, **kwargs):
        compile_options.append(kwargs)
        return compile_sharded(function, **kwargs)

    monkeypatch.setattr(flashsac_runtime, "compile_sharded", record_compile)
    flashsac_runtime.run_flashsac_training(
        _training(
            interactions=1,
            validation_active=True,
            validation_interval=1,
            num_devices=4,
        ),
        jax.random.key(0),
        config=OmegaConf.create({"experiment": {"distributed": {"num_devices": 4}}}),
    )

    assert [options["donate_argnums"] for options in compile_options] == [
        (),
        (0,),
        (1,),
    ]


def test_flashsac_host_loop_runs_pure_updates_and_async_logs():
    training_thread = get_ident()
    log_threads = []
    logs = []

    def log(values):
        log_threads.append(get_ident())
        logs.append(values)

    output = flashsac_runtime.run_flashsac_training(
        _training(),
        jax.random.key(0),
        config=OmegaConf.create({"experiment": {"distributed": {"num_devices": 1}}}),
        logging_callback=log,
    )

    assert int(output["agent_state"].train_state.update_step) == 3
    assert [log["max_timestep"] for log in logs if "max_timestep" in log] == [4, 8, 12]
    assert training_thread not in log_threads[1:]


def test_flashsac_single_device_warp_graphs_are_cleared_after_each_chunk(monkeypatch):
    events = []
    training = replace(_training(), env=SimpleNamespace(mjx_backend="warp"))
    real_block_until_ready = jax.block_until_ready

    monkeypatch.setattr(
        flashsac_runtime,
        "disable_warp_graph_eviction",
        lambda: events.append("disable"),
    )

    def block_until_ready(tree):
        events.append("block")
        return real_block_until_ready(tree)

    def clear_graphs():
        assert events[-1] == "block"
        events.append("clear")

    monkeypatch.setattr(flashsac_runtime.jax, "block_until_ready", block_until_ready)
    monkeypatch.setattr(flashsac_runtime, "clear_warp_graph_cache", clear_graphs)

    flashsac_runtime.run_flashsac_training(
        training,
        jax.random.key(0),
        config=OmegaConf.create({"experiment": {"distributed": {"num_devices": 1}}}),
    )

    assert events[:2] == ["block", "clear"]
    assert events.count("clear") == 1
    assert "disable" not in events


def test_flashsac_checkpoints_run_on_coordinator_thread(monkeypatch):
    events = []
    coordinator_thread = get_ident()

    class Saver:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, _state, *, interactions_done):
            events.append(("save", interactions_done, get_ident()))

        def close(self):
            events.append(("close", None, get_ident()))

    monkeypatch.setattr(flashsac_runtime, "FlashSACCheckpointSaver", Saver)

    flashsac_runtime.run_flashsac_training(
        _training(save_checkpoints=True),
        jax.random.key(0),
        config=OmegaConf.create({"experiment": {"distributed": {"num_devices": 1}}}),
    )

    assert events == [
        ("save", 2, coordinator_thread),
        ("save", 3, coordinator_thread),
        ("close", None, coordinator_thread),
    ]


def test_validation_window_keeps_logging_as_a_soft_boundary():
    config = OmegaConf.create(
        {
            "save_checkpoints": True,
            "checkpoint_interval": 8192,
            "validation_interval": 3906,
            "validation": {"active": True},
        }
    )

    window = flashsac_runtime._chunk_window(config, logging_interval=10)

    assert window == 10
    assert flashsac_runtime._next_chunk_size(0, 20_000, config, window) == 10
    assert flashsac_runtime._next_chunk_size(3906, 20_000, config, window) == 10
    assert flashsac_runtime._next_chunk_size(8188, 20_000, config, window) == 4


def test_chunk_logs_selected_interactions_and_policy_only_validation(monkeypatch):
    seen_policy_states = []
    device_get_threads = []
    coordinator_thread = get_ident()
    real_device_get = jax.device_get

    def validate(self, policy_state, validation_rng, *_trajectory_overrides):
        del self
        seen_policy_states.append(policy_state)
        return _ValidationResult(
            metrics=ValidationSummary(
                episode_count=jnp.asarray(1.0),
                mean_episode_return=jnp.asarray(4.0),
            ),
            rng=jax.random.fold_in(validation_rng, 1),
        )

    def device_get(tree):
        device_get_threads.append(get_ident())
        return real_device_get(tree)

    monkeypatch.setattr(_Training, "validate", validate)
    monkeypatch.setattr(flashsac_runtime.jax, "device_get", device_get)
    logs = []

    output = flashsac_runtime.run_flashsac_training(
        _training(
            interactions=5,
            validation_active=True,
            validation_interval=5,
        ),
        jax.random.key(0),
        config=OmegaConf.create({"experiment": {"distributed": {"num_devices": 1}}}),
        logging_callback=logs.append,
        logging_interval=2,
    )

    assert [log["max_timestep"] for log in logs if "max_timestep" in log] == [8, 16, 20]
    assert logs[-1]["has_validation_update"] is True
    assert logs[-1]["_validation_number"] == 1
    assert float(output["validation_metrics"].mean_episode_return) == 4.0
    assert seen_policy_states
    assert all(isinstance(state, FlashSACPolicyState) for state in seen_policy_states)
    assert device_get_threads
    assert coordinator_thread not in device_get_threads


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_logging_windows_include_unlogged_interactions_and_flush_validation_and_final(monkeypatch, chunk_size):
    original_update = _Training.update

    def update(self, runtime_state, _trajectory_override=None):
        step = original_update(self, runtime_state, _trajectory_override)
        index = step.runtime_state.agent.update_step - 1
        value = index.astype(jnp.float32) + 1.0
        tracking = jnp.asarray([1, 0, 1, 1, 0, 0, 0])[index]
        collection = step.collection_metrics._replace(
            done_count=jnp.asarray([1, 0, 3, 2, 4, 0, 1])[index],
            absorbing_count=jnp.asarray([1, 0, 2, 1, 1, 0, 0])[index],
            tracking_termination_count=tracking,
            horizon_end_count=jnp.asarray([0, 0, 0, 0, 1, 0, 0])[index],
            trajectory_end_count=jnp.asarray([0, 0, 1, 1, 2, 0, 1])[index],
            numerical_count=jnp.asarray([0, 0, 1, 0, 1, 0, 0])[index],
            site_position_count=tracking,
            episode_return=jnp.asarray([10.0, jnp.nan, 20.0, 30.0, 40.0, jnp.nan, 70.0])[index],
            episode_length=jnp.asarray([5.0, jnp.nan, 15.0, 25.0, 35.0, jnp.nan, 65.0])[index],
            reward_components={"reward/qpos": value / 10.0},
        )
        has_actor = value % 2 == 0
        has_critic = value >= 2
        actor_value = jnp.where(has_actor, value, jnp.nan)
        updates = step.update_metrics._replace(
            actor_loss=actor_value,
            critic_loss=jnp.where(has_critic, value, jnp.nan),
            temperature_loss=actor_value,
            entropy=actor_value,
            q_value=actor_value,
            has_actor_update=has_actor,
            has_critic_update=has_critic,
        )
        return step._replace(collection_metrics=collection, update_metrics=updates)

    monkeypatch.setattr(_Training, "update", update)
    training = _training(interactions=7, validation_active=True, validation_interval=3)
    training.config.update_chunk_size = chunk_size
    logs = []
    flashsac_runtime.run_flashsac_training(
        training,
        jax.random.key(0),
        config=OmegaConf.create({"experiment": {"distributed": {"num_devices": 1}}}),
        logging_callback=logs.append,
        logging_interval=5,
    )
    logs = [log for log in logs if "max_timestep" in log]

    assert [log["max_timestep"] for log in logs] == [12, 20, 24, 28]
    assert [log["flashsac/log_window_interactions"] for log in logs] == [3, 2, 1, 1]
    assert [log["episodes/completed"] for log in logs] == [4, 6, 0, 1]
    assert sum(log["episodes/early_terminated"] for log in logs) == 5
    assert sum(log["episodes/tracking_terminated"] for log in logs) == 3
    assert sum(log["episodes/horizon_ended"] for log in logs) == 1
    assert sum(log["episodes/trajectory_ended"] for log in logs) == 5
    assert sum(log["episodes/numerical_terminated"] for log in logs) == 2
    assert sum(log["episodes/site_position_terminated"] for log in logs) == 3
    assert logs[0]["mean_episode_return"] == pytest.approx(17.5)
    assert logs[0]["mean_episode_length"] == pytest.approx(12.5)
    assert logs[0]["episodes/early_termination_rate"] == pytest.approx(3.0 / 4.0)
    assert logs[0]["reward/mean_step"] == pytest.approx(2.0)
    assert logs[0]["reward/qpos"] == pytest.approx(0.2)
    assert logs[0]["flashsac/actor_loss"] == pytest.approx(2.0)
    assert logs[0]["flashsac/critic_loss"] == pytest.approx(2.5)
    assert logs[1]["mean_episode_return"] == pytest.approx(220.0 / 6.0)
    assert logs[1]["reward/mean_step"] == pytest.approx(4.5)
    assert logs[1]["flashsac/actor_loss"] == pytest.approx(4.0)
    assert "mean_episode_return" not in logs[2]
    assert "episodes/early_termination_rate" not in logs[2]
    assert "flashsac/actor_loss" not in logs[3]
    assert logs[3]["mean_episode_return"] == pytest.approx(70.0)


def test_validation_checkpoint_precedes_video_log_handoff(monkeypatch):
    events = []

    class Saver:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, _state, *, interactions_done):
            events.append(("save", interactions_done))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr(flashsac_runtime, "FlashSACCheckpointSaver", Saver)

    def log(payload):
        if payload.get("has_validation_update", False):
            events.append(("validation_log", payload["_checkpoint_update"]))

    flashsac_runtime.run_flashsac_training(
        _training(
            interactions=2,
            save_checkpoints=True,
            validation_active=True,
            validation_interval=2,
        ),
        jax.random.key(0),
        config=OmegaConf.create({"experiment": {"distributed": {"num_devices": 1}}}),
        logging_callback=log,
    )

    assert events == [
        ("save", 2),
        ("validation_log", 2),
        ("close", None),
    ]
