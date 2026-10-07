"""Tests for chunked PPO host coordination and data-parallel reductions."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace
from typing import NamedTuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.algorithms.common.distributed import DATA_AXIS_NAME, make_data_mesh
from musclemimic.algorithms.common.validation import reduce_validation_summary
from musclemimic.algorithms.ppo.train import PPOTrainChunk, TrainingProgress
from musclemimic.runner import training_runtime
from musclemimic.utils.metrics import ValidationSummary

N_DEVICES = 2


def _config(n_devices: int):
    return OmegaConf.create({"experiment": {"distributed": {"num_devices": n_devices}}})


def _require_devices():
    if jax.device_count() < N_DEVICES:
        pytest.skip(f"needs >= {N_DEVICES} devices")


@struct.dataclass
class _TrainState:
    params: dict
    step: jax.Array
    validation_count: jax.Array


@struct.dataclass
class _RuntimeState:
    train_state: _TrainState
    val_rng: jax.Array


class _UpdateResult(NamedTuple):
    runtime_state: _RuntimeState
    training_metrics: jax.Array
    ppo_metrics: jax.Array
    diagnostics: jax.Array
    validation_reset_weights: None
    counter: jax.Array


class _ValidationResult(NamedTuple):
    metrics: jax.Array
    rng: jax.Array


@struct.dataclass
class _AgentState:
    train_state: _TrainState


class _Training:
    validation_shard_spec = None
    shard_spec = None
    validation_shares_training_trajectory = False
    env = object()
    agent_conf = object()
    agent_state_cls = _AgentState

    def __init__(self, *, updates: int = 3, save_checkpoints: bool = False):
        self.updates = updates
        self.config = OmegaConf.create(
            {
                "validation_interval": 2,
                "save_checkpoints": save_checkpoints,
                "checkpoint_interval": 0,
                "checkpoints_on_validation": True,
                "num_minibatches": 2,
                "update_epochs": 1,
                "num_steps": 4,
                "num_envs": 8,
                "adaptive_sampling": {"enabled": False},
            }
        )

    def initialize(
        self,
        _rng,
        resume_state=None,
        trajectory_data_override=None,
    ):
        del trajectory_data_override
        initial_step = jnp.asarray(0, dtype=jnp.int32) if resume_state is None else resume_state["step"]
        return _RuntimeState(
            _TrainState(
                params={"actor_weight": jnp.ones((2,))},
                step=initial_step,
                validation_count=jnp.asarray(0, dtype=jnp.int32),
            ),
            val_rng=jnp.asarray(0, dtype=jnp.int32),
        )

    def update(self, runtime_state, _trajectory_data_override=None):
        train_state = runtime_state.train_state.replace(step=runtime_state.train_state.step + 2)
        return _UpdateResult(
            runtime_state=_RuntimeState(train_state, runtime_state.val_rng),
            training_metrics=jnp.asarray(1.0),
            ppo_metrics=jnp.asarray(2.0),
            diagnostics=jnp.asarray(3.0),
            validation_reset_weights=None,
            counter=train_state.step // 2,
        )

    def update_chunk(
        self,
        runtime_state,
        trajectory_data_override,
        num_updates,
        max_updates,
    ):
        del trajectory_data_override
        train_state = runtime_state.train_state.replace(
            step=runtime_state.train_state.step + 2 * num_updates,
        )
        runtime_state = _RuntimeState(train_state, runtime_state.val_rng)
        return PPOTrainChunk(
            runtime_state=runtime_state,
            training_metrics=jnp.ones((max_updates,)),
            ppo_metrics=jnp.full((max_updates,), 2.0),
            diagnostics=jnp.full((max_updates,), 3.0),
            validation_reset_weights=None,
        )

    def validate(self, _train_state, validation_rng, _weights, *_shards):
        return _ValidationResult(
            metrics=ValidationSummary(
                episode_count=jnp.asarray(1.0),
                mean_episode_return=jnp.asarray(4.0),
            ),
            rng=validation_rng + 1,
        )

    def progress(self, runtime_state, *, resumed):
        del resumed
        step = int(runtime_state.train_state.step)
        completed_updates = step // 2
        return TrainingProgress(
            completed_updates=completed_updates,
            remaining_updates=self.updates,
            base_global_timestep=completed_updates * 32,
            base_optimizer_step=step,
            target_global_timestep=(completed_updates + self.updates) * 32,
        )

    @property
    def base_learning_rate(self):
        return 3e-4


class _ObservedTraining(_Training):
    def __init__(self, events, **kwargs):
        super().__init__(**kwargs)
        self._events = events
        self._update = 0

    def update(self, runtime_state, *_shards):
        self._update += 1
        self._events.append(f"update:{self._update}")
        return super().update(runtime_state, *_shards)

    def update_chunk(
        self,
        runtime_state,
        trajectory_data_override,
        num_updates,
        max_updates,
    ):
        if isinstance(runtime_state.train_state.step, jax.core.Tracer):
            return super().update_chunk(
                runtime_state,
                trajectory_data_override,
                num_updates,
                max_updates,
            )
        steps = []
        for _ in range(int(num_updates)):
            step = self.update(runtime_state, trajectory_data_override)
            runtime_state = step.runtime_state
            steps.append(step)
        padding = max_updates - len(steps)
        return PPOTrainChunk(
            runtime_state=runtime_state,
            training_metrics=jnp.pad(
                jnp.stack([step.training_metrics for step in steps]),
                (0, padding),
            ),
            ppo_metrics=jnp.pad(
                jnp.stack([step.ppo_metrics for step in steps]),
                (0, padding),
            ),
            diagnostics=jnp.pad(
                jnp.stack([step.diagnostics for step in steps]),
                (0, padding),
            ),
            validation_reset_weights=None,
        )


def test_release_unused_host_memory_ignores_missing_malloc_trim(monkeypatch):
    monkeypatch.setattr(training_runtime.ctypes, "CDLL", lambda _name: object())
    training_runtime.release_unused_host_memory()


def test_training_rngs_validate_explicit_seed_count():
    config = OmegaConf.create({"experiment": {"n_seeds": 2, "seeds": [7]}})
    with pytest.raises(ValueError, match="must match n_seeds"):
        training_runtime.compute_training_rngs(config)


@pytest.mark.parametrize(
    ("counter", "remaining", "checkpoint_interval", "expected"),
    [(0, 150, 0, 75), (75, 100, 76, 1), (76, 100, 76, 74), (80, 10, 0, 10)],
)
def test_next_chunk_stops_at_resume_or_final_boundary(
    counter,
    remaining,
    checkpoint_interval,
    expected,
):
    config = OmegaConf.create(
        {
            "validation_interval": 75,
            "checkpoint_interval": checkpoint_interval,
        }
    )

    assert training_runtime._next_chunk_size(counter, remaining, config) == expected


def test_multi_seed_training_is_rejected():
    rngs = jnp.stack([jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
    with pytest.raises(NotImplementedError, match="n_seeds"):
        training_runtime.run_training(_Training(), rngs, config=_config(1))


def test_training_trajectory_requires_explicit_runtime_input():
    training = _Training()
    training.env = SimpleNamespace(th=SimpleNamespace(traj=object()))

    with pytest.raises(ValueError, match="Training trajectory data"):
        training_runtime.run_training(
            training,
            jax.random.PRNGKey(0),
            config=_config(1),
        )


def test_distinct_validation_trajectory_requires_explicit_runtime_input():
    training = _Training()
    training.val_env = SimpleNamespace(th=SimpleNamespace(traj=object()))

    with pytest.raises(ValueError, match="Validation trajectory data"):
        training_runtime.run_training(
            training,
            jax.random.PRNGKey(0),
            config=_config(1),
        )


def test_single_device_training_runs_updates_and_validation():
    output = training_runtime.run_training(
        _Training(updates=3),
        jax.random.PRNGKey(0),
        config=_config(1),
    )
    assert int(output["agent_state"].train_state.step) == 6
    assert int(output["agent_state"].train_state.validation_count) == 0
    assert int(output["validation_metrics"].mean_episode_return) == 4


def test_completed_exact_resume_logs_restored_global_timestep():
    logs = []

    training_runtime.run_training(
        _Training(updates=0),
        jax.random.PRNGKey(0),
        config=_config(1),
        resume_agent_state={"step": jnp.asarray(18, dtype=jnp.int32)},
        logging_callback=logs.append,
    )

    assert logs[-1]["max_timestep"] == 9 * 4 * 8


def test_multi_device_training_returns_one_global_state(monkeypatch):
    _require_devices()
    monkeypatch.setattr(
        training_runtime,
        "assert_warp_multi_gpu_compatible",
        lambda: None,
    )
    output = training_runtime.run_training(
        _Training(updates=3),
        jax.random.PRNGKey(0),
        config=_config(N_DEVICES),
    )
    assert int(output["agent_state"].train_state.step) == 6
    assert int(output["agent_state"].train_state.validation_count) == 0
    assert int(output["validation_metrics"].mean_episode_return) == 4


def test_main_model_checkpoint_resumes_on_multiple_devices(monkeypatch):
    _require_devices()
    monkeypatch.setattr(
        training_runtime,
        "assert_warp_multi_gpu_compatible",
        lambda: None,
    )

    output = training_runtime.run_training(
        _Training(updates=1),
        jax.random.PRNGKey(0),
        config=_config(N_DEVICES),
        resume_agent_state={"step": jnp.asarray(18, dtype=jnp.int32)},
    )

    assert int(output["agent_state"].train_state.step) == 20
    assert output["agent_state"].train_state.params["actor_weight"].shape == (2,)


@pytest.mark.parametrize("n_devices", [1, N_DEVICES])
def test_shard_map_donates_runtime_buffers(monkeypatch, n_devices):
    _require_devices()
    compile_options = []
    compile_sharded = training_runtime.compile_sharded

    def record_compile(function, **kwargs):
        compile_options.append(kwargs)
        return compile_sharded(function, **kwargs)

    monkeypatch.setattr(training_runtime, "compile_sharded", record_compile)
    monkeypatch.setattr(
        training_runtime,
        "assert_warp_multi_gpu_compatible",
        lambda: None,
    )

    training_runtime.run_training(
        _Training(updates=1),
        jax.random.PRNGKey(0),
        config=_config(n_devices),
        resume_agent_state={"step": jnp.asarray(0, dtype=jnp.int32)},
    )

    assert len(compile_options) == 3
    assert [options["donate_argnums"] for options in compile_options] == [
        (1,),
        (0,),
        (1,),
    ]


@pytest.mark.parametrize("n_devices", [1, N_DEVICES])
def test_warp_graphs_are_cleared_after_each_chunk(monkeypatch, n_devices):
    if n_devices > 1:
        _require_devices()
    events = []
    training = _Training(updates=3)
    training.env = SimpleNamespace(mjx_backend="warp")
    real_block_until_ready = jax.block_until_ready

    monkeypatch.setattr(
        training_runtime,
        "assert_warp_multi_gpu_compatible",
        lambda: None,
    )
    monkeypatch.setattr(
        training_runtime,
        "disable_warp_graph_eviction",
        lambda: events.append("disable"),
    )

    def block_until_ready(tree):
        events.append("block")
        return real_block_until_ready(tree)

    def clear_graphs():
        assert events[-1] == "block"
        events.append("clear")

    monkeypatch.setattr(training_runtime.jax, "block_until_ready", block_until_ready)
    monkeypatch.setattr(training_runtime, "clear_warp_graph_cache", clear_graphs)

    training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(n_devices),
    )

    expected_prefix = ["block", "block", "clear", "block", "clear"]
    if n_devices > 1:
        expected_prefix.insert(0, "disable")
    assert events[: len(expected_prefix)] == expected_prefix
    assert events.count("clear") == 2
    assert events.count("disable") == int(n_devices > 1)


def test_checkpoint_precedes_logging_on_host(monkeypatch):
    events = []
    training_thread = threading.get_ident()
    checkpoint_threads = []
    logging_threads = []

    class _Saver:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, _state, *, updates_done):
            checkpoint_threads.append(threading.get_ident())
            events.append(f"checkpoint:{updates_done}")

        def close(self):
            events.append("close")

    monkeypatch.setattr(training_runtime, "PPOCheckpointSaver", _Saver)
    monkeypatch.setattr(
        training_runtime,
        "format_ppo_log",
        lambda *_args, global_timestep, **_kwargs: {"step": global_timestep // 32},
    )

    def log(payload):
        if "step" in payload:
            logging_threads.append(threading.get_ident())
            events.append(f"log:{payload['step']}")

    training_runtime.run_training(
        _Training(updates=3, save_checkpoints=True),
        jax.random.PRNGKey(0),
        config=_config(1),
        logging_callback=log,
    )

    assert events == [
        "log:1",
        "checkpoint:2",
        "log:2",
        "checkpoint:3",
        "log:3",
        "close",
    ]
    assert set(checkpoint_threads) == {training_thread}
    assert logging_threads
    assert training_thread not in logging_threads
    assert len(set(logging_threads)) == 1


def test_regular_logging_overlaps_the_following_update(monkeypatch):
    events = []
    first_log_started = threading.Event()
    release_first_log = threading.Event()
    monkeypatch.setattr(
        training_runtime,
        "compile_sharded",
        lambda function, **_kwargs: function,
    )
    monkeypatch.setattr(
        training_runtime,
        "format_ppo_log",
        lambda *_args, global_timestep, **_kwargs: {"step": global_timestep // 32},
    )

    def log(payload):
        if "step" in payload:
            if payload["step"] == 1:
                events.append("log:1:start")
                first_log_started.set()
                assert release_first_log.wait(timeout=5.0)
            events.append(f"log:{payload['step']}")

    training_thread = threading.Thread(
        target=training_runtime.run_training,
        args=(_ObservedTraining(events, updates=3), jax.random.PRNGKey(0)),
        kwargs={
            "config": _config(1),
            "logging_callback": log,
            "logging_interval": 1,
        },
    )
    training_thread.start()

    assert first_log_started.wait(timeout=5.0)
    deadline = time.monotonic() + 5.0
    while "update:2" not in events and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "update:2" in events
    release_first_log.set()
    training_thread.join(timeout=5.0)
    assert not training_thread.is_alive()

    assert set(events) == {
        "update:1",
        "update:2",
        "update:3",
        "log:1:start",
        "log:1",
        "log:2",
        "log:3",
    }
    assert events.index("update:1") < events.index("log:1:start")
    assert events.index("update:2") < events.index("log:1")
    assert events.index("log:1") < events.index("log:2")
    assert events.index("log:2") < events.index("update:3")
    assert events.index("update:3") < events.index("log:3")


def test_training_prefetches_metrics_and_blocks_at_validation_boundary(monkeypatch):
    copy_calls = []
    block_calls = []
    real_copy_to_host_async = jax.copy_to_host_async
    real_block_until_ready = jax.block_until_ready

    def copy_to_host_async(tree):
        copy_calls.append(tree)
        return real_copy_to_host_async(tree)

    def block_until_ready(tree):
        block_calls.append(tree)
        return real_block_until_ready(tree)

    monkeypatch.setattr(
        training_runtime.jax,
        "copy_to_host_async",
        copy_to_host_async,
    )
    monkeypatch.setattr(
        training_runtime.jax,
        "block_until_ready",
        block_until_ready,
    )
    monkeypatch.setattr(
        training_runtime,
        "format_ppo_log",
        lambda *_args, global_timestep, **_kwargs: {"step": global_timestep // 32},
    )

    training_runtime.run_training(
        _Training(updates=3),
        jax.random.PRNGKey(0),
        config=_config(1),
        logging_callback=lambda _payload: None,
        logging_interval=1,
    )

    assert len(copy_calls) == 3
    assert len(block_calls) == 2


def test_multi_device_checkpoints_and_slow_logging_do_not_deadlock(monkeypatch):
    _require_devices()
    host_lock = threading.Lock()
    saved_updates = []
    logged_steps = []

    class _Saver:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, _state, *, updates_done):
            with host_lock:
                time.sleep(0.01)
                saved_updates.append(updates_done)

        def close(self):
            pass

    monkeypatch.setattr(training_runtime, "PPOCheckpointSaver", _Saver)
    monkeypatch.setattr(
        training_runtime,
        "assert_warp_multi_gpu_compatible",
        lambda: None,
    )
    monkeypatch.setattr(
        training_runtime,
        "format_ppo_log",
        lambda *_args, global_timestep, **_kwargs: {"step": global_timestep // 32},
    )

    def log(payload):
        if "step" not in payload:
            return
        with host_lock:
            time.sleep(0.01)
            logged_steps.append(payload["step"])

    training = _Training(updates=6, save_checkpoints=True)
    training.config.validation_interval = 100
    training.config.checkpoint_interval = 2
    training.config.checkpoints_on_validation = False

    output = training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(N_DEVICES),
        logging_callback=log,
    )

    assert int(output["agent_state"].train_state.step) == 12
    assert saved_updates == [2, 4, 6]
    assert logged_steps == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize(
    ("counter", "expected"),
    [(4, False), (5, True), (6, True), (10, True)],
)
def test_checkpoint_due_unifies_interval_and_validation(counter, expected):
    config = OmegaConf.create(
        {
            "checkpoint_interval": 5,
            "checkpoints_on_validation": True,
            "validation_interval": 3,
        }
    )

    assert training_runtime._checkpoint_due(counter, config) is expected


def test_video_validation_checkpoint_is_saved_only_at_video_frequency():
    config = OmegaConf.create(
        {
            "checkpoint_interval": 0,
            "checkpoints_on_validation": False,
            "validation_interval": 3,
            "validation": {
                "active": True,
                "video_frequency": 2,
                "video_length": 100,
            },
        }
    )

    assert training_runtime._checkpoint_due(3, config) is False
    assert training_runtime._checkpoint_due(6, config) is True


def test_exact_resume_requires_global_runtime_layout():
    _require_devices()
    resume_state = {
        "runtime_state": {
            "last_obs": np.zeros((4, 3), dtype=np.float32),
            "rng": np.zeros((N_DEVICES, 2), dtype=np.uint32),
        }
    }
    training_runtime._validate_exact_resume_layout(
        resume_state,
        global_num_envs=4,
        num_devices=N_DEVICES,
    )

    resume_state["runtime_state"]["last_obs"] = np.zeros(
        (N_DEVICES, 2, 3),
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="global environment batch"):
        training_runtime._validate_exact_resume_layout(
            resume_state,
            global_num_envs=4,
            num_devices=N_DEVICES,
        )


def _stack_summaries(*summaries):
    return jax.tree.map(
        lambda *values: jnp.stack([jnp.asarray(value) for value in values]),
        *summaries,
    )


def test_validation_summary_all_reduce_semantics():
    _require_devices()
    per_device = [
        ValidationSummary(
            episode_count=jnp.float32(1.0),
            mean_episode_return=jnp.float32(0.0),
            mean_episode_length=jnp.float32(10.0),
            early_termination_count=jnp.float32(1.0),
            early_termination_rate=jnp.float32(1.0),
            err_root_xyz=jnp.float32(1.0),
            max_timestep=jnp.int32(7),
        ),
        ValidationSummary(
            episode_count=jnp.float32(9.0),
            mean_episode_return=jnp.float32(100.0),
            mean_episode_length=jnp.float32(20.0),
            early_termination_count=jnp.float32(3.0),
            early_termination_rate=jnp.float32(3.0 / 9.0),
            err_root_xyz=jnp.float32(3.0),
            max_timestep=jnp.int32(8),
        ),
    ]

    mesh = make_data_mesh(N_DEVICES)
    summaries = jax.device_put(
        _stack_summaries(*per_device),
        jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec(DATA_AXIS_NAME),
        ),
    )
    output = jax.jit(
        jax.shard_map(
            lambda summary: reduce_validation_summary(
                summary,
                axis_name=DATA_AXIS_NAME,
            ),
            mesh=mesh,
            in_specs=jax.sharding.PartitionSpec(DATA_AXIS_NAME),
            out_specs=jax.sharding.PartitionSpec(),
        )
    )(summaries)

    np.testing.assert_allclose(np.asarray(output.episode_count), 10.0)
    np.testing.assert_allclose(np.asarray(output.mean_episode_return), 90.0)
    np.testing.assert_allclose(np.asarray(output.mean_episode_length), 19.0)
    np.testing.assert_allclose(np.asarray(output.early_termination_count), 4.0)
    np.testing.assert_allclose(np.asarray(output.early_termination_rate), 0.4)
    np.testing.assert_allclose(np.asarray(output.err_root_xyz), np.sqrt(5.0))
    np.testing.assert_array_equal(np.asarray(output.max_timestep), 8)


def test_validation_summary_all_reduce_zero_episodes():
    _require_devices()
    zeros = _stack_summaries(ValidationSummary(), ValidationSummary())
    mesh = make_data_mesh(N_DEVICES)
    zeros = jax.device_put(
        zeros,
        jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec(DATA_AXIS_NAME),
        ),
    )
    output = jax.jit(
        jax.shard_map(
            lambda summary: reduce_validation_summary(
                summary,
                axis_name=DATA_AXIS_NAME,
            ),
            mesh=mesh,
            in_specs=jax.sharding.PartitionSpec(DATA_AXIS_NAME),
            out_specs=jax.sharding.PartitionSpec(),
        )
    )(zeros)
    np.testing.assert_allclose(np.asarray(output.mean_episode_return), 0.0)
    np.testing.assert_allclose(np.asarray(output.early_termination_rate), 0.0)
