"""Tests for chunked PPO host coordination and pmap reductions."""

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

from musclemimic.algorithms.common.ppo_distributed import PMAP_AXIS_NAME
from musclemimic.algorithms.ppo.evaluation import reduce_validation_summary
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


class _NumericalDiagnostics(NamedTuple):
    numerics_invalid_transition_count: jax.Array


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
            metrics=jnp.asarray(4.0),
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


def test_release_jax_compilation_memory_clears_caches_before_host_gc(monkeypatch):
    events = []
    monkeypatch.setattr(training_runtime.jax, "clear_caches", lambda: events.append("jax"))
    monkeypatch.setattr(
        training_runtime,
        "release_unused_host_memory",
        lambda: events.append("host"),
    )

    training_runtime.release_jax_compilation_memory()

    assert events == ["jax", "host"]


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


def test_next_chunk_stops_at_nonfinite_check_boundary():
    config = OmegaConf.create(
        {
            "validation_interval": 500,
            "checkpoint_interval": 500,
            "nonfinite_guard": {
                "enabled": True,
                "max_updates_between_checks": 100,
            },
        }
    )

    assert training_runtime._next_chunk_size(0, 500, config) == 100
    assert training_runtime._next_chunk_size(175, 500, config) == 25


def test_nonfinite_chunk_saves_pre_chunk_state_and_fails(monkeypatch):
    saved = []

    class _Saver:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, state, *, updates_done):
            saved.append((updates_done, int(state.train_state.step)))
            return f"checkpoint_{updates_done}"

        def close(self):
            saved.append(("closed", "closed"))

    class _NonfiniteTraining(_Training):
        def update_chunk(
            self,
            runtime_state,
            trajectory_data_override,
            num_updates,
            max_updates,
        ):
            result = super().update_chunk(
                runtime_state,
                trajectory_data_override,
                num_updates,
                max_updates,
            )
            return result._replace(
                ppo_metrics=result.ppo_metrics.at[1].set(jnp.nan),
            )

    monkeypatch.setattr(training_runtime, "PPOCheckpointSaver", _Saver)
    training = _NonfiniteTraining(updates=3, save_checkpoints=True)
    training.config.nonfinite_guard = {
        "enabled": True,
        "max_updates_between_checks": 2,
    }

    with pytest.raises(
        FloatingPointError,
        match=r"update=2, metric=ppo<scalar>, last_safe_update=0.*checkpoint_0",
    ):
        training_runtime.run_training(
            training,
            jax.random.PRNGKey(0),
            config=_config(1),
        )

    assert saved == [(0, 0), ("closed", "closed")]


def test_numerical_fault_forces_log_at_exact_update_outside_interval(monkeypatch):
    class _NumericalFaultTraining(_Training):
        def update_chunk(
            self,
            runtime_state,
            trajectory_data_override,
            num_updates,
            max_updates,
        ):
            result = super().update_chunk(
                runtime_state,
                trajectory_data_override,
                num_updates,
                max_updates,
            )
            counts = jnp.zeros((max_updates,), dtype=jnp.float32).at[1].set(1.0)
            return result._replace(
                diagnostics=_NumericalDiagnostics(counts),
            )

    monkeypatch.setattr(
        training_runtime,
        "format_ppo_log",
        lambda *_args, global_timestep, **_kwargs: {
            "step": global_timestep // 32,
            "numerics/nonfinite_transition_count": 1.0,
        },
    )
    logs = []
    training = _NumericalFaultTraining(updates=3)
    training.config.validation_interval = 100

    training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(1),
        logging_callback=logs.append,
        logging_interval=100,
    )

    assert [log["step"] for log in logs if "step" in log] == [2]


def test_multi_seed_training_is_rejected():
    rngs = jnp.stack([jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
    with pytest.raises(NotImplementedError, match="n_seeds"):
        training_runtime.run_training(_Training(), rngs, config=_config(1))


def test_single_device_training_runs_updates_and_validation():
    output = training_runtime.run_training(
        _Training(updates=3),
        jax.random.PRNGKey(0),
        config=_config(1),
    )
    assert int(output["agent_state"].train_state.step) == 6
    assert int(output["agent_state"].train_state.validation_count) == 0
    assert int(output["validation_metrics"]) == 4


def test_single_device_warp_clears_graphs_after_each_quiescent_call(monkeypatch):
    events = []
    training = _Training(updates=3)
    training.env = SimpleNamespace(mjx_backend="warp")
    real_block_until_ready = jax.block_until_ready

    monkeypatch.setattr(
        training_runtime,
        "require_safe_warp_graph_mode",
        lambda _env: pytest.fail("single-device Warp unexpectedly used the multi-device guard"),
    )
    monkeypatch.setattr(
        training_runtime,
        "clear_warp_graph_cache",
        lambda: events.append("clear"),
    )

    def block_until_ready(tree):
        events.append("block")
        return real_block_until_ready(tree)

    monkeypatch.setattr(training_runtime.jax, "block_until_ready", block_until_ready)

    output = training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(1),
    )

    assert int(output["agent_state"].train_state.step) == 6
    assert events == [
        "block",
        "clear",  # initialization
        "block",
        "clear",  # first update chunk
        "block",
        "clear",  # validation
        "block",
        "clear",  # final update chunk
        "block",  # final returned state
    ]


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


def test_completed_multi_device_warp_resume_clears_initialized_graph(monkeypatch):
    _require_devices()
    events = []
    training = _Training(updates=0)
    training.env = SimpleNamespace(mjx_backend="warp")
    real_block_until_ready = jax.block_until_ready

    monkeypatch.setattr(training_runtime, "assert_warp_multi_gpu_compatible", lambda: None)
    monkeypatch.setattr(
        training_runtime,
        "require_safe_warp_graph_mode",
        lambda _env: events.append("validate"),
    )
    monkeypatch.setattr(
        training_runtime,
        "clear_warp_graph_cache",
        lambda: events.append("clear"),
    )

    def block_until_ready(tree):
        events.append("block")
        return real_block_until_ready(tree)

    monkeypatch.setattr(training_runtime.jax, "block_until_ready", block_until_ready)

    output = training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(N_DEVICES),
        resume_agent_state={"step": jnp.asarray(18, dtype=jnp.int32)},
    )

    assert int(output["agent_state"].train_state.step) == 18
    assert events == ["validate", "block", "clear"]


def test_multi_device_training_keeps_one_state_per_replica(monkeypatch):
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
    np.testing.assert_array_equal(np.asarray(output["validation_metrics"]), [4, 4])


@pytest.mark.parametrize(
    ("resume_state", "expected_phases"),
    [
        (None, ["initialize", "post_update"]),
        ({"step": jnp.asarray(4, dtype=jnp.int32)}, ["resume_initialize", "post_resume_update"]),
    ],
)
def test_multi_device_hardware_gate_runs_only_at_host_boundaries(
    monkeypatch,
    resume_state,
    expected_phases,
):
    _require_devices()
    events = []
    real_block_until_ready = jax.block_until_ready

    class _Recorder:
        def record_state(self, _state, *, phase, update_number, global_timestep):
            events.append(("state", phase, update_number, global_timestep))

        def record_post_update_once(self, _state, *, update_number, global_timestep):
            phase = "post_resume_update" if resume_state is not None else "post_update"
            if not any(event[:2] == ("state", phase) for event in events):
                self.record_state(
                    _state,
                    phase=phase,
                    update_number=update_number,
                    global_timestep=global_timestep,
                )

        def record_chunk(self, **fields):
            events.append(("chunk", fields["update_number"], fields["environment_steps"]))

    recorder = _Recorder()

    def block_until_ready(tree):
        events.append(("block",))
        return real_block_until_ready(tree)

    monkeypatch.setattr(
        training_runtime.HardwareGateEvidenceRecorder,
        "from_config",
        lambda *_args, **_kwargs: recorder,
    )
    monkeypatch.setattr(training_runtime, "assert_warp_multi_gpu_compatible", lambda: None)
    monkeypatch.setattr(
        training_runtime.jax,
        "block_until_ready",
        block_until_ready,
    )
    monkeypatch.setattr(
        training_runtime.jax.debug,
        "callback",
        lambda *_args, **_kwargs: pytest.fail("hardware-gate evidence used a host callback inside pmap"),
    )

    training_runtime.run_training(
        _Training(updates=3),
        jax.random.PRNGKey(0),
        config=_config(N_DEVICES),
        resume_agent_state=resume_state,
    )

    assert [event[1] for event in events if event[0] == "state"] == expected_phases
    chunks = [event for event in events if event[0] == "chunk"]
    assert len(chunks) == 2
    assert all(event[2] > 0 for event in chunks)
    assert sum(event[0] == "block" for event in events) == len(chunks) + 1
    chunk_positions = (index for index, event in enumerate(events) if event[0] == "chunk")
    assert all(events[index - 1] == ("block",) for index in chunk_positions)


def test_multi_device_pmap_disables_buffer_donation(monkeypatch):
    _require_devices()
    pmap_options = []
    pmap = jax.pmap

    def record_pmap(function, **kwargs):
        pmap_options.append(kwargs)
        return pmap(function, **kwargs)

    monkeypatch.setattr(training_runtime.jax, "pmap", record_pmap)
    monkeypatch.setattr(
        training_runtime,
        "assert_warp_multi_gpu_compatible",
        lambda: None,
    )

    training_runtime.run_training(
        _Training(updates=1),
        jax.random.PRNGKey(0),
        config=_config(N_DEVICES),
        resume_agent_state={"step": jnp.asarray(0, dtype=jnp.int32)},
    )

    assert len(pmap_options) == 3
    assert all("donate_argnums" not in options for options in pmap_options)


def test_multi_device_warp_clears_graphs_only_after_quiescent_boundaries(monkeypatch):
    _require_devices()
    events = []
    training = _Training(updates=3, save_checkpoints=True)
    training.env = SimpleNamespace(mjx_backend="warp")
    training.val_env = SimpleNamespace(mjx_backend="warp")
    training.config.validation = {"release_compilation_caches": True}
    real_block_until_ready = jax.block_until_ready

    class _Saver:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, _state, *, updates_done):
            events.append(f"save:{updates_done}")

        def close(self):
            events.append("close")

    class _Recorder:
        def record_state(self, *_args, **_kwargs):
            pass

        def record_chunk(self, **_kwargs):
            pass

        def record_post_update_once(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(training_runtime, "PPOCheckpointSaver", _Saver)
    monkeypatch.setattr(
        training_runtime.HardwareGateEvidenceRecorder,
        "from_config",
        lambda *_args, **_kwargs: _Recorder(),
    )
    monkeypatch.setattr(
        training_runtime,
        "assert_warp_multi_gpu_compatible",
        lambda: None,
    )
    monkeypatch.setattr(
        training_runtime,
        "require_safe_warp_graph_mode",
        lambda _env: events.append("validate"),
    )
    monkeypatch.setattr(
        training_runtime,
        "clear_warp_graph_cache",
        lambda: events.append("clear"),
    )
    monkeypatch.setattr(
        training_runtime,
        "release_jax_compilation_memory",
        lambda: events.append("release-jax"),
    )

    def block_until_ready(tree):
        events.append("block")
        return real_block_until_ready(tree)

    monkeypatch.setattr(training_runtime.jax, "block_until_ready", block_until_ready)

    training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(N_DEVICES),
    )

    assert events == [
        "validate",
        "validate",
        "block",
        "clear",
        "block",
        "clear",
        "release-jax",
        "block",
        "clear",
        "release-jax",
        "save:2",
        "block",
        "clear",
        "save:3",
        "block",
        "close",
    ]


def test_multi_device_warp_keeps_jax_caches_by_default(monkeypatch):
    _require_devices()
    training = _Training(updates=2)
    training.env = SimpleNamespace(mjx_backend="warp")
    training.val_env = SimpleNamespace(mjx_backend="warp")

    monkeypatch.setattr(training_runtime, "assert_warp_multi_gpu_compatible", lambda: None)
    monkeypatch.setattr(training_runtime, "require_safe_warp_graph_mode", lambda _env: None)
    monkeypatch.setattr(training_runtime, "clear_warp_graph_cache", lambda: None)
    monkeypatch.setattr(
        training_runtime,
        "release_jax_compilation_memory",
        lambda: pytest.fail("default validation unexpectedly released JAX caches"),
    )

    training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(N_DEVICES),
    )


@pytest.mark.parametrize(
    ("training_backend", "validation_backend", "expected_events"),
    [
        (
            "warp",
            "jax",
            [
                "guard:training",
                "block",
                "clear",
                "block",
                "clear",
                "release-jax",
                "block",
                "release-jax",
                "block",
            ],
        ),
        (
            "jax",
            "warp",
            [
                "guard:validation",
                "block",
                "release-jax",
                "block",
                "clear",
                "release-jax",
                "block",
            ],
        ),
    ],
)
def test_multi_device_mixed_backends_guard_only_warp_calls(
    monkeypatch,
    training_backend,
    validation_backend,
    expected_events,
):
    _require_devices()
    events = []
    training = _Training(updates=2)
    training.env = SimpleNamespace(mjx_backend=training_backend)
    training.val_env = SimpleNamespace(mjx_backend=validation_backend)
    training.config.validation = {"release_compilation_caches": True}
    real_block_until_ready = jax.block_until_ready

    monkeypatch.setattr(training_runtime, "assert_warp_multi_gpu_compatible", lambda: None)
    monkeypatch.setattr(
        training_runtime,
        "require_safe_warp_graph_mode",
        lambda env: events.append("guard:training" if env is training.env else "guard:validation"),
    )
    monkeypatch.setattr(
        training_runtime,
        "clear_warp_graph_cache",
        lambda: events.append("clear"),
    )
    monkeypatch.setattr(
        training_runtime,
        "release_jax_compilation_memory",
        lambda: events.append("release-jax"),
    )

    def block_until_ready(tree):
        events.append("block")
        return real_block_until_ready(tree)

    monkeypatch.setattr(training_runtime.jax, "block_until_ready", block_until_ready)

    training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(N_DEVICES),
    )

    assert events == expected_events


def test_multi_device_warp_resolves_pending_log_before_next_dispatch(monkeypatch):
    _require_devices()
    events = []
    training = _Training(updates=2)
    training.env = SimpleNamespace(mjx_backend="warp")
    training.config.validation_interval = 100
    training.config.checkpoint_interval = 1
    training.config.checkpoints_on_validation = False
    real_pmap = jax.pmap

    class _FailedLog:
        def result(self):
            events.append("log:result")
            raise RuntimeError("host log failed")

    class _LogWorker:
        def __init__(self, *_args, **_kwargs):
            pass

        def submit(self, *_args, **_kwargs):
            events.append("log:submit")
            return _FailedLog()

        def shutdown(self, *, wait):
            assert wait is True
            events.append("log:shutdown")

    def record_pmap(function, **kwargs):
        compiled = real_pmap(function, **kwargs)
        if function.__name__ != "update_chunk":
            return compiled

        def dispatch(*args):
            events.append("update:dispatch")
            return compiled(*args)

        return dispatch

    monkeypatch.setattr(training_runtime, "ThreadPoolExecutor", _LogWorker)
    monkeypatch.setattr(training_runtime.jax, "pmap", record_pmap)
    monkeypatch.setattr(training_runtime, "assert_warp_multi_gpu_compatible", lambda: None)
    monkeypatch.setattr(training_runtime, "require_safe_warp_graph_mode", lambda _env: None)
    monkeypatch.setattr(training_runtime, "clear_warp_graph_cache", lambda: None)

    with pytest.raises(RuntimeError, match="host log failed"):
        training_runtime.run_training(
            training,
            jax.random.PRNGKey(0),
            config=_config(N_DEVICES),
            logging_callback=lambda _payload: None,
        )

    assert events == [
        "update:dispatch",
        "log:submit",
        "log:result",
        "log:shutdown",
    ]


def test_multi_device_warp_does_not_clear_after_failed_device_block(monkeypatch):
    _require_devices()
    events = []
    training = _Training(updates=1)
    training.env = SimpleNamespace(mjx_backend="warp")
    real_block_until_ready = jax.block_until_ready

    monkeypatch.setattr(training_runtime, "assert_warp_multi_gpu_compatible", lambda: None)
    monkeypatch.setattr(training_runtime, "require_safe_warp_graph_mode", lambda _env: None)
    monkeypatch.setattr(
        training_runtime,
        "clear_warp_graph_cache",
        lambda: events.append("clear"),
    )

    def block_until_ready(tree):
        events.append("block")
        if events.count("block") == 2:
            raise RuntimeError("device work failed")
        return real_block_until_ready(tree)

    monkeypatch.setattr(training_runtime.jax, "block_until_ready", block_until_ready)

    with pytest.raises(RuntimeError, match="device work failed"):
        training_runtime.run_training(
            training,
            jax.random.PRNGKey(0),
            config=_config(N_DEVICES),
        )

    assert events == ["block", "clear", "block"]


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


def test_checkpoint_boundary_stop_finishes_save_and_log_before_return(monkeypatch, caplog):
    events = []

    class _Saver:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, _state, *, updates_done):
            events.append(f"checkpoint:{updates_done}")

        def close(self):
            events.append("close")

    monkeypatch.setattr(training_runtime, "PPOCheckpointSaver", _Saver)
    monkeypatch.setattr(
        training_runtime.HardwareGateEvidenceRecorder,
        "from_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        training_runtime,
        "format_ppo_log",
        lambda *_args, global_timestep, **_kwargs: {"step": global_timestep // 32},
    )
    training = _Training(updates=6, save_checkpoints=True)
    training.config.validation_interval = 100
    training.config.checkpoint_interval = 2
    training.config.checkpoints_on_validation = False
    training.config.async_checkpointing = False
    training.config.debug = {
        "hardware_gate": {
            "enabled": True,
            "stop_after_checkpoint_update": 2,
        }
    }

    output = training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(1),
        logging_callback=lambda payload: events.append(f"log:{payload['step']}") if "step" in payload else None,
    )

    assert int(output["agent_state"].train_state.step) == 4
    assert events == ["log:1", "checkpoint:2", "log:2", "close"]
    assert "checkpoint finalized, logs flushed, no later update dispatched" in caplog.text


@pytest.mark.parametrize(
    ("value", "config_updates", "completed_updates", "remaining_updates", "message"),
    [
        ("zero", {}, 0, 6, "must be a positive integer"),
        (2, {"debug": {"hardware_gate": {"enabled": False}}}, 0, 6, "requires hardware_gate.enabled"),
        (2, {"save_checkpoints": False}, 0, 6, "requires save_checkpoints=true"),
        (2, {"async_checkpointing": True}, 0, 6, "requires async_checkpointing=false"),
        (3, {}, 0, 6, "must be a positive multiple"),
        (6, {}, 0, 6, "must precede final update 6"),
        (8, {}, 0, 6, "must precede final update 6"),
    ],
)
def test_checkpoint_boundary_stop_rejects_unsafe_requests(
    value,
    config_updates,
    completed_updates,
    remaining_updates,
    message,
):
    config = OmegaConf.merge(
        OmegaConf.create(
            {
                "save_checkpoints": True,
                "async_checkpointing": False,
                "checkpoint_interval": 2,
                "debug": {
                    "hardware_gate": {
                        "enabled": True,
                        "stop_after_checkpoint_update": value,
                    }
                },
            }
        ),
        config_updates,
    )

    with pytest.raises(ValueError, match=message):
        training_runtime.checkpoint_boundary_stop_update_from_config(
            config,
            completed_updates=completed_updates,
            remaining_updates=remaining_updates,
        )


def test_checkpoint_boundary_is_satisfied_after_resume():
    config = OmegaConf.create(
        {
            "save_checkpoints": True,
            "async_checkpointing": False,
            "checkpoint_interval": 2,
            "debug": {
                "hardware_gate": {
                    "enabled": True,
                    "stop_after_checkpoint_update": 2,
                }
            },
        }
    )

    assert (
        training_runtime.checkpoint_boundary_stop_update_from_config(
            config,
            completed_updates=2,
            remaining_updates=4,
        )
        is None
    )


def test_resumed_training_continues_beyond_satisfied_checkpoint_boundary(monkeypatch):
    class _Saver:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, _state, *, updates_done):
            assert updates_done == 4

        def close(self):
            pass

    monkeypatch.setattr(training_runtime, "PPOCheckpointSaver", _Saver)
    monkeypatch.setattr(
        training_runtime.HardwareGateEvidenceRecorder,
        "from_config",
        lambda *_args, **_kwargs: None,
    )
    training = _Training(updates=2, save_checkpoints=True)
    training.config.validation_interval = 100
    training.config.checkpoint_interval = 2
    training.config.checkpoints_on_validation = False
    training.config.async_checkpointing = False
    training.config.debug = {
        "hardware_gate": {
            "enabled": True,
            "stop_after_checkpoint_update": 2,
        }
    }

    output = training_runtime.run_training(
        training,
        jax.random.PRNGKey(0),
        config=_config(1),
        resume_agent_state={"step": jnp.asarray(4, dtype=jnp.int32)},
    )

    assert int(output["agent_state"].train_state.step) == 8


def test_regular_logging_overlaps_the_following_update(monkeypatch):
    events = []
    first_log_started = threading.Event()
    release_first_log = threading.Event()
    monkeypatch.setattr(
        training_runtime.jax,
        "jit",
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


def test_training_prefetches_metrics_and_only_blocks_before_return(monkeypatch):
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
    assert len(block_calls) == 1


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


def test_disabled_video_does_not_force_a_checkpoint():
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
    assert training_runtime._checkpoint_due(6, config) is False


def test_exact_resume_runtime_is_sharded_without_extra_device_axis():
    _require_devices()
    devices = jax.local_devices()[:N_DEVICES]
    runtime_state = {
        "last_obs": np.arange(N_DEVICES * 6, dtype=np.float32).reshape(
            N_DEVICES,
            2,
            3,
        )
    }
    resume_state = {
        "train_state": {"weights": np.arange(4, dtype=np.float32)},
        "runtime_state": runtime_state,
    }

    placed = training_runtime._place_resume_state(resume_state, devices)

    np.testing.assert_array_equal(
        np.asarray(placed["runtime_state"]["last_obs"]),
        runtime_state["last_obs"],
    )
    replicated_weights = np.asarray(placed["train_state"]["weights"])
    assert replicated_weights.shape == (N_DEVICES, 4)


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

    output = jax.pmap(
        lambda summary: reduce_validation_summary(
            summary,
            axis_name=PMAP_AXIS_NAME,
        ),
        axis_name=PMAP_AXIS_NAME,
    )(_stack_summaries(*per_device))

    np.testing.assert_allclose(np.asarray(output.episode_count), [10.0, 10.0])
    np.testing.assert_allclose(
        np.asarray(output.mean_episode_return),
        [90.0, 90.0],
    )
    np.testing.assert_allclose(np.asarray(output.mean_episode_length), [19.0, 19.0])
    np.testing.assert_allclose(
        np.asarray(output.early_termination_count),
        [4.0, 4.0],
    )
    np.testing.assert_allclose(
        np.asarray(output.early_termination_rate),
        [0.4, 0.4],
    )
    np.testing.assert_allclose(
        np.asarray(output.err_root_xyz),
        [np.sqrt(5.0), np.sqrt(5.0)],
    )
    np.testing.assert_array_equal(np.asarray(output.max_timestep), [8, 8])


def test_validation_summary_all_reduce_zero_episodes():
    _require_devices()
    zeros = _stack_summaries(ValidationSummary(), ValidationSummary())
    output = jax.pmap(
        lambda summary: reduce_validation_summary(
            summary,
            axis_name=PMAP_AXIS_NAME,
        ),
        axis_name=PMAP_AXIS_NAME,
    )(zeros)
    np.testing.assert_allclose(np.asarray(output.mean_episode_return), [0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(output.early_termination_rate),
        [0.0, 0.0],
    )
