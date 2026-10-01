"""Forced-CPU tests for bounded PPO hardware-gate evidence."""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import pytest
from flax import struct
from omegaconf import OmegaConf

from musclemimic.runner.hardware_gate import HardwareGateEvidenceRecorder


@struct.dataclass
class _TrainState:
    params: dict


@struct.dataclass
class _TrajectoryState:
    traj_no: jax.Array


@struct.dataclass
class _Carry:
    traj_state: _TrajectoryState
    terrain_state: dict
    key: jax.Array


@struct.dataclass
class _EnvironmentState:
    observation: jax.Array
    qpos: jax.Array
    qvel: jax.Array
    additional_carry: _Carry


@struct.dataclass
class _RuntimeState:
    train_state: _TrainState
    env_state: _EnvironmentState
    last_obs: jax.Array
    rng: jax.Array


def _runtime_state(devices, *, parameters_equal: bool = True) -> _RuntimeState:
    """Build actual PmapSharding leaves, not merely arrays with a leading axis."""

    def build(replica_index):
        parameters = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        if not parameters_equal:
            parameters = parameters.at[-1].set(jnp.where(replica_index == 1, 9.0, 4.0))
        offset = replica_index.astype(jnp.float32) * 4.0
        observation = jnp.asarray([[1.0, 2.0], [3.0, 4.0]]) + offset
        return _RuntimeState(
            train_state=_TrainState(params={"w": parameters}),
            env_state=_EnvironmentState(
                observation=observation,
                qpos=jnp.asarray([[1.0, 2.0]]) + offset,
                qvel=jnp.asarray([[0.1, 0.2]]) + offset,
                additional_carry=_Carry(
                    traj_state=_TrajectoryState(traj_no=jnp.full((2,), replica_index, dtype=jnp.int32)),
                    # Paired box terrain has an EmptyState. Path mapping,
                    # rather than this dynamic state, establishes identity.
                    terrain_state={},
                    key=jax.random.fold_in(jax.random.PRNGKey(10), replica_index),
                ),
            ),
            last_obs=observation,
            rng=jax.random.fold_in(jax.random.PRNGKey(0), replica_index),
        )

    return jax.pmap(build, devices=devices)(jnp.arange(len(devices), dtype=jnp.int32))


def _config(path):
    return OmegaConf.create(
        {
            "checkpoint_dir": str(path.parent),
            "debug": {
                "hardware_gate": {
                    "enabled": True,
                    "output_path": str(path),
                    "max_state_sample_values": 8,
                    "max_identity_values": 4,
                }
            },
        }
    )


def _environment(tmp_path):
    motion_paths = []
    terrain_paths = []
    for index in range(2):
        motion = tmp_path / f"motion_{index}.npz"
        terrain = tmp_path / f"motion_{index}_terrain.json"
        motion.write_bytes(f"motion-{index}".encode())
        terrain.write_text(json.dumps({"terrain": index}), encoding="utf-8")
        motion_paths.append(motion)
        terrain_paths.append(terrain)
    return SimpleNamespace(
        paired_motion_paths=tuple(motion_paths),
        paired_terrain_paths=tuple(terrain_paths),
    )


def _read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_hardware_gate_records_replica_state_paths_and_performance(tmp_path, monkeypatch):
    if jax.local_device_count() < 2:
        pytest.skip("needs two forced CPU devices")
    evidence_path = tmp_path / "hardware_gate_evidence.jsonl"
    environment = _environment(tmp_path)
    devices = jax.local_devices()[:2]

    def reject_host_callback(*_args, **_kwargs):
        raise AssertionError("hardware-gate evidence must not use jax.debug.callback")

    monkeypatch.setattr(jax.debug, "callback", reject_host_callback)
    recorder = HardwareGateEvidenceRecorder.from_config(
        _config(evidence_path),
        devices,
        environment,
        resumed=False,
    )
    assert recorder is not None
    state = _runtime_state(devices)
    recorder.record_state(state, phase="initialize", update_number=0, global_timestep=0)
    recorder.record_chunk(
        chunk_updates=5,
        environment_steps=6400,
        elapsed_seconds=2.0,
        update_number=5,
        global_timestep=6400,
        validation_included=False,
        training_metrics={"episode_return": jnp.asarray([1.0, 2.0])},
        ppo_metrics={"loss": jnp.asarray([0.5, 0.25])},
        diagnostics={"gradient_norm": jnp.asarray([3.0, 4.0])},
        validation_metrics=None,
    )
    recorder.record_post_update_once(state, update_number=5, global_timestep=6400)
    recorder.record_post_update_once(state, update_number=10, global_timestep=12800)

    records = _read_records(evidence_path)
    assert [record["record_type"] for record in records] == [
        "replica_state",
        "chunk_performance",
        "replica_state",
    ]
    assert [records[0]["phase"], records[2]["phase"]] == ["initialize", "post_update"]
    initial = records[0]
    assert initial["parameters_equal"] is True
    assert initial["parameter_fingerprints_equal"] is True
    assert initial["rng_digests_distinct"] is True
    assert initial["environment_state_digests_distinct"] is True
    assert [replica["replica_index"] for replica in initial["replicas"]] == [0, 1]
    assert initial["components"]["parameter_fingerprint"]["sampled_values_per_replica"] <= 8
    assert initial["components"]["trajectory_identity"]["source"].endswith("traj_state.traj_no")

    replica_one_identity = initial["replicas"][1]["trajectory_identity"]
    assert replica_one_identity["unique_values_sample"] == [1]
    paired = replica_one_identity["paired_artifacts_sample"][0]
    terrain_path = tmp_path / "motion_1_terrain.json"
    assert paired["trajectory_id"] == 1
    assert paired["terrain_path"] == str(terrain_path)
    assert paired["terrain_sha256"] == hashlib.sha256(terrain_path.read_bytes()).hexdigest()

    performance = records[1]
    assert performance["environment_steps_per_second"] == 3200.0
    assert performance["gpu_memory"]["scope"] == "jax_allocator_only"
    assert performance["gpu_memory"]["external_total_gpu_memory_required"] is True
    assert performance["metric_finiteness"]["all_finite"] is True
    assert performance["metric_finiteness"]["training_metrics"]["all_finite"] is True
    assert performance["metric_finiteness"]["validation_metrics"] == {
        "available": False,
        "reason": "validation was not run at this chunk boundary",
    }
    assert all(
        not replica["cuda_device_memory_snapshot"]["available"] for replica in performance["gpu_memory"]["replicas"]
    )


def test_hardware_gate_appends_resume_session_and_detects_parameter_divergence(tmp_path):
    if jax.local_device_count() < 2:
        pytest.skip("needs two forced CPU devices")
    evidence_path = tmp_path / "hardware_gate_evidence.jsonl"
    environment = _environment(tmp_path)
    devices = jax.local_devices()[:2]

    initial = HardwareGateEvidenceRecorder.from_config(
        _config(evidence_path),
        devices,
        environment,
        resumed=False,
    )
    resumed = HardwareGateEvidenceRecorder.from_config(
        _config(evidence_path),
        devices,
        environment,
        resumed=True,
    )
    assert initial is not None and resumed is not None
    initial.record_state(_runtime_state(devices), phase="initialize", update_number=0, global_timestep=0)
    resumed.record_state(
        _runtime_state(devices, parameters_equal=False),
        phase="resume_initialize",
        update_number=5,
        global_timestep=6400,
    )
    resumed.record_post_update_once(
        _runtime_state(devices),
        update_number=6,
        global_timestep=7680,
    )

    records = _read_records(evidence_path)
    assert [record["phase"] for record in records] == [
        "initialize",
        "resume_initialize",
        "post_resume_update",
    ]
    assert records[0]["session_id"] != records[1]["session_id"]
    assert records[1]["parameters_equal"] is False
    assert records[1]["parameter_fingerprints_equal"] is False
    assert records[2]["parameters_equal"] is True


def test_hardware_gate_reports_exact_nan_counts_at_chunk_boundary(tmp_path):
    if jax.local_device_count() < 2:
        pytest.skip("needs two forced CPU devices")
    evidence_path = tmp_path / "hardware_gate_evidence.jsonl"
    devices = jax.local_devices()[:2]
    recorder = HardwareGateEvidenceRecorder.from_config(
        _config(evidence_path),
        devices,
        _environment(tmp_path),
        resumed=False,
    )
    assert recorder is not None

    replica_indices = jnp.arange(len(devices), dtype=jnp.int32)
    pmapped_metrics = jax.pmap(
        lambda replica_index: {
            "finite": replica_index.astype(jnp.float32),
            "possibly_nan": jnp.where(
                replica_index == 1,
                jnp.asarray(jnp.nan),
                jnp.asarray(7.0),
            ),
        },
        devices=devices,
    )(replica_indices)
    recorder.record_chunk(
        chunk_updates=1,
        environment_steps=1280,
        elapsed_seconds=1.0,
        update_number=1,
        global_timestep=1280,
        validation_included=True,
        training_metrics={"reward": jnp.asarray([1.0, 2.0])},
        ppo_metrics=pmapped_metrics,
        diagnostics={"clip_fraction": jnp.asarray([0.0, 1.0])},
        validation_metrics={"reward": jnp.asarray([3.0, 4.0])},
    )

    finiteness = _read_records(evidence_path)[0]["metric_finiteness"]
    assert finiteness["scope"] == "host_chunk_boundary_final_update_all_replicas"
    assert finiteness["all_finite"] is False
    assert finiteness["ppo_metrics"]["numeric_value_count"] == 4
    assert finiteness["ppo_metrics"]["finite_value_count"] == 3
    assert finiteness["ppo_metrics"]["nonfinite_value_count"] == 1
    assert finiteness["ppo_metrics"]["nan_value_count"] == 1
    assert finiteness["ppo_metrics"]["infinity_value_count"] == 0
    assert finiteness["ppo_metrics"]["nonfinite_leaf_count"] == 1
    assert finiteness["ppo_metrics"]["nonfinite_leaves_sample"] == [
        {
            "path": "possibly_nan",
            "value_count": 2,
            "nonfinite_value_count": 1,
            "nan_value_count": 1,
            "infinity_value_count": 0,
        }
    ]
    assert finiteness["validation_metrics"]["all_finite"] is True


def test_hardware_gate_is_default_off_and_requires_a_persistence_path(tmp_path):
    devices = jax.local_devices()[:2]
    assert (
        HardwareGateEvidenceRecorder.from_config(
            OmegaConf.create({"debug": {"hardware_gate": {"enabled": False}}}),
            devices,
            object(),
            resumed=False,
        )
        is None
    )
    with pytest.raises(ValueError, match="requires checkpoint_dir"):
        HardwareGateEvidenceRecorder.from_config(
            OmegaConf.create({"debug": {"hardware_gate": {"enabled": True}}}),
            devices,
            object(),
            resumed=False,
        )
    assert not (tmp_path / "hardware_gate_evidence.jsonl").exists()
