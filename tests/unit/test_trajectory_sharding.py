"""Tests for multi-device trajectory sharding."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.common import trajectory_sharding as sharding

N_DEVICES = 4


@pytest.fixture(scope="module", autouse=True)
def verify_device_simulator() -> None:
    if jax.local_device_count() != N_DEVICES:
        pytest.skip(
            f"Expected {N_DEVICES} simulated devices, got {jax.local_device_count()}. "
            "Run this test in a fresh process with the configured XLA_FLAGS."
        )


@dataclass
class FakeTrajectoryData:
    qpos: np.ndarray
    qvel: np.ndarray
    site_xpos: np.ndarray
    split_points: np.ndarray
    metadata: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass(frozen=True)
class FakeTrajectory:
    data: object


def make_trajectory_data(motion_lengths: list[int], *, nq: int = 4, nv: int = 3) -> FakeTrajectoryData:
    n_samples = sum(motion_lengths)
    rng = np.random.default_rng(0)
    return FakeTrajectoryData(
        qpos=rng.standard_normal((n_samples, nq)).astype(np.float32),
        qvel=rng.standard_normal((n_samples, nv)).astype(np.float32),
        site_xpos=rng.standard_normal((n_samples, 5, 3)).astype(np.float32),
        split_points=np.concatenate([[0], np.cumsum(motion_lengths)]).astype(np.int32),
    )


def test_trajectory_sharding_mode_defaults_to_none():
    config = SimpleNamespace(experiment=SimpleNamespace())
    assert sharding.trajectory_sharding_mode(config) == "none"


def test_trajectory_sharding_mode_accepts_static():
    config = SimpleNamespace(experiment=SimpleNamespace(trajectory={"sharding": "static"}))
    assert sharding.trajectory_sharding_mode(config) == "static"


def test_trajectory_sharding_mode_rejects_unknown_mode():
    config = SimpleNamespace(experiment=SimpleNamespace(trajectory={"sharding": "dynamic"}))
    with pytest.raises(ValueError, match=r"trajectory\.sharding"):
        sharding.trajectory_sharding_mode(config)


def test_shard_spec_splits_evenly():
    spec = sharding.ShardSpec.build(n_traj_global=100, n_devices=4)
    assert spec.n_global == 100
    assert spec.n_devices == 4
    assert spec.counts == (25, 25, 25, 25)
    assert spec.offsets == (0, 25, 50, 75, 100)


def test_shard_spec_assigns_remainder_to_first_devices():
    spec = sharding.ShardSpec.build(n_traj_global=103, n_devices=4)
    assert spec.counts == (26, 26, 26, 25)
    assert spec.offsets == (0, 26, 52, 78, 103)


def test_shard_spec_rejects_too_few_motions():
    with pytest.raises(ValueError, match="Cannot shard"):
        sharding.ShardSpec.build(n_traj_global=3, n_devices=4)


@pytest.mark.parametrize(
    ("n_traj_global", "n_devices", "message"),
    [
        (0, 1, "n_traj_global must be positive"),
        (1, 0, "n_devices must be positive"),
    ],
)
def test_shard_spec_rejects_non_positive_sizes(n_traj_global, n_devices, message):
    with pytest.raises(ValueError, match=message):
        sharding.ShardSpec.build(n_traj_global, n_devices)


def test_shard_ranges_cover_every_motion_once():
    spec = sharding.ShardSpec.build(n_traj_global=314, n_devices=4)
    covered_ids = []
    for device_id in range(spec.n_devices):
        start, stop = spec.local_range(device_id)
        covered_ids.extend(range(start, stop))
    assert covered_ids == list(range(314))


def test_shard_spec_is_deterministic():
    first = sharding.ShardSpec.build(100, 4)
    second = sharding.ShardSpec.build(100, 4)
    assert first == second


def test_device_masks_form_a_disjoint_cover():
    spec = sharding.ShardSpec.build(n_traj_global=100, n_devices=4)
    masks = np.stack([np.asarray(spec.mask_for_device(device_id)) for device_id in range(4)])
    np.testing.assert_array_equal(masks.sum(axis=0), np.ones(100, dtype=np.float32))


def test_uniform_device_weights_sum_to_one():
    spec = sharding.ShardSpec.build(n_traj_global=100, n_devices=4)
    for device_id in range(4):
        weights = np.asarray(spec.uniform_weights_for_device(device_id))
        start, stop = spec.local_range(device_id)
        assert weights.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(weights[start:stop], 1.0 / spec.local_count(device_id))
        assert np.count_nonzero(weights) == spec.local_count(device_id)


def test_shard_mask_restricts_and_renormalizes_global_weights():
    spec = sharding.ShardSpec.build(n_traj_global=100, n_devices=4)
    global_weights = np.zeros(100, dtype=np.float32)
    global_weights[[20, 55, 60]] = [3.0, 10.0, 5.0]
    global_weights /= global_weights.sum()

    device_zero_weights = np.asarray(sharding.apply_shard_mask_and_renormalize(jnp.asarray(global_weights), spec, 0))
    assert device_zero_weights[20] == pytest.approx(1.0)
    assert device_zero_weights[[55, 60]].sum() == 0.0

    device_two_weights = np.asarray(sharding.apply_shard_mask_and_renormalize(jnp.asarray(global_weights), spec, 2))
    assert device_two_weights[55] == pytest.approx(2.0 / 3.0)
    assert device_two_weights[60] == pytest.approx(1.0 / 3.0)


def test_shard_mask_uses_uniform_weights_when_shard_has_zero_mass():
    spec = sharding.ShardSpec.build(n_traj_global=100, n_devices=4)
    global_weights = np.zeros(100, dtype=np.float32)
    global_weights[:25] = 1.0 / 25

    weights = np.asarray(sharding.apply_shard_mask_and_renormalize(jnp.asarray(global_weights), spec, 3))
    start, stop = spec.local_range(3)
    np.testing.assert_allclose(weights[start:stop], 1.0 / 25)
    assert weights.sum() == pytest.approx(1.0)


def test_shard_trajectory_data_covers_library_and_pads_samples():
    data = make_trajectory_data([3, 5, 2, 4])
    spec = sharding.ShardSpec.build(n_traj_global=4, n_devices=2)
    shards, plan = sharding.shard_trajectory_data(data, spec)

    assert plan.n_traj_global == 4
    assert plan.n_devices == 2
    assert plan.n_traj_per_shard == 2
    assert plan.n_samples_per_shard == 8
    np.testing.assert_array_equal(shards[0].qpos[:8], data.qpos[:8])
    np.testing.assert_array_equal(shards[1].qpos[:6], data.qpos[8:14])
    np.testing.assert_array_equal(shards[1].qpos[6:], np.zeros_like(shards[1].qpos[6:]))


@pytest.mark.parametrize(
    ("split_points", "message"),
    [
        (np.array([[0, 2, 4]], dtype=np.int32), "one-dimensional integer array"),
        (np.array([0.0, 2.0, 4.0]), "one-dimensional integer array"),
        (np.array([0, 4], dtype=np.int32), "describe 1 motions"),
        (np.array([1, 2, 4], dtype=np.int32), "start at zero"),
        (np.array([0, 2, 2], dtype=np.int32), "at least one sample"),
        (np.array([0, 3, 2], dtype=np.int32), "at least one sample"),
    ],
)
def test_shard_trajectory_data_rejects_invalid_split_points(split_points, message):
    data = make_trajectory_data([2, 2])
    data.split_points = split_points
    spec = sharding.ShardSpec.build(n_traj_global=2, n_devices=1)

    with pytest.raises(ValueError, match=message):
        sharding.shard_trajectory_data(data, spec)


@pytest.mark.parametrize("field_name", ["qpos", "qvel"])
def test_shard_trajectory_data_rejects_sample_count_mismatch(field_name):
    data = make_trajectory_data([2, 2])
    setattr(data, field_name, getattr(data, field_name)[:-1])
    spec = sharding.ShardSpec.build(n_traj_global=2, n_devices=1)

    with pytest.raises(ValueError, match=rf"{field_name} has 3 samples"):
        sharding.shard_trajectory_data(data, spec)


def test_local_split_points_are_padded_to_equal_shapes():
    data = make_trajectory_data([3, 5, 2, 4, 7])
    spec = sharding.ShardSpec.build(n_traj_global=5, n_devices=2)
    shards, _ = sharding.shard_trajectory_data(data, spec)

    np.testing.assert_array_equal(shards[0].split_points, np.array([0, 3, 8, 10], dtype=np.int32))
    np.testing.assert_array_equal(shards[1].split_points, np.array([0, 4, 11, 11], dtype=np.int32))


def test_padded_motion_slots_are_zero_length():
    data = make_trajectory_data([3, 5, 2])
    spec = sharding.ShardSpec.build(n_traj_global=3, n_devices=2)
    shards, _ = sharding.shard_trajectory_data(data, spec)
    assert shards[1].split_points[1] == shards[1].split_points[2]


def test_all_shard_field_shapes_match_across_devices():
    data = make_trajectory_data([3, 5, 2, 4, 7, 1, 6])
    spec = sharding.ShardSpec.build(n_traj_global=7, n_devices=3)
    shards, _ = sharding.shard_trajectory_data(data, spec)

    for field_name in ("qpos", "qvel", "site_xpos", "split_points"):
        shapes = {getattr(shard, field_name).shape for shard in shards}
        assert len(shapes) == 1, f"{field_name} shapes differ: {shapes}"


def test_global_split_points_keep_global_trajectory_ids():
    data = make_trajectory_data([3, 5, 2, 4, 7])
    spec = sharding.ShardSpec.build(n_traj_global=5, n_devices=2)
    shards, _ = sharding.shard_trajectory_data(data, spec, global_indexed_split_points=True)

    np.testing.assert_array_equal(
        shards[0].split_points,
        np.array([0, 3, 8, 10, 10, 10], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        shards[1].split_points,
        np.array([0, 0, 0, 0, 4, 11], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        shards[1].qpos[shards[1].split_points[4]],
        data.qpos[data.split_points[4]],
    )


def test_shard_host_to_devices_places_each_payload():
    per_device = [{"values": jnp.full((3,), float(device_id), dtype=jnp.float32)} for device_id in range(N_DEVICES)]
    result = sharding.shard_host_to_devices(per_device)

    assert result["values"].shape == (N_DEVICES, 3)
    for device_id in range(N_DEVICES):
        np.testing.assert_allclose(np.asarray(result["values"])[device_id], float(device_id))


def test_shard_host_to_devices_stacks_payloads_on_host(monkeypatch):
    per_device = [{"values": np.full((3,), float(device_id), dtype=np.float32)} for device_id in range(N_DEVICES)]
    device_put_inputs = []
    real_device_put = sharding.jax.device_put

    def record_device_put(value, target_sharding):
        device_put_inputs.append(value)
        return real_device_put(value, target_sharding)

    monkeypatch.setattr(sharding.jax, "device_put", record_device_put)

    result = sharding.shard_host_to_devices(per_device)

    assert len(device_put_inputs) == 1
    assert isinstance(device_put_inputs[0], np.ndarray)
    assert result["values"].shape == (N_DEVICES, 3)


def test_shard_host_to_devices_rejects_mismatched_shapes():
    per_device = [jnp.zeros((3,), dtype=jnp.float32) for _ in range(N_DEVICES)]
    per_device[1] = jnp.zeros((4,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="mismatched shapes"):
        sharding.shard_host_to_devices(per_device)


def test_prepare_trajectory_shards_requires_multiple_devices():
    config = OmegaConf.create({"experiment": {"trajectory": {"sharding": "static"}}})
    env = SimpleNamespace(th=SimpleNamespace(traj=object()))
    with pytest.raises(ValueError, match="requires more than one device"):
        sharding.prepare_trajectory_data_shards(env, config, n_devices=1)


def test_prepare_trajectory_shards_requires_handler():
    config = OmegaConf.create({"experiment": {"trajectory": {"sharding": "static"}}})
    env = SimpleNamespace(th=None)
    with pytest.raises(ValueError, match="trajectory handler"):
        sharding.prepare_trajectory_data_shards(env, config, n_devices=2)


def test_prepare_trajectory_shards_keeps_resident_data_on_host(monkeypatch):
    config = OmegaConf.create({"experiment": {"trajectory": {"sharding": "static"}}})
    trajectory = FakeTrajectory(data=make_trajectory_data([3, 5, 2, 4]))
    env = SimpleNamespace(
        th=SimpleNamespace(
            traj=trajectory,
            n_trajectories=4,
            _is_numpy=True,
        )
    )
    captured = {}

    def capture_device_shards(per_device_trees, devices):
        captured["shards"] = per_device_trees
        captured["devices"] = devices
        return "device-shards"

    monkeypatch.setattr(sharding, "shard_host_to_devices", capture_device_shards)

    result = sharding.prepare_trajectory_data_shards(env, config, n_devices=2)

    assert result == "device-shards"
    assert env.th.traj.data is captured["shards"][0]
    assert isinstance(env.th.traj.data.qpos, np.ndarray)
    assert env.th._is_numpy is True
    assert captured["devices"] == jax.local_devices()[:2]


def test_distinct_validation_library_uses_its_own_shards(monkeypatch):
    config = OmegaConf.create(
        {
            "experiment": {
                "trajectory": {"sharding": "static"},
                "validation": {},
            }
        }
    )
    val_env = SimpleNamespace(th=SimpleNamespace(n_trajectories=12))
    captured = {}

    def fake_prepare(env, passed_config, n_devices, *, label):
        captured.update(env=env, config=passed_config, n_devices=n_devices, label=label)
        return "validation-shards"

    monkeypatch.setattr(sharding, "prepare_trajectory_data_shards", fake_prepare)
    shards = sharding.prepare_validation_trajectory_data_shards(
        val_env,
        config,
        n_devices=4,
        shares_training=False,
    )

    assert shards == "validation-shards"
    assert config.experiment.validation.n_trajectories == 12
    assert captured == {
        "env": val_env,
        "config": config,
        "n_devices": 4,
        "label": "validation",
    }


def test_distinct_training_and_validation_shards_are_installed_and_restored():
    train_trajectory = FakeTrajectory(data="full-training-data")
    validation_trajectory = FakeTrajectory(data="full-validation-data")
    env = SimpleNamespace(th=SimpleNamespace(traj=train_trajectory, _is_numpy=True))
    val_env = SimpleNamespace(th=SimpleNamespace(traj=validation_trajectory, _is_numpy=True))

    restore_tokens = sharding.install_trajectory_data_overrides(
        env,
        val_env,
        trajectory_data_override="training-shard",
        validation_trajectory_data_override="validation-shard",
    )

    assert env.th.traj.data == "training-shard"
    assert val_env.th.traj.data == "validation-shard"
    assert env.th._is_numpy is False
    assert val_env.th._is_numpy is False
    sharding.restore_trajectory_data_overrides(restore_tokens)
    assert env.th.traj is train_trajectory
    assert val_env.th.traj is validation_trajectory
    assert env.th._is_numpy is True
    assert val_env.th._is_numpy is True


def test_shared_validation_receives_training_shard_object():
    trajectory = FakeTrajectory(data="full-data")
    env = SimpleNamespace(th=SimpleNamespace(traj=trajectory, _is_numpy=True))
    val_env = SimpleNamespace(th=SimpleNamespace(traj=trajectory, _is_numpy=True))

    restore_tokens = sharding.install_trajectory_data_overrides(
        env,
        val_env,
        trajectory_data_override="training-shard",
        validation_trajectory_data_override=None,
    )

    assert val_env.th.traj is env.th.traj
    assert env.th.traj.data == "training-shard"
    assert env.th._is_numpy is False
    assert val_env.th._is_numpy is False
    sharding.restore_trajectory_data_overrides(restore_tokens)
    assert env.th.traj is trajectory
    assert val_env.th.traj is trajectory
    assert env.th._is_numpy is True
    assert val_env.th._is_numpy is True
