"""CPU-testable control-plane checks for the standalone two-GPU Warp probe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from musclemimic.runner import warp_multi_gpu_probe as probe


@dataclass(frozen=True)
class _FakeDevice:
    id: int
    platform: str = "gpu"
    device_kind: str = "fake GPU"


@dataclass(frozen=True)
class _FakeShard:
    device: _FakeDevice


class _FakeArray:
    def __init__(self, devices):
        self.addressable_shards = tuple(_FakeShard(device) for device in devices)


def test_sentinel_placement_uses_explicit_named_sharding(monkeypatch):
    devices = (_FakeDevice(0), _FakeDevice(1))
    events = {}
    placed = object()

    def make_mesh(device_array, axis_names):
        events["mesh"] = (tuple(device_array.tolist()), axis_names)
        return "mesh"

    def make_sharding(mesh, partition_spec):
        events["sharding"] = (mesh, partition_spec)
        return "named-sharding"

    monkeypatch.setattr(probe.jax.sharding, "Mesh", make_mesh)
    monkeypatch.setattr(probe.jax.sharding, "NamedSharding", make_sharding)
    monkeypatch.setattr(probe.jax.sharding, "PartitionSpec", lambda *axes: axes)
    monkeypatch.setattr(
        probe.jax,
        "device_put",
        lambda values, sharding: events.update(values=np.asarray(values), device_put_sharding=sharding) or placed,
    )

    actual = probe._put_sentinel_shards(np.asarray((101.0, 102.0)), devices)

    assert actual is placed
    assert events["mesh"] == (devices, (probe._AXIS_NAME,))
    assert events["sharding"] == ("mesh", (probe._AXIS_NAME,))
    assert events["device_put_sharding"] == "named-sharding"
    np.testing.assert_array_equal(events["values"], np.asarray((101.0, 102.0), dtype=np.float32))


def test_device_inventory_requires_exactly_two_distinct_gpu_devices():
    devices = (_FakeDevice(0), _FakeDevice(1))
    probe._validate_device_inventory(devices, ("cuda:0", "cuda:1"))

    invalid_inventories = (
        (devices[:1], ("cuda:0", "cuda:1")),
        ((*devices, _FakeDevice(2)), ("cuda:0", "cuda:1")),
        ((_FakeDevice(0), _FakeDevice(0)), ("cuda:0", "cuda:1")),
        ((_FakeDevice(0, platform="cpu"), _FakeDevice(1)), ("cuda:0", "cuda:1")),
        (devices, ("cuda:0",)),
        (devices, ("cuda:0", "cuda:1", "cuda:2")),
    )
    for jax_devices, warp_devices in invalid_inventories:
        with pytest.raises(probe.ProbeError):
            probe._validate_device_inventory(jax_devices, warp_devices)


def test_probe_requires_plain_warp_graph_mode():
    from mujoco.mjx.warp.types import GraphMode

    plain = SimpleNamespace(opt=SimpleNamespace(_impl=SimpleNamespace(graph_mode=GraphMode.WARP)))
    staged = SimpleNamespace(opt=SimpleNamespace(_impl=SimpleNamespace(graph_mode=GraphMode.WARP_STAGED_EX)))

    probe._require_plain_warp_graph_mode(plain)
    with pytest.raises(probe.ProbeError, match=r"GraphMode\.WARP"):
        probe._require_plain_warp_graph_mode(staged)
    with pytest.raises(probe.ProbeError, match="does not expose"):
        probe._require_plain_warp_graph_mode(object())


def test_execute_twice_blocks_and_clears_each_result_before_recapture():
    events = []

    def run(value):
        events.append(f"run:{value}")
        return f"result:{value}"

    def block(value):
        events.append(f"block:{value}")
        return f"blocked:{value}"

    def clear():
        events.append("clear")

    results = probe._execute_twice(run, ("first", "second"), block_until_ready=block, clear_graph_cache=clear)

    assert results == ("blocked:result:first", "blocked:result:second")
    assert events == [
        "run:first",
        "block:result:first",
        "clear",
        "run:second",
        "block:result:second",
        "clear",
    ]


def test_execute_twice_does_not_clear_after_failed_block():
    events = []

    def block(_value):
        raise RuntimeError("device did not become quiescent")

    with pytest.raises(RuntimeError, match="quiescent"):
        probe._execute_twice(
            lambda value: value,
            ("first", "second"),
            block_until_ready=block,
            clear_graph_cache=lambda: events.append("clear"),
        )

    assert events == []


def test_device_shard_validation_requires_both_expected_devices_for_every_output():
    devices = (_FakeDevice(0), _FakeDevice(1))
    valid = {"axis_id": _FakeArray(devices), "trajectory": _FakeArray(devices)}
    probe._validate_device_shards(valid, devices)

    with pytest.raises(probe.ProbeError, match="trajectory"):
        probe._validate_device_shards(
            {"axis_id": _FakeArray(devices), "trajectory": _FakeArray((devices[0], devices[0]))},
            devices,
        )


def test_host_result_validation_checks_axis_ids_sentinels_and_finiteness():
    sentinels = np.asarray((101.0, 102.0), dtype=np.float32)
    valid = {
        "axis_id": np.asarray((0, 1), dtype=np.int32),
        "final_x": sentinels.copy(),
        "initial_x": sentinels.copy(),
        "sentinel": sentinels.copy(),
        "trajectory": np.zeros((2, probe._SCAN_STEPS, 13), dtype=np.float32),
    }
    valid["trajectory"][:, :, 0] = sentinels[:, np.newaxis]
    probe._validate_host_result(valid, sentinels)

    invalid_axis_ids = {**valid, "axis_id": np.asarray((1, 0), dtype=np.int32)}
    with pytest.raises(AssertionError):
        probe._validate_host_result(invalid_axis_ids, sentinels)

    non_finite = {**valid, "trajectory": valid["trajectory"].copy()}
    non_finite["trajectory"][1, 0, 0] = np.nan
    with pytest.raises(probe.ProbeError, match="non-finite"):
        probe._validate_host_result(non_finite, sentinels)

    wrong_physics = {**valid, "trajectory": valid["trajectory"].copy()}
    wrong_physics["trajectory"][:, :, 0] = 0.0
    with pytest.raises(AssertionError):
        probe._validate_host_result(wrong_physics, sentinels)


def test_execution_validation_requires_changed_replica_sentinels(monkeypatch):
    devices = (_FakeDevice(0), _FakeDevice(1))
    device_result = {
        name: _FakeArray(devices) for name in ("axis_id", "final_x", "initial_x", "sentinel", "trajectory")
    }
    sentinels = np.asarray((101.0, 102.0), dtype=np.float32)
    host_result = {
        "axis_id": np.asarray((0, 1), dtype=np.int32),
        "final_x": sentinels.copy(),
        "initial_x": sentinels.copy(),
        "sentinel": sentinels.copy(),
        "trajectory": np.zeros((2, probe._SCAN_STEPS, 13), dtype=np.float32),
    }
    host_result["trajectory"][:, :, 0] = sentinels[:, np.newaxis]
    monkeypatch.setattr(probe, "_to_host_result", lambda _result: host_result)

    with pytest.raises(probe.ProbeError, match="must change"):
        probe._validate_executions((device_result, device_result), (sentinels, sentinels), devices)


def test_installed_provenance_reports_module_distribution_and_direct_url(monkeypatch, tmp_path):
    import musclemimic

    package_file = tmp_path / "installed" / "musclemimic" / "__init__.py"
    direct_url = {"url": "https://example.invalid/musclemimic.git", "vcs_info": {"commit_id": "abc123"}}
    fake_distribution = SimpleNamespace(
        version="9.8.7",
        read_text=lambda name: json.dumps(direct_url) if name == "direct_url.json" else None,
    )
    monkeypatch.setattr(musclemimic, "__file__", str(package_file))
    monkeypatch.setattr(probe.metadata, "distribution", lambda _name: fake_distribution)

    assert probe._installed_provenance() == {
        "direct_url": direct_url,
        "distribution_version": "9.8.7",
        "musclemimic_file": str(package_file.resolve()),
    }


def test_expected_commit_requires_exact_pep610_vcs_pin():
    expected = "a" * 40
    provenance = {
        "direct_url": {
            "url": "https://example.invalid/musclemimic.git",
            "vcs_info": {"commit_id": expected.upper()},
        }
    }

    assert probe._validate_expected_commit(provenance, expected) == expected

    with pytest.raises(probe.ProbeError, match="40-character"):
        probe._validate_expected_commit(provenance, "abc123")
    with pytest.raises(probe.ProbeError, match="does not match"):
        probe._validate_expected_commit(provenance, "b" * 40)
    with pytest.raises(probe.ProbeError, match="PEP 610"):
        probe._validate_expected_commit({"direct_url": None}, expected)
