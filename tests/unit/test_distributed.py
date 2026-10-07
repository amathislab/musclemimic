"""Tests for multi-device runtime configuration."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import pytest

N_DEVICES = 4


def load_distributed_module():
    module_name = "musclemimic_distributed_under_test"
    module_path = (
        Path(__file__).resolve().parents[2] / "musclemimic" / "algorithms" / "common" / "distributed.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


distributed = load_distributed_module()


@pytest.fixture(scope="module", autouse=True)
def verify_device_simulator() -> None:
    if jax.local_device_count() != N_DEVICES:
        pytest.skip(
            f"Expected {N_DEVICES} simulated devices, got {jax.local_device_count()}. "
            "Run this test in a fresh process with the configured XLA_FLAGS."
        )


def test_device_count_defaults_to_single_device():
    config = SimpleNamespace(experiment=SimpleNamespace())
    assert distributed.device_count(config) == 1


def test_device_count_respects_explicit_override():
    config = SimpleNamespace(experiment=SimpleNamespace(distributed=SimpleNamespace(num_devices=2)))
    assert distributed.device_count(config) == 2


def test_device_count_rejects_oversize_request():
    config = SimpleNamespace(experiment=SimpleNamespace(distributed=SimpleNamespace(num_devices=99)))
    with pytest.raises(ValueError, match="exceeds available"):
        distributed.device_count(config)


def test_device_count_rejects_zero():
    config = SimpleNamespace(experiment=SimpleNamespace(distributed=SimpleNamespace(num_devices=0)))
    with pytest.raises(ValueError, match="num_devices"):
        distributed.device_count(config)


def test_device_count_minus_one_uses_all_devices():
    config = SimpleNamespace(experiment=SimpleNamespace(distributed=SimpleNamespace(num_devices=-1)))
    assert distributed.device_count(config) == N_DEVICES
