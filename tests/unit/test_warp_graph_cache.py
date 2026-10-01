"""Compatibility tests for the MJX-Warp graph lifecycle adapter."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from musclemimic.core.mujoco_mjx import Mjx, _parse_warp_graph_mode
from musclemimic.runner import warp_graph_cache


class _GraphCacheApi:
    def __init__(self):
        self.cleared = 0

    def clear_jax_callable_graph_cache(self):
        self.cleared += 1


class _FunctionalTerrainHook:
    def update(self, _env, model, data, carry, backend):
        assert backend is jnp
        geom_pos = model.geom_pos.at[0, 0].set(carry + 1.25)
        return model.tree_replace({"geom_pos": geom_pos}), data, carry


class _FunctionalRandomizerHook:
    def update(self, _env, model, data, carry, backend):
        assert backend is jnp
        body_mass = model.body_mass.at[1].set(carry + 2.5)
        return model.tree_replace({"body_mass": body_mass}), data, carry


def test_warp_ffi_api_prefers_mujoco_311_registry(monkeypatch):
    from mujoco.mjx.warp import ffi

    current = _GraphCacheApi()
    legacy = _GraphCacheApi()
    monkeypatch.setattr(ffi, "warp_ffi", current, raising=False)
    monkeypatch.setattr(ffi, "ffi", legacy, raising=False)

    assert warp_graph_cache._warp_ffi_api() is current
    warp_graph_cache.clear_warp_graph_cache()
    assert current.cleared == 1
    assert legacy.cleared == 0


def test_warp_ffi_api_supports_legacy_registry(monkeypatch):
    from mujoco.mjx.warp import ffi

    legacy = _GraphCacheApi()
    monkeypatch.setattr(ffi, "warp_ffi", None, raising=False)
    monkeypatch.setattr(ffi, "ffi", legacy, raising=False)

    assert warp_graph_cache._warp_ffi_api() is legacy


def test_warp_ffi_api_fails_closed_without_clear_capability(monkeypatch):
    from mujoco.mjx.warp import ffi

    monkeypatch.setattr(ffi, "warp_ffi", object(), raising=False)
    monkeypatch.setattr(ffi, "ffi", object(), raising=False)

    with pytest.raises(RuntimeError, match="graph-cache lifecycle"):
        warp_graph_cache._warp_ffi_api()


def test_warp_staged_ex_config_maps_to_mujoco_graph_mode():
    from mujoco.mjx.warp.types import GraphMode

    assert _parse_warp_graph_mode("warp_staged_ex") == GraphMode.WARP_STAGED_EX
    with pytest.raises(ValueError, match="warp_staged_ex"):
        _parse_warp_graph_mode("warp_staged")


def test_multi_gpu_requires_plain_warp_graph_mode():
    from mujoco.mjx.warp.types import GraphMode

    safe_env = SimpleNamespace(sys=SimpleNamespace(graph_mode=GraphMode.WARP))
    warp_graph_cache.require_safe_warp_graph_mode(safe_env)

    for unsafe_mode in (GraphMode.JAX, GraphMode.WARP_STAGED, GraphMode.WARP_STAGED_EX):
        unsafe_env = SimpleNamespace(sys=SimpleNamespace(graph_mode=unsafe_mode))
        with pytest.raises(RuntimeError, match="Unsafe MJX-Warp graph mode"):
            warp_graph_cache.require_safe_warp_graph_mode(unsafe_env)


def test_multi_gpu_reads_mujoco_311_nested_graph_mode():
    from mujoco.mjx.warp.types import GraphMode

    safe_env = SimpleNamespace(
        sys=SimpleNamespace(
            opt=SimpleNamespace(
                _impl=SimpleNamespace(graph_mode=GraphMode.WARP),
            )
        )
    )
    warp_graph_cache.require_safe_warp_graph_mode(safe_env)


def test_multi_gpu_rejects_model_without_graph_mode():
    with pytest.raises(RuntimeError, match=r"GraphMode\.WARP"):
        warp_graph_cache.require_safe_warp_graph_mode(SimpleNamespace(sys=object()))


def test_multi_gpu_preflight_requires_graph_cache_lifecycle(monkeypatch):
    from mujoco.mjx.warp.types import GraphMode

    def unavailable():
        raise RuntimeError("missing graph-cache lifecycle")

    monkeypatch.setattr(warp_graph_cache, "_warp_ffi_api", unavailable)
    env = SimpleNamespace(sys=SimpleNamespace(graph_mode=GraphMode.WARP))

    with pytest.raises(RuntimeError, match="graph-cache lifecycle"):
        warp_graph_cache.require_safe_warp_graph_mode(env)


def test_current_mujoco_warp_model_preserves_plain_warp_graph_mode_on_host():
    from mujoco.mjx.warp.types import GraphMode

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>'
    )
    system = jax.device_get(mjx.put_model(model, impl="warp", graph_mode=GraphMode.WARP))

    warp_graph_cache.require_safe_warp_graph_mode(SimpleNamespace(sys=system))
    assert isinstance(system.body_mass, np.ndarray)


def test_pre_step_materializes_host_model_for_functional_hooks():
    from mujoco.mjx.warp.types import GraphMode

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="free"/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    host_system = jax.device_get(mjx.put_model(model, impl="warp", graph_mode=GraphMode.WARP))
    original_geom_x = host_system.geom_pos[0, 0]
    original_body_mass = host_system.body_mass[1]
    env = Mjx.__new__(Mjx)
    env._terrain = _FunctionalTerrainHook()
    env._domain_randomizer = _FunctionalRandomizerHook()

    def apply_hooks(replica_index):
        system, _, _ = env._mjx_simulation_pre_step(host_system, None, replica_index)
        return system.geom_pos[0, 0], system.body_mass[1]

    replica_indices = jnp.arange(jax.local_device_count())
    geom_x, body_mass = jax.pmap(apply_hooks)(replica_indices)

    np.testing.assert_allclose(np.asarray(geom_x), np.arange(jax.local_device_count()) + 1.25)
    np.testing.assert_allclose(np.asarray(body_mass), np.arange(jax.local_device_count()) + 2.5)
    assert isinstance(host_system.geom_pos, np.ndarray)
    assert isinstance(host_system.body_mass, np.ndarray)
    assert host_system.geom_pos[0, 0] == original_geom_x
    assert host_system.body_mass[1] == original_body_mass


def test_current_mujoco_put_data_can_trace_inside_pmap():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="free"/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    devices = jax.local_devices()

    def place_data(_):
        placed = mjx.put_data(
            model,
            data,
            impl="warp",
            naconmax=16,
            njmax=16,
        )
        return placed.qpos

    result = jax.pmap(place_data, devices=devices)(np.arange(len(devices)))

    assert result.shape == (len(devices), model.nq)
