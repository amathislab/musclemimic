"""MuJoCo-version-aware MJX-Warp graph lifecycle helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def disable_warp_graph_eviction() -> None:
    """Defer CUDA graph destruction to a synchronized host boundary."""
    api = _warp_ffi_api()
    setter = getattr(api, "set_jax_callable_default_graph_cache_max", None)
    if not callable(setter):
        raise RuntimeError("This MuJoCo build cannot disable MJX-Warp graph-cache eviction.")
    setter(None)


def _warp_ffi_api() -> Any:
    """Return the MJX-vendored Warp FFI registry across supported MuJoCo APIs."""
    from mujoco.mjx.warp import ffi as mjx_warp_ffi

    for attribute in ("warp_ffi", "ffi"):
        api = getattr(mjx_warp_ffi, attribute, None)
        if api is not None and callable(getattr(api, "clear_jax_callable_graph_cache", None)):
            return api
    raise RuntimeError(
        "This MuJoCo build does not expose the MJX-Warp graph-cache lifecycle API. "
        "Multi-GPU MJX-Warp training is disabled until the installed version is adapted."
    )


def clear_warp_graph_cache() -> None:
    """Clear plain-Warp captures after all device work has completed."""
    _warp_ffi_api().clear_jax_callable_graph_cache()


def _model_graph_mode(model: Any) -> Any | None:
    """Read the Warp graph mode across the MuJoCo 3.7 and 3.11 layouts."""
    direct_mode = getattr(model, "graph_mode", None)
    if direct_mode is not None:
        return direct_mode

    # MuJoCo 3.11 stores the setting on the Warp-specific option payload.
    options = getattr(model, "opt", None)
    implementation_options = getattr(options, "_impl", None)
    return getattr(implementation_options, "graph_mode", None)


def require_safe_warp_graph_mode(env: Any) -> None:
    """Require the device-keyed plain-Warp cache for pmapped execution.

    MuJoCo keys plain-Warp captures by CUDA device and XLA buffer pointers.
    Staged modes instead keep one mutable staging descriptor per traced FFI
    call, which cannot be shared by multiple ``pmap`` replicas.
    """
    from mujoco.mjx.warp.types import GraphMode

    model = getattr(env, "sys", None)
    mode = None if model is None else _model_graph_mode(model)
    if mode is None:
        raise RuntimeError("Multi-GPU MJX-Warp requires a model created with graph_mode=GraphMode.WARP.")
    resolved_mode = int(np.asarray(mode))
    if resolved_mode != int(GraphMode.WARP):
        raise RuntimeError(
            f"Unsafe MJX-Warp graph mode for multi-GPU training: expected GraphMode.WARP, got {resolved_mode}."
        )
    # Resolve the lifecycle hook during preflight, before any pmapped CUDA work
    # can create captures that this MuJoCo build cannot safely retire.
    _warp_ffi_api()


__all__ = ["clear_warp_graph_cache", "disable_warp_graph_eviction", "require_safe_warp_graph_mode"]
