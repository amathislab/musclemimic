"""Safe CUDA graph cache lifecycle for MJX-Warp training."""

from __future__ import annotations


def disable_warp_graph_eviction() -> None:
    """Defer CUDA graph destruction to a synchronized host boundary."""
    from mujoco.mjx.warp import ffi as mjx_warp_ffi

    mjx_warp_ffi.ffi.set_jax_callable_default_graph_cache_max(None)


def clear_warp_graph_cache() -> None:
    """Destroy cached graphs after all device work reaches a host boundary."""
    from mujoco.mjx.warp import ffi as mjx_warp_ffi

    mjx_warp_ffi.ffi.clear_jax_callable_graph_cache()
