"""Helpers for single-process JAX data-parallel training."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np

PMAP_AXIS_NAME = "devices"


def replicate_for_pmap(tree: Any, devices: list[Any] | tuple[Any, ...]) -> Any:
    """Place one identical leading-axis value per pmap device.

    JAX 0.10 removed ``device_put_replicated``. A fully replicated
    ``NamedSharding(..., PartitionSpec())`` is not equivalent for
    ``pmap(in_axes=0)`` because it does not add the replica axis.
    """
    devices = tuple(devices)
    if not devices:
        raise ValueError("replicate_for_pmap requires at least one device")
    mesh = jax.sharding.Mesh(np.asarray(devices), (PMAP_AXIS_NAME,))
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(PMAP_AXIS_NAME),
    )

    def place(value):
        if value is None:
            return None
        host_value = np.asarray(jax.device_get(value))
        replicated = np.broadcast_to(
            host_value[None],
            (len(devices), *host_value.shape),
        )
        return jax.device_put(replicated, sharding)

    return jax.tree_util.tree_map(
        place,
        tree,
        is_leaf=lambda value: value is None,
    )


def assert_warp_multi_gpu_compatible() -> None:
    """Fail early for Warp versions known to produce bad pmapped results."""
    import warp as wp
    from mujoco.mjx.warp.types import GraphMode
    from packaging.version import Version

    if Version(wp.__version__) < Version("1.10.0"):
        raise RuntimeError(
            f"warp-lang=={wp.__version__} is too old for multi-GPU training. "
            "warp-lang<1.10 can silently corrupt non-zero pmap replicas; "
            "upgrade warp-lang or set experiment.distributed.num_devices=1."
        )
    if not hasattr(GraphMode, "WARP"):
        raise RuntimeError(
            "This MuJoCo build does not provide GraphMode.WARP, which is required for multi-GPU MJX-Warp training."
        )


def _experiment_config(config: Any) -> Any:
    return getattr(config, "experiment", config)


def device_count(config: Any) -> int:
    """Resolve experiment.distributed.num_devices, where -1 means all visible devices."""
    exp = _experiment_config(config)
    dist_cfg = getattr(exp, "distributed", None)
    requested = 1 if dist_cfg is None else int(getattr(dist_cfg, "num_devices", 1))
    if requested == -1:
        requested = jax.local_device_count()
    if requested < 1:
        raise ValueError("experiment.distributed.num_devices must be >= 1 or -1 for all visible devices.")
    available = jax.local_device_count()
    if requested > available:
        raise ValueError(f"experiment.distributed.num_devices={requested} exceeds available devices ({available}).")
    return requested
