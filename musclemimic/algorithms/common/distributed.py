"""Helpers for single-process JAX data-parallel training."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

DATA_AXIS_NAME = "data"


def psum(value: jax.Array, axis_name: str | None) -> jax.Array:
    """Sum `value` across `axis_name` if distributed; otherwise return as-is."""
    return jax.lax.psum(value, axis_name=axis_name) if axis_name is not None else value


def pmean(value: jax.Array, axis_name: str | None) -> jax.Array:
    """Mean-reduce `value` across `axis_name` if distributed; otherwise return as-is."""
    return jax.lax.pmean(value, axis_name=axis_name) if axis_name is not None else value


def pmax(value: jax.Array, axis_name: str | None) -> jax.Array:
    """Max-reduce `value` across `axis_name` if distributed; otherwise return as-is."""
    return jax.lax.pmax(value, axis_name=axis_name) if axis_name is not None else value


def axis_size(axis_name: str | None) -> jax.Array:
    """Return the effective replica count for the distributed axis."""
    if axis_name is None:
        return jnp.asarray(1.0, dtype=jnp.float32)
    return jax.lax.psum(jnp.asarray(1.0, dtype=jnp.float32), axis_name=axis_name)


def make_data_mesh(
    num_devices: int,
    *,
    devices: list[jax.Device] | None = None,
) -> jax.sharding.Mesh:
    """Create the one-dimensional mesh used for data-parallel training."""
    devices = (devices or jax.local_devices())[:num_devices]
    if len(devices) != num_devices:
        raise ValueError(f"requested {num_devices} devices, got {len(devices)}")
    return jax.make_mesh(
        (num_devices,),
        (DATA_AXIS_NAME,),
        devices=devices,
    )


def mark_data_varying(
    tree: Any,
    axis_name: str | None,
    *,
    empty_reference: Any | None = None,
) -> Any:
    """Mark non-scalar shard-local arrays as varying over the data axis."""
    if axis_name is None:
        return tree
    # ``lax.pvary`` rejects values already marked as varying. This identity
    # selection is safe for both cases and makes the data-axis dependency
    # explicit to shard_map's variation checker.
    local_value = jax.lax.axis_index(axis_name) >= 0

    def mark(value: Any, reference: Any | None = None) -> Any:
        if value.ndim == 0:
            return value
        if value.size == 0:
            return value if reference is None else reference
        return jnp.where(local_value, value, value)

    if empty_reference is not None:
        return jax.tree.map(mark, tree, empty_reference)
    return jax.tree.map(mark, tree)


def assert_warp_multi_gpu_compatible() -> None:
    """Fail early for Warp versions known to produce bad multi-GPU results."""
    import warp as wp
    from packaging.version import Version

    if Version(wp.__version__) < Version("1.10.0"):
        raise RuntimeError(
            f"warp-lang=={wp.__version__} is too old for multi-GPU training. "
            "warp-lang<1.10 can silently corrupt non-zero device replicas; "
            "upgrade warp-lang or set experiment.distributed.num_devices=1."
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
