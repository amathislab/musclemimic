"""Explicit device placement and optional partitioning for trajectories."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from omegaconf import open_dict

from musclemimic.algorithms.common.distributed import DATA_AXIS_NAME, make_data_mesh
from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__)


def trajectory_sharding_mode(config: Any) -> str:
    """Return the configured trajectory sharding mode."""
    experiment = getattr(config, "experiment", config)
    trajectory_config = getattr(experiment, "trajectory", {}) or {}
    mode = str(trajectory_config.get("sharding", "none") or "none").lower()
    if mode not in {"none", "static"}:
        raise ValueError("experiment.trajectory.sharding must be 'none' or 'static'.")
    return mode


@struct.dataclass
class ShardSpec:
    """A deterministic contiguous split of global trajectory IDs."""

    offsets: tuple[int, ...] = struct.field(pytree_node=False)
    counts: tuple[int, ...] = struct.field(pytree_node=False)
    n_global: int = struct.field(pytree_node=False)
    n_devices: int = struct.field(pytree_node=False)

    @classmethod
    def build(cls, n_traj_global: int, n_devices: int) -> ShardSpec:
        if n_traj_global <= 0:
            raise ValueError("n_traj_global must be positive.")
        if n_devices <= 0:
            raise ValueError("n_devices must be positive.")
        if n_traj_global < n_devices:
            raise ValueError(
                f"Cannot shard {n_traj_global} motions across {n_devices} devices (some devices would be empty)."
            )

        base, remainder = divmod(int(n_traj_global), int(n_devices))
        counts = tuple(base + (device_id < remainder) for device_id in range(n_devices))
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        return cls(tuple(offsets), counts, int(n_traj_global), int(n_devices))

    def local_range(self, device_id: int) -> tuple[int, int]:
        return self.offsets[device_id], self.offsets[device_id + 1]

    def local_count(self, device_id: int) -> int:
        return self.counts[device_id]

    def mask_for_device(self, device_id: jax.Array | int) -> jax.Array:
        offsets = jnp.asarray(self.offsets, dtype=jnp.int32)
        trajectory_ids = jnp.arange(self.n_global, dtype=jnp.int32)
        device_id = jnp.asarray(device_id, dtype=jnp.int32)
        start = offsets[device_id]
        stop = offsets[device_id + 1]
        return ((trajectory_ids >= start) & (trajectory_ids < stop)).astype(jnp.float32)

    def uniform_weights_for_device(self, device_id: jax.Array | int) -> jax.Array:
        mask = self.mask_for_device(device_id)
        return mask / mask.sum()


def apply_shard_mask_and_renormalize(
    weights: jax.Array,
    spec: ShardSpec,
    device_id: jax.Array | int,
) -> jax.Array:
    """Restrict a global sampling distribution to one device's shard."""
    masked_weights = weights * spec.mask_for_device(device_id)
    total = masked_weights.sum()
    uniform_weights = spec.uniform_weights_for_device(device_id)
    return jax.lax.cond(
        total > 1e-12,
        lambda: masked_weights / total,
        lambda: uniform_weights,
    )


@struct.dataclass
class TrajectoryShardPlan:
    """Static shape metadata for padded per-device trajectory shards."""

    spec: ShardSpec
    n_traj_per_shard: int = struct.field(pytree_node=False)
    n_samples_per_shard: int = struct.field(pytree_node=False)

    @property
    def n_traj_global(self) -> int:
        return self.spec.n_global

    @property
    def n_devices(self) -> int:
        return self.spec.n_devices


def shard_trajectory_data(
    data: Any,
    spec: ShardSpec,
    *,
    global_indexed_split_points: bool = False,
) -> tuple[list[Any], TrajectoryShardPlan]:
    """Split flat trajectory data into same-shaped, padded device shards.

    Global-indexed split points keep global trajectory IDs addressable. IDs
    outside a device's shard map to zero-length ranges.
    """
    global_split_points = _validate_sharding_input(data, spec)
    local_ranges = [spec.local_range(device_id) for device_id in range(spec.n_devices)]
    live_sample_counts = [int(global_split_points[stop] - global_split_points[start]) for start, stop in local_ranges]
    n_samples_per_shard = max(live_sample_counts)
    plan = TrajectoryShardPlan(
        spec=spec,
        n_traj_per_shard=max(spec.counts),
        n_samples_per_shard=n_samples_per_shard,
    )

    sample_fields, replicated_fields = _split_trajectory_fields(data, int(global_split_points[-1]))
    shards = []
    for device_id, (start_traj, stop_traj) in enumerate(local_ranges):
        sample_start = int(global_split_points[start_traj])
        sample_stop = int(global_split_points[stop_traj])
        live_samples = sample_stop - sample_start

        shard_values = {
            field: _slice_and_pad(getattr(data, field), sample_start, sample_stop, n_samples_per_shard)
            for field in sample_fields
        }
        shard_values.update({field: getattr(data, field) for field in replicated_fields})
        shard_values["split_points"] = _make_split_points(
            global_split_points,
            spec,
            device_id,
            sample_start,
            live_samples,
            global_indexed=global_indexed_split_points,
        )
        shards.append(data.__class__(**shard_values))

    return shards, plan


def _validate_sharding_input(data: Any, spec: ShardSpec) -> np.ndarray:
    """Validate the source trajectory layout before creating padded shards."""
    split_points = np.asarray(data.split_points)
    if split_points.ndim != 1 or not np.issubdtype(split_points.dtype, np.integer):
        raise ValueError("trajectory split_points must be a one-dimensional integer array.")
    if split_points.size != spec.n_global + 1:
        raise ValueError(
            f"trajectory split_points describe {split_points.size - 1} motions; the shard spec expects {spec.n_global}."
        )
    if split_points[0] != 0:
        raise ValueError("trajectory split_points must start at zero.")
    if np.any(np.diff(split_points) <= 0):
        raise ValueError("source trajectories must each contain at least one sample.")

    n_samples = int(split_points[-1])
    for field_name in ("qpos", "qvel"):
        values = np.asarray(getattr(data, field_name))
        if values.ndim < 1 or values.shape[0] != n_samples:
            raise ValueError(
                f"trajectory {field_name} has {values.shape[0] if values.ndim else 0} samples; "
                f"split_points describe {n_samples}."
            )
    return split_points


def _split_trajectory_fields(data: Any, global_n_samples: int) -> tuple[list[str], list[str]]:
    sample_fields = []
    replicated_fields = []
    for field in data.__dataclass_fields__:
        if field == "split_points":
            continue
        value = getattr(data, field)
        array = np.asarray(value) if value is not None else None
        if array is not None and array.ndim >= 1 and array.shape[0] == global_n_samples:
            sample_fields.append(field)
        else:
            replicated_fields.append(field)
    return sample_fields, replicated_fields


def _slice_and_pad(value: Any, start: int, stop: int, target_len: int) -> np.ndarray:
    live_values = np.asarray(value)[start:stop]
    pad_len = target_len - live_values.shape[0]
    if pad_len <= 0:
        return live_values
    padding = np.zeros((pad_len, *live_values.shape[1:]), dtype=live_values.dtype)
    return np.concatenate([live_values, padding], axis=0)


def _make_split_points(
    global_split_points: np.ndarray,
    spec: ShardSpec,
    device_id: int,
    sample_start: int,
    live_samples: int,
    *,
    global_indexed: bool,
) -> np.ndarray:
    start_traj, stop_traj = spec.local_range(device_id)

    if global_indexed:
        local_split_points = np.zeros(spec.n_global + 1, dtype=global_split_points.dtype)
        for trajectory_id in range(start_traj, stop_traj):
            local_split_points[trajectory_id] = int(global_split_points[trajectory_id]) - sample_start
        local_split_points[stop_traj:] = live_samples
        return local_split_points

    local_split_points = np.zeros(spec.local_count(device_id) + 1, dtype=global_split_points.dtype)
    for local_id, trajectory_id in enumerate(range(start_traj, stop_traj)):
        local_split_points[local_id] = int(global_split_points[trajectory_id]) - sample_start
    local_split_points[-1] = live_samples

    target_len = max(spec.counts) + 1
    return np.pad(
        local_split_points,
        (0, target_len - local_split_points.shape[0]),
        mode="edge",
    )


def assemble_global_device_shards(
    per_device_trees: list[Any],
    devices: list[jax.Device] | None = None,
) -> Any:
    """Assemble equal-shaped local pytrees into an axis-0 sharded array."""
    devices = devices or jax.local_devices()
    if len(per_device_trees) != len(devices):
        raise ValueError(f"got {len(per_device_trees)} trees for {len(devices)} devices")

    mesh = make_data_mesh(len(devices), devices=devices)
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(DATA_AXIS_NAME),
    )

    def assemble(*leaves):
        host_leaves = [np.asarray(leaf) for leaf in leaves]
        shapes = {leaf.shape for leaf in host_leaves}
        if len(shapes) != 1:
            raise ValueError(f"device shards have mismatched shapes {shapes}")
        local_shape = host_leaves[0].shape
        if not local_shape:
            raise ValueError("device-sharded trajectory leaves must have an array axis")
        device_arrays = [jax.device_put(leaf, device) for leaf, device in zip(host_leaves, devices, strict=True)]
        global_shape = (len(devices) * local_shape[0], *local_shape[1:])
        return jax.make_array_from_single_device_arrays(
            global_shape,
            sharding,
            device_arrays,
        )

    return jax.tree_util.tree_map(assemble, *per_device_trees)


def _replicate_trajectory_data(
    data: Any,
    devices: list[jax.Device],
) -> Any:
    """Place a full trajectory library on every selected device."""
    mesh = make_data_mesh(len(devices), devices=devices)
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )
    return jax.device_put(data, sharding)


def prepare_trajectory_data_shards(env: Any, config: Any, n_devices: int, *, label: str = "training") -> Any:
    """Place trajectory data as an explicit replicated or partitioned input."""
    mode = trajectory_sharding_mode(config)
    if env.th is None or env.th.traj is None:
        if mode == "static":
            raise ValueError("experiment.trajectory.sharding=static requires an environment with a trajectory handler.")
        return None

    n_trajectories = int(env.th.n_trajectories)
    devices = jax.local_devices()[:n_devices]
    if len(devices) != n_devices:
        raise ValueError(f"Requested {n_devices} devices, but only {len(jax.local_devices())} are available.")
    if not getattr(env.th, "is_numpy", True):
        env.th.to_numpy()
    host_data = jax.device_get(env.th.traj.data)

    if mode == "none":
        env.th.traj = replace(env.th.traj, data=host_data)
        env.th._is_numpy = True
        device_data = _replicate_trajectory_data(host_data, devices=devices)
        logger.info(
            f"[trajectory] {label} replicated trajectory library: n_devices={n_devices}, n_traj={n_trajectories}"
        )
        return device_data

    spec = ShardSpec.build(n_trajectories, n_devices)
    shards, plan = shard_trajectory_data(
        host_data,
        spec,
        global_indexed_split_points=True,
    )
    # Keep the Python environment's resident view on the host. During tracing,
    # ``install_trajectory_data_overrides`` installs the matching
    # device shard for both environment stepping and validation metrics.
    env.th.traj = replace(env.th.traj, data=shards[0])
    env.th._is_numpy = True

    device_shards = assemble_global_device_shards(
        shards,
        devices=devices,
    )
    logger.info(
        f"[trajectory] {label} static trajectory library: "
        f"n_devices={n_devices}, n_traj={n_trajectories}, "
        f"per-device motion counts={list(spec.counts)}, "
        f"padded_samples_per_device={plan.n_samples_per_shard}"
    )
    return device_shards


def prepare_validation_trajectory_data_shards(
    val_env: Any,
    config: Any,
    n_devices: int,
    *,
    shares_training: bool,
) -> Any:
    """Place a distinct validation library using the training layout."""
    if val_env is None or shares_training:
        return None

    if trajectory_sharding_mode(config) == "static":
        with open_dict(config.experiment.validation):
            config.experiment.validation.n_trajectories = int(val_env.th.n_trajectories)
    return prepare_trajectory_data_shards(val_env, config, n_devices, label="validation")


def _set_trajectory_data(env: Any, trajectory_data: Any) -> tuple[Any, Any, bool]:
    restore_token = (env, env.th.traj, env.th._is_numpy)
    env.th.traj = replace(env.th.traj, data=trajectory_data)
    env.th._is_numpy = False
    return restore_token


def install_trajectory_data_overrides(
    env: Any,
    val_env: Any,
    trajectory_data_override: Any,
    validation_trajectory_data_override: Any,
) -> list[tuple[Any, Any, bool]]:
    """Install placed training and validation data before wrapping."""
    restore_tokens = []
    trajectories_are_shared = val_env is not None and val_env.th.traj is env.th.traj

    if trajectory_data_override is not None:
        restore_tokens.append(_set_trajectory_data(env, trajectory_data_override))
        if trajectories_are_shared:
            restore_tokens.append((val_env, val_env.th.traj, val_env.th._is_numpy))
            val_env.th.traj = env.th.traj
            val_env.th._is_numpy = False

    if validation_trajectory_data_override is not None:
        if trajectories_are_shared:
            raise ValueError("A shared validation trajectory cannot have a separate data input.")
        restore_tokens.append(_set_trajectory_data(val_env, validation_trajectory_data_override))

    return restore_tokens


def restore_trajectory_data_overrides(restore_tokens: list[tuple[Any, Any, bool]]) -> None:
    """Restore Python environment objects after tracing installs placed data."""
    for env, old_trajectory, old_is_numpy in reversed(restore_tokens):
        env.th.traj = old_trajectory
        env.th._is_numpy = old_is_numpy


@contextmanager
def with_trajectory_data_overrides(
    env: Any,
    val_env: Any,
    trajectory_data_override: Any,
    validation_trajectory_data_override: Any,
):
    """Temporarily install explicit trajectory data during tracing."""
    restore_tokens = install_trajectory_data_overrides(
        env,
        val_env,
        trajectory_data_override,
        validation_trajectory_data_override,
    )
    try:
        yield
    finally:
        restore_trajectory_data_overrides(restore_tokens)
