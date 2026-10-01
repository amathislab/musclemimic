"""Bounded host-side evidence for multi-device PPO hardware gates."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__)

_EVIDENCE_FILENAME = "hardware_gate_evidence.jsonl"
_MAX_STATE_SAMPLE_VALUES = 4096
_MAX_IDENTITY_VALUES = 256
_MAX_NONFINITE_METRIC_LEAVES = 32


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _bounded_positive(value: Any, *, name: str, maximum: int) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}, got {parsed}")
    return parsed


def _key_path(path: Sequence[Any]) -> str:
    components = []
    for key in path:
        if hasattr(key, "key"):
            components.append(str(key.key))
        elif hasattr(key, "idx"):
            components.append(str(key.idx))
        elif hasattr(key, "name"):
            components.append(str(key.name))
        else:
            components.append(str(key))
    return ".".join(components)


def _native_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _metric_finiteness_summary(tree: Any) -> dict[str, Any]:
    """Return exact numeric finiteness counts without copying metric arrays.

    Metric leaves are reduced on device and only scalar counts are transferred
    to the host.  Aggregate counts cover every numeric value; the diagnostic
    list of offending leaf paths is capped so the JSONL record remains bounded.
    """
    if tree is None:
        return {"available": False, "reason": "metric tree is unavailable"}

    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    numeric_leaves: list[tuple[str, int, Any | None]] = []
    skipped_leaf_count = 0
    for path, value in leaves_with_paths:
        if value is None:
            skipped_leaf_count += 1
            continue
        try:
            dtype = np.dtype(value.dtype) if hasattr(value, "dtype") else np.asarray(value).dtype
            raw_shape = value.shape if hasattr(value, "shape") else np.asarray(value).shape
            shape = tuple(int(dimension) for dimension in raw_shape)
        except (TypeError, ValueError):
            skipped_leaf_count += 1
            continue
        if dtype.kind not in "biufc":
            skipped_leaf_count += 1
            continue

        value_count = int(np.prod(shape, dtype=np.int64))
        reductions = None
        if dtype.kind in "fc" and value_count:
            array = jnp.asarray(value)
            reductions = jnp.stack(
                (
                    jnp.count_nonzero(jnp.isfinite(array)),
                    jnp.count_nonzero(jnp.isnan(array)),
                    jnp.count_nonzero(jnp.isinf(array)),
                )
            )
        numeric_leaves.append((_key_path(path) or "<root>", value_count, reductions))

    if not numeric_leaves:
        return {
            "available": False,
            "reason": "metric tree has no numeric leaves",
            "skipped_leaf_count": skipped_leaf_count,
        }

    pending_reductions = tuple(reductions for _, _, reductions in numeric_leaves if reductions is not None)
    host_reductions = iter(jax.device_get(pending_reductions))
    numeric_value_count = 0
    finite_value_count = 0
    nan_value_count = 0
    infinity_value_count = 0
    nonfinite_leaf_count = 0
    nonfinite_leaves_sample = []
    for path, value_count, reductions in numeric_leaves:
        if reductions is None:
            finite_count = value_count
            nan_count = 0
            infinity_count = 0
        else:
            finite_count, nan_count, infinity_count = (int(count) for count in np.asarray(next(host_reductions)))
        nonfinite_count = value_count - finite_count
        numeric_value_count += value_count
        finite_value_count += finite_count
        nan_value_count += nan_count
        infinity_value_count += infinity_count
        if nonfinite_count:
            nonfinite_leaf_count += 1
            if len(nonfinite_leaves_sample) < _MAX_NONFINITE_METRIC_LEAVES:
                nonfinite_leaves_sample.append(
                    {
                        "path": path,
                        "value_count": value_count,
                        "nonfinite_value_count": nonfinite_count,
                        "nan_value_count": nan_count,
                        "infinity_value_count": infinity_count,
                    }
                )

    nonfinite_value_count = numeric_value_count - finite_value_count
    return {
        "available": True,
        "all_finite": nonfinite_value_count == 0,
        "numeric_leaf_count": len(numeric_leaves),
        "numeric_value_count": numeric_value_count,
        "finite_value_count": finite_value_count,
        "nonfinite_value_count": nonfinite_value_count,
        "nan_value_count": nan_value_count,
        "infinity_value_count": infinity_value_count,
        "nonfinite_leaf_count": nonfinite_leaf_count,
        "nonfinite_leaves_sample": nonfinite_leaves_sample,
        "nonfinite_leaves_truncated": nonfinite_leaf_count > len(nonfinite_leaves_sample),
        "skipped_leaf_count": skipped_leaf_count,
    }


def _device_attribute(device: Any, name: str) -> Any:
    try:
        value = getattr(device, name)
        return _native_value(value() if callable(value) else value)
    except Exception:
        return None


def _device_identity(replica_index: int, device: Any) -> dict[str, Any]:
    return {
        "replica_index": replica_index,
        "device_id": _device_attribute(device, "id"),
        "process_index": _device_attribute(device, "process_index"),
        "platform": _device_attribute(device, "platform"),
        "device_kind": _device_attribute(device, "device_kind"),
        "description": str(device),
    }


def _hash_array(hasher: Any, path: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    header = {
        "path": path,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
    }
    hasher.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(contiguous.tobytes(order="C"))
    hasher.update(b"\0")


def _local_replica_array(value: Any, replica_index: int, num_replicas: int) -> np.ndarray:
    host_value = np.asarray(jax.device_get(value))
    if num_replicas == 1:
        return host_value
    if host_value.ndim == 0 or host_value.shape[0] != num_replicas:
        raise ValueError(f"expected a leading replica axis of size {num_replicas}, got shape {host_value.shape}")
    return np.asarray(host_value[replica_index])


def _exact_replica_digests(tree: Any, num_replicas: int) -> dict[str, Any]:
    """Hash every byte in a replicated pytree after transferring it to the host."""
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    if not leaves_with_paths:
        return {"available": False, "reason": "pytree has no array leaves"}

    hashers = [hashlib.sha256() for _ in range(num_replicas)]
    try:
        for path, value in leaves_with_paths:
            path_text = _key_path(path)
            for replica_index, hasher in enumerate(hashers):
                _hash_array(
                    hasher,
                    path_text,
                    _local_replica_array(value, replica_index, num_replicas),
                )
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    return {
        "available": True,
        "algorithm": "sha256",
        "coverage": "all_leaves_all_values",
        "leaf_count": len(leaves_with_paths),
        "digests": [hasher.hexdigest() for hasher in hashers],
    }


def _sample_array(value: Any, num_replicas: int, indices: np.ndarray) -> np.ndarray:
    shape = tuple(int(dimension) for dimension in value.shape)
    if num_replicas > 1:
        if not shape or shape[0] != num_replicas:
            raise ValueError(f"expected a leading replica axis of size {num_replicas}, got shape {shape}")
        local_size = int(np.prod(shape[1:], dtype=np.int64))
        flattened = jnp.reshape(value, (num_replicas, local_size))
    else:
        local_size = int(np.prod(shape, dtype=np.int64))
        flattened = jnp.reshape(value, (1, local_size))
    if np.any(indices < 0) or np.any(indices >= local_size):
        raise ValueError(f"sample indices exceed local array size {local_size}")
    sampled = jnp.take(flattened, jnp.asarray(indices), axis=1)
    return np.asarray(jax.device_get(sampled))


def _sampled_replica_digests(
    tree: Any,
    num_replicas: int,
    max_values: int,
) -> dict[str, Any]:
    """Hash a deterministic sample capped across the entire pytree."""
    leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    if not leaves_with_paths:
        return {"available": False, "reason": "pytree has no array leaves"}

    sampleable = []
    skipped: list[str] = []
    total_values = 0
    for path, value in leaves_with_paths:
        path_text = _key_path(path)
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            skipped.append(path_text)
            continue
        shape = tuple(int(dimension) for dimension in value.shape)
        if num_replicas > 1:
            if not shape or shape[0] != num_replicas:
                skipped.append(f"{path_text}: missing replica axis")
                continue
            local_size = int(np.prod(shape[1:], dtype=np.int64))
        else:
            local_size = int(np.prod(shape, dtype=np.int64))
        if local_size == 0:
            skipped.append(f"{path_text}: empty")
            continue
        sampleable.append((path_text, value, local_size, total_values))
        total_values += local_size

    if not sampleable or total_values == 0:
        reason = "no sampleable replicated array leaves"
        if skipped:
            reason += f"; skipped={skipped[:4]}"
        return {"available": False, "reason": reason}

    global_indices = np.unique(
        np.linspace(
            0,
            total_values - 1,
            min(total_values, max_values),
            dtype=np.int64,
        )
    )
    hashers = [hashlib.sha256() for _ in range(num_replicas)]
    sampled_leaves = 0
    sampled_values = 0
    for path_text, value, local_size, offset in sampleable:
        local_indices = global_indices[(global_indices >= offset) & (global_indices < offset + local_size)] - offset
        if local_indices.size == 0:
            continue
        try:
            sampled = _sample_array(value, num_replicas, local_indices)
        except Exception as error:
            skipped.append(f"{path_text}: {type(error).__name__}")
            continue
        for replica_index, hasher in enumerate(hashers):
            sampled_path = f"{path_text}@{','.join(str(index) for index in local_indices)}"
            _hash_array(hasher, sampled_path, sampled[replica_index])
        sampled_leaves += 1
        sampled_values += int(local_indices.size)

    if sampled_leaves == 0:
        reason = "no sampleable replicated array leaves"
        if skipped:
            reason += f"; skipped={skipped[:4]}"
        return {"available": False, "reason": reason}

    return {
        "available": True,
        "algorithm": "sha256",
        "coverage": "deterministic_even_sample_across_flattened_leaves",
        "max_values_per_replica": max_values,
        "total_values_per_replica": total_values,
        "sampled_leaf_count": sampled_leaves,
        "sampled_values_per_replica": sampled_values,
        "sample_fraction": sampled_values / total_values,
        "skipped_leaf_count": len(skipped),
        "digests": [hasher.hexdigest() for hasher in hashers],
    }


def _parameter_equality(tree: Any, num_replicas: int) -> dict[str, Any]:
    """Compare every replicated parameter value on device and return one host bool."""
    if num_replicas == 1:
        return {"available": True, "all_replicas_equal": True, "coverage": "all_parameter_values"}
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return {"available": False, "reason": "parameter pytree has no leaves"}

    def compare(parameters):
        parameter_leaves = jax.tree_util.tree_leaves(parameters)
        comparisons = [jnp.all(value == value[0]) for value in parameter_leaves]
        return jnp.all(jnp.stack(comparisons))

    try:
        all_equal = bool(np.asarray(jax.device_get(jax.jit(compare)(tree))))
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    return {
        "available": True,
        "all_replicas_equal": all_equal,
        "coverage": "all_parameter_values",
        "method": "device_side_exact_comparison",
        "leaf_count": len(leaves),
    }


def _find_state_field(state: Any, name: str) -> Any | None:
    """Find a named field through common wrapper and carry structures."""
    queue = [state]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Mapping):
            if name in current:
                return current[name]
            queue.extend(current.values())
            continue
        try:
            if hasattr(current, name):
                return getattr(current, name)
        except Exception:
            pass
        for child_name in ("env_state", "additional_carry", "traj_state", "data"):
            try:
                child = getattr(current, child_name)
            except Exception:
                continue
            queue.append(child)
    return None


def _environment_probe(runtime_state: Any) -> dict[str, Any]:
    """Select small state fields that establish per-replica environment diversity."""
    env_state = getattr(runtime_state, "env_state", None)
    probe = {"last_obs": getattr(runtime_state, "last_obs", None)}
    for name in (
        "observation",
        "qpos",
        "qvel",
        "traj_no",
        "selected_traj_idx",
        "key",
        "autoreset_rng",
    ):
        value = _find_state_field(env_state, name)
        if value is not None:
            probe[name] = value
    return {key: value for key, value in probe.items() if value is not None}


def _identity_evidence(
    value: Any | None,
    num_replicas: int,
    max_values: int,
    *,
    source: str,
) -> dict[str, Any]:
    if value is None:
        return {"available": False, "reason": f"{source} is unavailable"}
    try:
        replicas = []
        for replica_index in range(num_replicas):
            local = _local_replica_array(value, replica_index, num_replicas).reshape(-1)
            digest = hashlib.sha256()
            _hash_array(digest, source, local)
            unique = np.unique(local)
            replicas.append(
                {
                    "sha256": digest.hexdigest(),
                    "value_count": int(local.size),
                    "values_sample": [_native_value(item) for item in local[:max_values]],
                    "unique_count": int(unique.size),
                    "unique_values_sample": [_native_value(item) for item in unique[:max_values]],
                    "sample_truncated": bool(local.size > max_values or unique.size > max_values),
                }
            )
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    return {
        "available": True,
        "source": source,
        "replicas": replicas,
    }


def _find_environment_attribute(env: Any, name: str) -> Any | None:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        try:
            value = getattr(current, name)
        except Exception:
            value = None
        if value is not None:
            return value
        try:
            current = current.env
        except Exception:
            current = None
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_sequence(env: Any, plural_name: str, singular_name: str) -> tuple[Path, ...]:
    paths = _find_environment_attribute(env, plural_name)
    if paths is None:
        path = _find_environment_attribute(env, singular_name)
        return () if path is None else (Path(path).expanduser(),)
    if isinstance(paths, (str, os.PathLike)):
        return (Path(paths).expanduser(),)
    return tuple(Path(path).expanduser() for path in paths)


def _paired_artifact_catalog(env: Any) -> dict[str, Any]:
    motion_paths = _path_sequence(env, "paired_motion_paths", "paired_motion_path")
    terrain_paths = _path_sequence(env, "paired_terrain_paths", "paired_terrain_path")
    if not motion_paths or not terrain_paths:
        return {
            "available": False,
            "reason": "environment exposes no paired_motion_paths/paired_terrain_paths catalog",
        }
    if len(motion_paths) != len(terrain_paths):
        return {
            "available": False,
            "reason": (f"paired motion/terrain catalog lengths differ: {len(motion_paths)} != {len(terrain_paths)}"),
        }
    manifest = [
        {"trajectory_id": index, "motion_path": str(motion), "terrain_path": str(terrain)}
        for index, (motion, terrain) in enumerate(zip(motion_paths, terrain_paths, strict=True))
    ]
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "available": True,
        "source": "env.paired_motion_paths/env.paired_terrain_paths",
        "trajectory_count": len(manifest),
        "path_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "motion_paths": motion_paths,
        "terrain_paths": terrain_paths,
    }


def _cuda_device_memory_snapshot(device_id: Any) -> dict[str, Any]:
    try:
        import warp as wp

        device = wp.get_device(f"cuda:{int(device_id)}")
        total = int(device.total_memory)
        free = int(device.free_memory)
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    return {
        "available": True,
        "source": "CUDA whole-device free/total memory queried through warp.Device",
        "total_bytes": total,
        "free_bytes": free,
        "used_bytes_snapshot": max(0, total - free),
    }


def _memory_evidence(
    devices: Sequence[Any],
    observed_total_device_used: dict[int, int],
) -> dict[str, Any]:
    replicas = []
    peak_values = []
    for replica_index, device in enumerate(devices):
        platform = _device_attribute(device, "platform")
        entry: dict[str, Any] = {
            "replica_index": replica_index,
            "device_id": _device_attribute(device, "id"),
            "platform": platform,
            "available": False,
        }
        if platform in {"gpu", "cuda"} and entry["device_id"] is not None:
            cuda_memory = _cuda_device_memory_snapshot(entry["device_id"])
            entry["cuda_device_memory_snapshot"] = cuda_memory
            used = cuda_memory.get("used_bytes_snapshot")
            if isinstance(used, int):
                observed_total_device_used[replica_index] = max(
                    observed_total_device_used.get(replica_index, 0),
                    used,
                )
                cuda_memory["peak_observed_used_bytes"] = observed_total_device_used[replica_index]
        else:
            entry["cuda_device_memory_snapshot"] = {
                "available": False,
                "reason": "replica is not a JAX GPU device",
            }
        try:
            stats_method = device.memory_stats
            stats = stats_method()
        except Exception as error:
            entry["reason"] = f"{type(error).__name__}: {error}"
            replicas.append(entry)
            continue
        if not isinstance(stats, Mapping):
            entry["reason"] = "jax.Device.memory_stats returned no mapping"
            replicas.append(entry)
            continue
        numeric_stats = {
            str(key): _native_value(value)
            for key, value in stats.items()
            if isinstance(value, (bool, int, float, np.generic))
        }
        entry.update({"available": True, "stats": numeric_stats})
        peak = numeric_stats.get("peak_bytes_in_use")
        if isinstance(peak, (int, float)):
            peak_values.append(int(peak))
        replicas.append(entry)

    return {
        "source": "jax.Device.memory_stats",
        "scope": "jax_allocator_only",
        "peak_bytes_in_use": max(peak_values) if peak_values else None,
        "replicas": replicas,
        "external_total_gpu_memory_required": True,
        "external_metric_reason": (
            "JAX reports its allocator peak and CUDA free/total is sampled only at chunk boundaries, "
            "not continuously; capture RunAI/W&B system metrics or nvidia-smi for the external total peak."
        ),
    }


class HardwareGateEvidenceRecorder:
    """Append durable, bounded hardware-gate records from the host coordinator."""

    def __init__(
        self,
        path: Path,
        devices: Sequence[Any],
        env: Any,
        *,
        resumed: bool,
        max_state_sample_values: int,
        max_identity_values: int,
    ) -> None:
        self.path = path
        self.devices = tuple(devices)
        self.num_replicas = len(self.devices)
        self.resumed = resumed
        self.max_state_sample_values = max_state_sample_values
        self.max_identity_values = max_identity_values
        self.session_id = uuid.uuid4().hex
        self._chunk_count = 0
        self._post_update_recorded = False
        self._paired_artifacts = _paired_artifact_catalog(env)
        self._paired_artifact_cache: dict[int, dict[str, Any]] = {}
        self._observed_total_device_used: dict[int, int] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        config: Any,
        devices: Sequence[Any],
        env: Any,
        *,
        resumed: bool,
    ) -> HardwareGateEvidenceRecorder | None:
        debug_config = _config_value(config, "debug", {})
        hardware_config = _config_value(debug_config, "hardware_gate", {})
        if isinstance(hardware_config, bool):
            enabled = hardware_config
            hardware_config = {}
        else:
            enabled = bool(_config_value(hardware_config, "enabled", False))
        if not enabled:
            return None

        configured_path = _config_value(hardware_config, "output_path")
        checkpoint_dir = _config_value(config, "checkpoint_dir")
        if configured_path:
            path = Path(str(configured_path)).expanduser()
            if not path.is_absolute() and checkpoint_dir:
                path = Path(str(checkpoint_dir)).expanduser() / path
        elif checkpoint_dir:
            path = Path(str(checkpoint_dir)).expanduser() / _EVIDENCE_FILENAME
        else:
            raise ValueError(
                "debug.hardware_gate.enabled=true requires checkpoint_dir or debug.hardware_gate.output_path"
            )

        max_state_sample_values = _bounded_positive(
            _config_value(hardware_config, "max_state_sample_values", 64),
            name="debug.hardware_gate.max_state_sample_values",
            maximum=_MAX_STATE_SAMPLE_VALUES,
        )
        max_identity_values = _bounded_positive(
            _config_value(hardware_config, "max_identity_values", 16),
            name="debug.hardware_gate.max_identity_values",
            maximum=_MAX_IDENTITY_VALUES,
        )
        return cls(
            path,
            devices,
            env,
            resumed=resumed,
            max_state_sample_values=max_state_sample_values,
            max_identity_values=max_identity_values,
        )

    def _emit(self, record: dict[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "resumed": self.resumed,
            **record,
        }
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        encoded = f"{line}\n".encode()
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("hardware-gate evidence append wrote zero bytes")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        logger.info("PPO hardware gate evidence: %s", line)

    def _paired_artifact(self, trajectory_id: Any) -> dict[str, Any] | None:
        if not self._paired_artifacts.get("available", False):
            return None
        try:
            index = int(trajectory_id)
        except (TypeError, ValueError, OverflowError):
            return None
        motion_paths = self._paired_artifacts["motion_paths"]
        terrain_paths = self._paired_artifacts["terrain_paths"]
        if index < 0 or index >= len(motion_paths):
            return {
                "trajectory_id": index,
                "available": False,
                "reason": f"trajectory id is outside paired catalog [0, {len(motion_paths)})",
            }
        if index in self._paired_artifact_cache:
            return self._paired_artifact_cache[index]

        motion_path = motion_paths[index]
        terrain_path = terrain_paths[index]
        artifact: dict[str, Any] = {
            "trajectory_id": index,
            "available": motion_path.is_file() and terrain_path.is_file(),
            "motion_path": str(motion_path),
            "terrain_path": str(terrain_path),
        }
        if not artifact["available"]:
            artifact["reason"] = "paired motion or terrain path is not a file"
        else:
            try:
                artifact["terrain_sha256"] = _file_sha256(terrain_path)
            except OSError as error:
                artifact["available"] = False
                artifact["reason"] = f"{type(error).__name__}: {error}"
        self._paired_artifact_cache[index] = artifact
        return artifact

    def _attach_paired_artifacts(self, trajectory: dict[str, Any]) -> None:
        if not trajectory.get("available", False):
            return
        for replica in trajectory.get("replicas", []):
            artifacts = []
            for trajectory_id in replica.get("unique_values_sample", []):
                artifact = self._paired_artifact(trajectory_id)
                if artifact is not None:
                    artifacts.append(artifact)
            replica["paired_artifacts_sample"] = artifacts

    def _paired_artifact_summary(self) -> dict[str, Any]:
        if not self._paired_artifacts.get("available", False):
            return dict(self._paired_artifacts)
        return {
            "available": True,
            "source": self._paired_artifacts["source"],
            "trajectory_count": self._paired_artifacts["trajectory_count"],
            "path_manifest_sha256": self._paired_artifacts["path_manifest_sha256"],
        }

    def record_state(
        self,
        runtime_state: Any,
        *,
        phase: str,
        update_number: int,
        global_timestep: int,
    ) -> None:
        parameter_equality = _parameter_equality(runtime_state.train_state.params, self.num_replicas)
        parameter = _sampled_replica_digests(
            runtime_state.train_state.params,
            self.num_replicas,
            self.max_state_sample_values,
        )
        rng = _exact_replica_digests(getattr(runtime_state, "rng", {}), self.num_replicas)
        environment = _sampled_replica_digests(
            _environment_probe(runtime_state),
            self.num_replicas,
            self.max_state_sample_values,
        )
        env_state = getattr(runtime_state, "env_state", None)
        trajectory_value = _find_state_field(env_state, "traj_no")
        trajectory_source = "env_state.additional_carry.traj_state.traj_no"
        if trajectory_value is None:
            trajectory_value = _find_state_field(env_state, "selected_traj_idx")
            trajectory_source = "env_state.additional_carry.selected_traj_idx"
        trajectory = _identity_evidence(
            trajectory_value,
            self.num_replicas,
            self.max_identity_values,
            source=trajectory_source,
        )
        self._attach_paired_artifacts(trajectory)
        terrain = _sampled_replica_digests(
            _find_state_field(env_state, "terrain_state"),
            self.num_replicas,
            self.max_state_sample_values,
        )

        def digests(component: Mapping[str, Any]) -> list[str]:
            values = component.get("digests", [])
            return list(values) if isinstance(values, Sequence) else []

        parameter_digests = digests(parameter)
        rng_digests = digests(rng)
        environment_digests = digests(environment)
        terrain_digests = digests(terrain)
        trajectory_replicas = trajectory.get("replicas", []) if trajectory.get("available") else []
        replicas = []
        for replica_index, device in enumerate(self.devices):
            replica = {
                "replica_index": replica_index,
                "device": _device_identity(replica_index, device),
                "parameter_fingerprint_sha256": (
                    parameter_digests[replica_index] if len(parameter_digests) == self.num_replicas else None
                ),
                "rng_sha256": rng_digests[replica_index] if len(rng_digests) == self.num_replicas else None,
                "environment_state_sample_sha256": (
                    environment_digests[replica_index] if len(environment_digests) == self.num_replicas else None
                ),
                "terrain_state_sample_sha256": (
                    terrain_digests[replica_index] if len(terrain_digests) == self.num_replicas else None
                ),
                "trajectory_identity": (
                    trajectory_replicas[replica_index] if len(trajectory_replicas) == self.num_replicas else None
                ),
            }
            replicas.append(replica)

        parameter_fingerprints_equal = bool(parameter_digests) and len(set(parameter_digests)) == 1
        parameters_equal = bool(
            parameter_equality.get("available", False) and parameter_equality.get("all_replicas_equal", False)
        )
        rng_distinct = len(rng_digests) == self.num_replicas and len(set(rng_digests)) == self.num_replicas
        environment_distinct = (
            len(environment_digests) == self.num_replicas and len(set(environment_digests)) == self.num_replicas
        )
        self._emit(
            {
                "record_type": "replica_state",
                "phase": phase,
                "update_number": int(update_number),
                "global_timestep": int(global_timestep),
                "replica_count": self.num_replicas,
                "parameters_equal": parameters_equal,
                "parameter_fingerprints_equal": parameter_fingerprints_equal,
                "rng_digests_distinct": rng_distinct,
                "environment_state_digests_distinct": environment_distinct,
                "components": {
                    "parameter_equality": parameter_equality,
                    "parameter_fingerprint": parameter,
                    "rng": rng,
                    "environment_state": environment,
                    "trajectory_identity": trajectory,
                    "terrain_state": terrain,
                    "paired_artifact_catalog": self._paired_artifact_summary(),
                },
                "replicas": replicas,
            }
        )

    def record_post_update_once(
        self,
        runtime_state: Any,
        *,
        update_number: int,
        global_timestep: int,
    ) -> None:
        if self._post_update_recorded:
            return
        phase = "post_resume_update" if self.resumed else "post_update"
        self.record_state(
            runtime_state,
            phase=phase,
            update_number=update_number,
            global_timestep=global_timestep,
        )
        self._post_update_recorded = True

    def record_chunk(
        self,
        *,
        chunk_updates: int,
        environment_steps: int,
        elapsed_seconds: float,
        update_number: int,
        global_timestep: int,
        validation_included: bool,
        training_metrics: Any,
        ppo_metrics: Any,
        diagnostics: Any,
        validation_metrics: Any | None,
    ) -> None:
        self._chunk_count += 1
        safe_elapsed = max(float(elapsed_seconds), 1e-9)
        metric_finiteness = {
            "scope": "host_chunk_boundary_final_update_all_replicas",
            "training_metrics": _metric_finiteness_summary(training_metrics),
            "ppo_metrics": _metric_finiteness_summary(ppo_metrics),
            "diagnostics": _metric_finiteness_summary(diagnostics),
            "validation_metrics": (
                _metric_finiteness_summary(validation_metrics)
                if validation_included
                else {
                    "available": False,
                    "reason": "validation was not run at this chunk boundary",
                }
            ),
        }
        required_summaries = (
            metric_finiteness["training_metrics"],
            metric_finiteness["ppo_metrics"],
            metric_finiteness["diagnostics"],
        )
        validation_summary = metric_finiteness["validation_metrics"]
        metric_finiteness["all_finite"] = all(
            summary.get("available", False) and summary.get("all_finite", False) for summary in required_summaries
        ) and (
            not validation_included
            or (validation_summary.get("available", False) and validation_summary.get("all_finite", False))
        )
        self._emit(
            {
                "record_type": "chunk_performance",
                "chunk_index": self._chunk_count,
                "chunk_updates": int(chunk_updates),
                "environment_steps": int(environment_steps),
                "elapsed_seconds": float(elapsed_seconds),
                "updates_per_second": float(chunk_updates) / safe_elapsed,
                "environment_steps_per_second": float(environment_steps) / safe_elapsed,
                "update_number": int(update_number),
                "global_timestep": int(global_timestep),
                "validation_included": bool(validation_included),
                "metric_finiteness": metric_finiteness,
                "compilation_may_be_included": self._chunk_count == 1,
                "gpu_memory": _memory_evidence(self.devices, self._observed_total_device_used),
            }
        )
