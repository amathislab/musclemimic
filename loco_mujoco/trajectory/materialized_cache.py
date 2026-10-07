from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from .dataclasses import Trajectory, TrajectoryCacheType

MATERIALIZED_CACHE_SCHEMA_VERSION = 3
_CACHE_METADATA_KEY = "materialized_cache"
MATERIALIZED_MOTION_NAMES_METADATA_KEY = "materialized_motion_names"


_MODEL_ARRAY_FIELDS = (
    "body_parentid",
    "body_rootid",
    "body_weldid",
    "body_mocapid",
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "body_mass",
    "body_inertia",
    "jnt_type",
    "jnt_bodyid",
    "jnt_qposadr",
    "jnt_dofadr",
    "jnt_pos",
    "jnt_axis",
    "dof_bodyid",
    "dof_jntid",
    "dof_parentid",
    "qpos0",
    "site_type",
    "site_bodyid",
    "site_pos",
    "site_quat",
)


def _model_trajectory_hash(model: mujoco.MjModel) -> str:
    """Hash compiled model fields that determine cached trajectory values."""
    digest = hashlib.sha256()
    dimensions = (model.nq, model.nv, model.njnt, model.nbody, model.nsite)
    digest.update(json.dumps(dimensions, separators=(",", ":")).encode())
    for object_type, count in (
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        (mujoco.mjtObj.mjOBJ_SITE, model.nsite),
    ):
        names = [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]
        digest.update(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode())
    for field in _MODEL_ARRAY_FIELDS:
        value = np.ascontiguousarray(getattr(model, field))
        digest.update(field.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def materialized_trajectory_fingerprint(
    motion_names: list[str],
    model: mujoco.MjModel,
    control_dt: float,
    cache_type: TrajectoryCacheType | str,
    site_names: list[str] | None,
    retargeting_method: str,
    model_spec: mujoco.MjSpec | None = None,
) -> str:
    """Return a stable fingerprint for one materialized trajectory cache."""
    # Keep model_spec in the public signature for compatibility. XML ordering,
    # generated asset paths, and unrelated visual assets do not determine the
    # qpos, qvel, body, or site values stored in a trajectory cache.
    del model_spec
    model_hash = _model_trajectory_hash(model)

    payload = {
        "schema_version": MATERIALIZED_CACHE_SCHEMA_VERSION,
        "motion_names": sorted(set(motion_names)),
        "model_sha256": model_hash,
        "control_dt": float(control_dt).hex(),
        "cache_type": TrajectoryCacheType(cache_type).value,
        "site_names": list(site_names or []),
        "retargeting_method": retargeting_method,
        "mujoco_version": mujoco.__version__,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def load_or_build_materialized_trajectory(
    cache_path: str | Path,
    fingerprint: str,
    build: Callable[[], Trajectory],
    rebuild: bool = False,
) -> Trajectory:
    """Load a materialized trajectory, or build and atomically save it when absent."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        trajectory = Trajectory.load(cache_path, backend=np)
        metadata = trajectory.info.metadata or {}
        cached_fingerprint = metadata.get(_CACHE_METADATA_KEY, {}).get("fingerprint")
        if cached_fingerprint != fingerprint:
            raise ValueError(
                f"Materialized trajectory cache fingerprint mismatch in {cache_path}: "
                f"expected {fingerprint}, got {cached_fingerprint!r}."
            )
        return trajectory

    trajectory = build()
    metadata = dict(trajectory.info.metadata or {})
    metadata[_CACHE_METADATA_KEY] = {
        "schema_version": MATERIALIZED_CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
    }
    trajectory = replace(trajectory, info=replace(trajectory.info, metadata=metadata))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.stem}.{uuid.uuid4().hex}.tmp.npz")
    try:
        trajectory.save(temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return trajectory
