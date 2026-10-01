"""Persistent, content-addressed caches for materialized trajectories."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from .dataclasses import Trajectory, TrajectoryCacheType

MATERIALIZED_TRAJECTORY_CACHE_VERSION = 1
_CACHE_KEY = re.compile(r"[0-9a-f]{64}")
_MANIFEST_NAME = "manifest.json"
_TRAJECTORY_NAME = "trajectory.npz"
_METADATA_KEY = "materialized_trajectory_cache"


def mujoco_model_sha256(model: mujoco.MjModel) -> str:
    """Return a stable digest of the complete compiled MuJoCo model."""
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer).hexdigest()


class MaterializedTrajectoryCache:
    """One immutable trajectory cache variant for a cohort and MuJoCo model."""

    def __init__(
        self,
        root: str | Path,
        cohort_key: str,
        *,
        model: mujoco.MjModel,
        control_dt: float,
        cache_type: TrajectoryCacheType | str,
        site_names: list[str] | None,
        clip_joint_ranges: bool,
    ) -> None:
        cohort_key = str(cohort_key).strip().lower()
        if _CACHE_KEY.fullmatch(cohort_key) is None:
            raise ValueError("trajectory cache cohort key must be a lowercase SHA-256 digest")

        cache_type = TrajectoryCacheType(cache_type)
        self.root = Path(root).expanduser().resolve()
        self.cohort_key = cohort_key
        self.descriptor = {
            "schema_version": MATERIALIZED_TRAJECTORY_CACHE_VERSION,
            "cohort_key": cohort_key,
            "model_sha256": mujoco_model_sha256(model),
            "control_dt": float(control_dt),
            "cache_type": cache_type.value,
            "site_names": None if site_names is None else list(site_names),
            "clip_joint_ranges": bool(clip_joint_ranges),
        }
        encoded = json.dumps(
            self.descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.variant_key = hashlib.sha256(encoded).hexdigest()
        self.cohort_dir = self.root / cohort_key
        self.path = self.cohort_dir / self.variant_key
        self.trajectory_path = self.path / _TRAJECTORY_NAME
        self.manifest_path = self.path / _MANIFEST_NAME
        self.lock_path = self.cohort_dir / f".{self.variant_key}.lock"

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        """Serialize cold materialization and publication for this exact variant."""
        self.cohort_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if not self.path.exists():
                    for stage in self.cohort_dir.glob(f".{self.variant_key}.stage-*"):
                        if stage.is_dir():
                            shutil.rmtree(stage)
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def load(self) -> Trajectory | None:
        """Load a complete cache, or return ``None`` when it has not been published."""
        if not self.path.exists():
            return None
        if not self.path.is_dir() or not self.manifest_path.is_file() or not self.trajectory_path.is_file():
            raise ValueError(f"materialized trajectory cache is incomplete: {self.path}")

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"materialized trajectory cache manifest is invalid: {self.manifest_path}") from error
        if manifest.get("descriptor") != self.descriptor or manifest.get("variant_key") != self.variant_key:
            raise ValueError(f"materialized trajectory cache identity mismatch: {self.path}")

        trajectory = Trajectory.load(self.trajectory_path, backend=np)
        metadata = dict(trajectory.info.metadata or {})
        if metadata.get(_METADATA_KEY) != self.variant_key:
            raise ValueError(f"materialized trajectory cache payload identity mismatch: {self.trajectory_path}")
        if int(np.asarray(trajectory.data.n_samples)) != int(manifest.get("num_samples", -1)):
            raise ValueError(f"materialized trajectory cache sample count mismatch: {self.trajectory_path}")
        if len(np.asarray(trajectory.data.split_points)) - 1 != int(manifest.get("num_trajectories", -1)):
            raise ValueError(f"materialized trajectory cache trajectory count mismatch: {self.trajectory_path}")
        return trajectory

    def publish(self, trajectory: Trajectory) -> None:
        """Atomically publish one fully materialized trajectory."""
        if self.path.exists():
            if self.load() is None:  # pragma: no cover - guarded by exists
                raise RuntimeError(f"materialized trajectory cache disappeared: {self.path}")
            return

        self.cohort_dir.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{self.variant_key}.stage-", dir=self.cohort_dir))
        try:
            metadata = dict(trajectory.info.metadata or {})
            metadata[_METADATA_KEY] = self.variant_key
            cached = replace(trajectory, info=replace(trajectory.info, metadata=metadata))
            cached.save(stage / _TRAJECTORY_NAME)
            manifest = {
                "descriptor": self.descriptor,
                "variant_key": self.variant_key,
                "num_samples": int(np.asarray(cached.data.n_samples)),
                "num_trajectories": len(np.asarray(cached.data.split_points)) - 1,
            }
            (stage / _MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(stage, self.path)
        finally:
            if stage.exists():
                shutil.rmtree(stage)


__all__ = [
    "MATERIALIZED_TRAJECTORY_CACHE_VERSION",
    "MaterializedTrajectoryCache",
    "mujoco_model_sha256",
]
