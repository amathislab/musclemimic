from __future__ import annotations

from typing import TYPE_CHECKING

from .dataclasses import (
    LoadedTrajectorySet,
    Trajectory,
    TrajectoryCacheType,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryTransitions,
    compute_trajectory_kinematic_caches,
    interpolate_trajectories,
    recompute_trajectory_velocities,
)

__all__ = [
    "LoadedTrajectorySet",
    "TrajState",
    "Trajectory",
    "TrajectoryCacheType",
    "TrajectoryData",
    "TrajectoryHandler",
    "TrajectoryInfo",
    "TrajectoryModel",
    "TrajectoryTransitions",
    "compute_trajectory_kinematic_caches",
    "interpolate_trajectories",
    "materialize_trajectory",
    "recompute_trajectory_velocities",
]

_LAZY_ATTRS: dict[str, str] = {
    "TrajectoryHandler": "TrajectoryHandler",
    "TrajState": "TrajState",
    "materialize_trajectory": "materialize_trajectory",
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        from . import handler as _handler

        value = getattr(_handler, _LAZY_ATTRS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:
    from .handler import TrajectoryHandler, TrajState, materialize_trajectory
