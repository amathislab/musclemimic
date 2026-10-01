from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from .dataclasses import (
    Trajectory,
    TrajectoryCacheType,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    TrajectoryTransitions,
    interpolate_trajectories,
    recompute_trajectory_velocities,
    compute_trajectory_kinematic_caches,
)

__all__ = [
    "Trajectory",
    "TrajectoryCacheType",
    "TrajectoryInfo",
    "TrajectoryModel",
    "TrajectoryData",
    "TrajectoryTransitions",
    "interpolate_trajectories",
    "recompute_trajectory_velocities",
    "compute_trajectory_kinematic_caches",
    "materialize_trajectory",
    "TrajectoryHandler",
    "TrajState",
]

_LAZY_ATTRS: Dict[str, str] = {
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
