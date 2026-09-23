from types import SimpleNamespace

import numpy as np
import pytest

from loco_mujoco.trajectory import LoadedTrajectorySet, Trajectory


def _fake_trajectory(n_trajectories: int):
    return SimpleNamespace(data=SimpleNamespace(n_trajectories=n_trajectories))


def test_loaded_trajectory_set_requires_one_name_per_trajectory():
    with pytest.raises(ValueError, match="Expected 2 motion names"):
        LoadedTrajectorySet(_fake_trajectory(2), ("motion-a",))


def test_loaded_trajectory_set_normalizes_names():
    loaded = LoadedTrajectorySet(_fake_trajectory(2), (123, None))

    assert loaded.motion_names == ("123", None)


def test_loaded_trajectory_sets_concatenate_names_in_trajectory_order(monkeypatch):
    first = LoadedTrajectorySet(_fake_trajectory(1), ("motion-a",))
    second = LoadedTrajectorySet(_fake_trajectory(1), ("motion-b",))
    combined_trajectory = _fake_trajectory(2)
    monkeypatch.setattr(
        Trajectory,
        "concatenate",
        staticmethod(lambda trajectories, backend: combined_trajectory),
    )

    combined = LoadedTrajectorySet.concatenate([first, second], backend=np)

    assert combined.trajectory is combined_trajectory
    assert combined.motion_names == ("motion-a", "motion-b")


def test_loaded_trajectory_set_concatenation_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one loaded trajectory set"):
        LoadedTrajectorySet.concatenate([], backend=np)


def test_loaded_trajectory_set_concatenation_honors_backend_for_singleton(monkeypatch):
    loaded = LoadedTrajectorySet(_fake_trajectory(1), ("motion-a",))
    combined_trajectory = _fake_trajectory(1)
    call = {}

    def concatenate(trajectories, backend):
        call["trajectories"] = trajectories
        call["backend"] = backend
        return combined_trajectory

    monkeypatch.setattr(Trajectory, "concatenate", staticmethod(concatenate))

    combined = LoadedTrajectorySet.concatenate([loaded], backend=np)

    assert call == {"trajectories": [loaded.trajectory], "backend": np}
    assert combined.trajectory is combined_trajectory
    assert combined.motion_names == ("motion-a",)
