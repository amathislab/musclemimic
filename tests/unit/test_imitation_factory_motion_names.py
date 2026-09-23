from types import SimpleNamespace

import numpy as np

from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.trajectory import LoadedTrajectorySet
from musclemimic.environments.base import LocoEnv


def test_imitation_factory_passes_motion_names_to_environment(monkeypatch):
    trajectory = SimpleNamespace(data=SimpleNamespace(n_trajectories=2))
    loaded = LoadedTrajectorySet(trajectory, ("motion-a", "motion-b"))

    class FakeMotionEnv:
        def __init__(self, **_kwargs):
            self.loaded = None

        def load_trajectory(self, **kwargs):
            self.loaded = kwargs

    monkeypatch.setitem(LocoEnv.registered_envs, "FakeMotionEnv", FakeMotionEnv)
    monkeypatch.setattr(
        ImitationFactory,
        "get_amass_trajectory_set",
        classmethod(lambda _cls, _env, _conf, visualize_goal=False: loaded),
    )
    monkeypatch.setattr(
        LoadedTrajectorySet,
        "concatenate",
        classmethod(lambda _cls, loaded_sets, backend=np: loaded_sets[0]),
    )

    env = ImitationFactory.make(
        env_name="FakeMotionEnv",
        amass_dataset_conf={"rel_dataset_path": ["motion-a", "motion-b"]},
    )

    assert env.loaded["traj"] is trajectory
    assert env.loaded["motion_names"] == ("motion-a", "motion-b")
