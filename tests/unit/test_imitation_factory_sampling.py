import random
from copy import copy

import mujoco
import numpy as np

from loco_mujoco.task_factories import imitation_factory
from loco_mujoco.task_factories.dataset_confs import AMASSDatasetConf
from loco_mujoco.task_factories.imitation_factory import ImitationFactory
from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryData,
    TrajectoryHandler,
    TrajectoryInfo,
    TrajectoryModel,
)


class MyoBimanualArm:
    """Minimal fake whose class name selects the bimanual loading path."""


def test_amass_max_motions_subset_is_reproducible(monkeypatch):
    motion_paths = [f"motion_{index}" for index in range(20)]
    loaded_subsets = []

    monkeypatch.setattr(
        imitation_factory,
        "get_amass_dataset_groups",
        lambda: {"TEST_MOTIONS": motion_paths},
    )
    monkeypatch.setattr(
        imitation_factory,
        "expand_amass_dataset_group_spec",
        lambda group: [group],
    )
    monkeypatch.setattr(
        imitation_factory,
        "retarget_smpl_to_bimanual_via_intermediate",
        lambda paths, **_kwargs: loaded_subsets.append(list(paths)),
    )

    config = AMASSDatasetConf(
        dataset_group="TEST_MOTIONS",
        max_motions=5,
        max_motions_seed=7,
    )
    random.seed(1)
    ImitationFactory.get_amass_traj(MyoBimanualArm(), config)
    random.seed(999)
    ImitationFactory.get_amass_traj(MyoBimanualArm(), config)

    expected = random.Random(7).sample(motion_paths, 5)
    assert loaded_subsets == [expected, expected]


def _cache_test_model():
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root">
              <joint name="hinge" type="hinge"/>
              <geom type="sphere" size="0.05" mass="1"/>
              <site name="mimic"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _cache_test_trajectory(model):
    return Trajectory(
        info=TrajectoryInfo(
            joint_names=["hinge"],
            model=TrajectoryModel(njnt=1, jnt_type=np.asarray(model.jnt_type)),
            frequency=100.0,
        ),
        data=TrajectoryData(
            qpos=np.zeros((3, model.nq), dtype=np.float32),
            qvel=np.zeros((3, model.nv), dtype=np.float32),
            split_points=np.asarray([0, 3]),
        ),
    )


class _CacheTestEnv:
    def __init__(self, th_params=None, **_kwargs):
        self._model = _cache_test_model()
        self._th_params = th_params
        self.dt = 0.01
        self.sites_for_mimic = ["mimic"]
        self.th = None

    def load_trajectory(self, **kwargs):
        self.th = TrajectoryHandler(
            self._model,
            control_dt=self.dt,
            random_start=True,
            cache_type=kwargs.pop("cache_type"),
            site_names=kwargs.pop("site_names"),
            **kwargs,
        )

    def install_shared_trajectory_handler(self, source):
        self.th = copy(source)


def test_imitation_factory_reuses_persistent_materialized_cache(monkeypatch, tmp_path):
    from musclemimic.environments.base import LocoEnv

    monkeypatch.setitem(LocoEnv.registered_envs, "CacheTestEnv", _CacheTestEnv)
    loads = []

    def load_source(_cls, env, _config, **_kwargs):
        loads.append(True)
        return _cache_test_trajectory(env._model)

    monkeypatch.setattr(ImitationFactory, "get_amass_traj", classmethod(load_source))
    config = AMASSDatasetConf(rel_dataset_path=["Study/Motion"], trajectory_cache_type="sparse")
    kwargs = {
        "env_name": "CacheTestEnv",
        "amass_dataset_conf": config,
        "trajectory_cache_root": str(tmp_path),
        "trajectory_cache_key": "c" * 64,
    }

    cold = ImitationFactory.make(**kwargs)
    warm = ImitationFactory.make(**kwargs)

    assert loads == [True]
    assert cold.th.traj is not warm.th.traj
    np.testing.assert_array_equal(cold.th.traj.data.site_xpos, warm.th.traj.data.site_xpos)


def test_imitation_factory_shared_handler_keeps_one_trajectory_object(monkeypatch):
    from musclemimic.environments.base import LocoEnv

    monkeypatch.setitem(LocoEnv.registered_envs, "CacheTestEnv", _CacheTestEnv)
    model = _cache_test_model()
    source = TrajectoryHandler(
        model,
        traj=_cache_test_trajectory(model),
        control_dt=0.01,
        random_start=True,
        cache_type="sparse",
        site_names=["mimic"],
    )

    validation = ImitationFactory.make(
        env_name="CacheTestEnv",
        amass_dataset_conf=AMASSDatasetConf(rel_dataset_path=["Study/Motion"]),
        trajectory_handler=source,
    )

    assert validation.th is not source
    assert validation.th.traj is source.traj
