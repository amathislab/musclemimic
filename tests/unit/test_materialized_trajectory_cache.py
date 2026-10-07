import sys
import types
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

if "musclemimic_models" not in sys.modules:
    musclemimic_models = types.ModuleType("musclemimic_models")
    musclemimic_models.get_xml_path = lambda *_args, **_kwargs: Path("/tmp/nonexistent.xml")
    sys.modules["musclemimic_models"] = musclemimic_models

from loco_mujoco.task_factories import imitation_factory as imitation_factory_module
from loco_mujoco.task_factories.dataset_confs import AMASSDatasetConf
from loco_mujoco.task_factories.imitation_factory import ImitationFactory
from loco_mujoco.trajectory import (
    LoadedTrajectorySet,
    Trajectory,
    TrajectoryCacheType,
    TrajectoryData,
    TrajectoryHandler,
    TrajectoryInfo,
    TrajectoryModel,
    materialize_trajectory,
)
from loco_mujoco.trajectory import handler as handler_module
from loco_mujoco.trajectory.materialized_cache import (
    load_or_build_materialized_trajectory,
    materialized_trajectory_fingerprint,
)


def _make_model():
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root" pos="0 0 1">
              <freejoint name="root_free"/>
              <geom type="sphere" size="0.05" mass="1"/>
              <site name="root_site"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _make_source_trajectory(model, frequency=50.0):
    qpos = np.zeros((3, model.nq), dtype=np.float32)
    qpos[:, 0] = np.arange(3, dtype=np.float32) * 0.01
    qpos[:, 2] = 1.0
    qpos[:, 3] = 1.0
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    return Trajectory(
        info=TrajectoryInfo(
            joint_names=joint_names,
            model=TrajectoryModel(njnt=model.njnt, jnt_type=np.array(model.jnt_type)),
            frequency=frequency,
        ),
        data=TrajectoryData(
            qpos=qpos,
            qvel=np.zeros((3, model.nv), dtype=np.float32),
            split_points=np.array([0, 3]),
        ),
    )


def _make_materialized_trajectory(model, control_dt=0.02):
    return materialize_trajectory(
        _make_source_trajectory(model, frequency=1.0 / control_dt),
        model=model,
        control_dt=control_dt,
        cache_type=TrajectoryCacheType.SPARSE,
        site_names=["root_site"],
    )


def test_materialized_cache_fingerprint_uses_motion_set_and_control_dt():
    model = _make_model()
    kwargs = {
        "model": model,
        "cache_type": TrajectoryCacheType.SPARSE,
        "site_names": ["root_site"],
        "retargeting_method": "gmr",
    }

    fingerprint = materialized_trajectory_fingerprint(["b", "a", "b"], control_dt=0.02, **kwargs)

    assert fingerprint == materialized_trajectory_fingerprint(["a", "b"], control_dt=0.02, **kwargs)
    assert fingerprint != materialized_trajectory_fingerprint(["a", "b"], control_dt=0.01, **kwargs)
    assert fingerprint == materialized_trajectory_fingerprint(
        ["a", "b"],
        model=_make_model(),
        control_dt=0.02,
        cache_type=TrajectoryCacheType.SPARSE,
        site_names=["root_site"],
        retargeting_method="gmr",
    )


def test_materialized_cache_fingerprint_ignores_visual_asset_paths(tmp_path):
    model = _make_model()
    first_asset = tmp_path / "first" / "texture.png"
    second_asset = tmp_path / "second" / "texture.png"
    first_asset.parent.mkdir()
    second_asset.parent.mkdir()
    first_asset.write_bytes(b"same asset contents")
    second_asset.write_bytes(b"same asset contents")

    def model_spec(asset_path):
        return SimpleNamespace(
            modelfiledir=str(asset_path.parent),
            to_xml=lambda: f'<mujoco><asset><texture name="texture" type="2d" file="{asset_path}"/></asset></mujoco>',
        )

    kwargs = {
        "motion_names": ["motion"],
        "model": model,
        "control_dt": 0.02,
        "cache_type": TrajectoryCacheType.SPARSE,
        "site_names": ["root_site"],
        "retargeting_method": "gmr",
    }
    first_fingerprint = materialized_trajectory_fingerprint(
        **kwargs,
        model_spec=model_spec(first_asset),
    )
    second_fingerprint = materialized_trajectory_fingerprint(
        **kwargs,
        model_spec=model_spec(second_asset),
    )

    assert first_fingerprint == second_fingerprint
    second_asset.write_bytes(b"different asset contents")
    assert first_fingerprint == materialized_trajectory_fingerprint(
        **kwargs,
        model_spec=model_spec(second_asset),
    )


def test_materialized_cache_fingerprint_tracks_compiled_kinematics():
    model = _make_model()
    kwargs = {
        "motion_names": ["motion"],
        "control_dt": 0.02,
        "cache_type": TrajectoryCacheType.SPARSE,
        "site_names": ["root_site"],
        "retargeting_method": "gmr:gmr_v6",
    }
    original = materialized_trajectory_fingerprint(model=model, **kwargs)

    model.body_pos[1, 0] += 0.01

    assert materialized_trajectory_fingerprint(model=model, **kwargs) != original


def test_amass_dataset_conf_validates_gmr_cache_namespace() -> None:
    conf = AMASSDatasetConf(
        rel_dataset_path="motion",
        retargeting_method="gmr",
        gmr_cache_namespace="gmr_v6",
    )
    assert conf.gmr_cache_namespace == "gmr_v6"

    default_conf = AMASSDatasetConf(
        rel_dataset_path="motion",
        retargeting_method="gmr",
    )
    assert default_conf.gmr_cache_namespace == "gmr_v6"

    with pytest.raises(ValueError, match="must use the canonical gmr_v6"):
        AMASSDatasetConf(
            rel_dataset_path="motion",
            retargeting_method="gmr",
            gmr_cache_namespace="experimental_gmr",
        )

    with pytest.raises(ValueError, match="one non-empty path component"):
        AMASSDatasetConf(
            rel_dataset_path="motion",
            retargeting_method="gmr",
            gmr_cache_namespace="../gmr_v6",
        )


def test_materialized_cache_builds_once_then_loads(tmp_path):
    model = _make_model()
    trajectory = _make_materialized_trajectory(model)
    cache_path = tmp_path / "cache.npz"
    build_calls = 0

    def build():
        nonlocal build_calls
        build_calls += 1
        return trajectory

    first = load_or_build_materialized_trajectory(cache_path, "fingerprint", build)
    second = load_or_build_materialized_trajectory(cache_path, "fingerprint", build)

    assert build_calls == 1
    assert cache_path.exists()
    assert not list(tmp_path.glob(".*.tmp.npz"))
    np.testing.assert_array_equal(first.data.qpos, second.data.qpos)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_or_build_materialized_trajectory(cache_path, "different", build)

    load_or_build_materialized_trajectory(cache_path, "fingerprint", build, rebuild=True)
    assert build_calls == 2


def test_materialized_handler_skips_preprocessing(monkeypatch):
    model = _make_model()
    trajectory = _make_materialized_trajectory(model)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("materialize_trajectory should not be called")

    monkeypatch.setattr(handler_module, "materialize_trajectory", fail_if_called)

    handler = TrajectoryHandler(
        model,
        traj=trajectory,
        control_dt=0.02,
        random_start=False,
        fixed_start_conf=(0, 0),
        cache_type=TrajectoryCacheType.SPARSE,
        site_names=["root_site"],
        materialized=True,
        motion_names=("motion-a",),
    )

    assert handler.traj is trajectory
    assert handler.cache_type is TrajectoryCacheType.SPARSE
    assert handler.cache_site_names == ["root_site"]
    assert handler.get_motion_name(0) == "motion-a"

    with pytest.raises(ValueError, match="Expected 1 motion names"):
        TrajectoryHandler(
            model,
            traj=trajectory,
            control_dt=0.02,
            random_start=False,
            fixed_start_conf=(0, 0),
            cache_type=TrajectoryCacheType.SPARSE,
            site_names=["root_site"],
            materialized=True,
            motion_names=("motion-a", "motion-b"),
        )

    with pytest.raises(ValueError, match="frequency"):
        TrajectoryHandler(
            model,
            traj=trajectory,
            control_dt=0.01,
            random_start=False,
            fixed_start_conf=(0, 0),
            cache_type=TrajectoryCacheType.SPARSE,
            site_names=["root_site"],
            materialized=True,
        )


def test_amass_materialized_cache_uses_environment_control_dt(monkeypatch, tmp_path):
    model = _make_model()
    single_source = _make_source_trajectory(model)
    source = Trajectory.concatenate([single_source, single_source], backend=np)
    loaded_source = LoadedTrajectorySet(source, ("motion_a", "motion_b"))
    seen_control_dt = None
    real_fingerprint = materialized_trajectory_fingerprint

    def capture_control_dt(*args, control_dt, **kwargs):
        nonlocal seen_control_dt
        seen_control_dt = control_dt
        return real_fingerprint(*args, control_dt=control_dt, **kwargs)

    monkeypatch.setattr(imitation_factory_module, "get_converted_amass_dataset_path", lambda: str(tmp_path))
    monkeypatch.setattr(imitation_factory_module, "materialized_trajectory_fingerprint", capture_control_dt)
    monkeypatch.setattr(
        ImitationFactory,
        "_load_amass_dataset_paths",
        staticmethod(lambda _env, _conf, _paths, _visualize: loaded_source),
    )

    class FakeEnv:
        dt = 0.02

        def __init__(self):
            self._model = model
            self.sites_for_mimic = ["root_site"]
            self.loaded = None

        def load_trajectory(self, **kwargs):
            self.loaded = kwargs

    env = FakeEnv()
    conf = AMASSDatasetConf(
        rel_dataset_path=["motion_b", "motion_a", "motion_b"],
        retargeting_method="gmr",
    )

    assert ImitationFactory.resolve_amass_dataset_paths(conf) == ["motion_a", "motion_b"]
    ImitationFactory._load_materialized_amass_trajectory(env, conf, visualize_goal=False)

    assert seen_control_dt == env.dt
    assert env.loaded["materialized"] is True
    assert env.loaded["motion_names"] == ("motion_a", "motion_b")
    assert env.loaded["traj"].info.frequency == 1.0 / env.dt
    assert len(list(tmp_path.rglob("*.npz"))) == 1


def test_amass_materialized_cache_rejects_an_incomplete_motion_set(monkeypatch, tmp_path):
    model = _make_model()
    source = _make_source_trajectory(model)
    loaded_source = LoadedTrajectorySet(source, ("motion_a",))

    monkeypatch.setattr(imitation_factory_module, "get_converted_amass_dataset_path", lambda: str(tmp_path))
    monkeypatch.setattr(
        ImitationFactory,
        "_load_amass_dataset_paths",
        staticmethod(lambda _env, _conf, _paths, _visualize: loaded_source),
    )

    class FakeEnv:
        dt = 0.02
        sites_for_mimic = ("root_site",)

        def __init__(self):
            self._model = model

        def load_trajectory(self, **_kwargs):
            raise AssertionError("Incomplete materialized cache must not be loaded")

    conf = AMASSDatasetConf(
        rel_dataset_path=["motion_a", "motion_b"],
        retargeting_method="gmr",
    )

    with pytest.raises(RuntimeError, match=r"Failed motions: \['motion_b'\]"):
        ImitationFactory._load_materialized_amass_trajectory(FakeEnv(), conf, visualize_goal=False)

    assert not list(tmp_path.rglob("*.npz"))


def test_amass_materialized_cache_excludes_persistent_quality_rejection(monkeypatch, tmp_path):
    model = _make_model()
    source = _make_source_trajectory(model)
    loaded_source = LoadedTrajectorySet(source, ("motion_a",))

    monkeypatch.setattr(imitation_factory_module, "get_converted_amass_dataset_path", lambda: str(tmp_path))
    monkeypatch.setattr(
        ImitationFactory,
        "_load_amass_dataset_paths",
        staticmethod(lambda _env, _conf, _paths, _visualize: loaded_source),
    )
    rejection_path = tmp_path / "FakeEnv/gmr_v6/motion_b_quality_rejected.json"
    rejection_path.parent.mkdir(parents=True)
    rejection_path.write_text("{}\n")

    class FakeEnv:
        dt = 0.02
        sites_for_mimic = ("root_site",)

        def __init__(self):
            self._model = model
            self.loaded = None

        def load_trajectory(self, **kwargs):
            self.loaded = kwargs

    env = FakeEnv()
    conf = AMASSDatasetConf(
        rel_dataset_path=["motion_a", "motion_b"],
        retargeting_method="gmr",
    )

    ImitationFactory._load_materialized_amass_trajectory(env, conf, visualize_goal=False)

    assert env.loaded["motion_names"] == ("motion_a",)
    assert env.loaded["traj"].data.n_trajectories == 1
    assert len(list(tmp_path.rglob("*.npz"))) == 1


def test_amass_materialized_cache_uses_custom_gmr_namespace(monkeypatch, tmp_path):
    model = _make_model()
    source = _make_source_trajectory(model)
    loaded_source = LoadedTrajectorySet(source, ("motion_a",))
    seen_retargeting_identities = []
    real_fingerprint = materialized_trajectory_fingerprint

    def capture_identity(*args, retargeting_method, **kwargs):
        seen_retargeting_identities.append(retargeting_method)
        return real_fingerprint(
            *args,
            retargeting_method=retargeting_method,
            **kwargs,
        )

    monkeypatch.setattr(imitation_factory_module, "get_converted_amass_dataset_path", lambda: str(tmp_path))
    monkeypatch.setattr(
        imitation_factory_module,
        "materialized_trajectory_fingerprint",
        capture_identity,
    )
    monkeypatch.setattr(
        ImitationFactory,
        "_load_amass_dataset_paths",
        staticmethod(lambda _env, _conf, _paths, _visualize: loaded_source),
    )

    class FakeEnv:
        dt = 0.02
        sites_for_mimic = ("root_site",)

        def __init__(self):
            self._model = model
            self.loaded = None

        def load_trajectory(self, **kwargs):
            self.loaded = kwargs

    conf = AMASSDatasetConf(
        rel_dataset_path="motion_a",
        retargeting_method="gmr",
        gmr_cache_namespace="gmr_v6",
    )
    env = FakeEnv()

    ImitationFactory._load_materialized_amass_trajectory(env, conf, visualize_goal=False)

    cache_identity, _cache_variant = imitation_factory_module.get_gmr_cache_identity(None)
    assert seen_retargeting_identities == [f"gmr:gmr_v6:{cache_identity}"]
    assert env.loaded["materialized"] is True


def test_amass_factory_uses_materialized_cache_when_enabled(monkeypatch):
    from musclemimic.environments.base import LocoEnv

    class FakeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loaded = None

        def load_trajectory(self, **kwargs):
            self.loaded = kwargs

    monkeypatch.setitem(LocoEnv.registered_envs, "FakeEnv", FakeEnv)

    calls = []

    def fake_load_materialized(cls, env, conf, visualize_goal):
        calls.append((env, conf, visualize_goal))
        env.load_trajectory(traj="cached", materialized=True)

    monkeypatch.setattr(
        ImitationFactory,
        "_load_materialized_amass_trajectory",
        classmethod(fake_load_materialized),
    )

    env = ImitationFactory.make(
        env_name="FakeEnv",
        amass_dataset_conf={"rel_dataset_path": ["motion_a"], "write_materialized_cache": True},
    )

    assert calls
    assert calls[0][1].rel_dataset_path == ["motion_a"]
    assert env.loaded == {"traj": "cached", "materialized": True}


def test_amass_factory_uses_runtime_loader_by_default(monkeypatch):
    from musclemimic.environments.base import LocoEnv

    model = _make_model()
    source = _make_source_trajectory(model)
    loaded_source = LoadedTrajectorySet(source, ("motion_a",))

    class FakeEnv:
        dt = 0.02
        sites_for_mimic = ("root_site",)

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._model = model
            self.loaded = None

        def load_trajectory(self, **kwargs):
            self.loaded = kwargs

    monkeypatch.setitem(LocoEnv.registered_envs, "FakeEnv", FakeEnv)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("persistent materialized AMASS cache should not be used")

    monkeypatch.setattr(ImitationFactory, "_load_materialized_amass_trajectory", fail_if_called)
    monkeypatch.setattr(
        ImitationFactory,
        "_load_amass_dataset_paths",
        staticmethod(lambda _env, _conf, _paths, _visualize: loaded_source),
    )

    env = ImitationFactory.make(
        env_name="FakeEnv",
        amass_dataset_conf={"rel_dataset_path": ["motion_a"]},
    )

    assert env.loaded["traj"] is source
    assert env.loaded["motion_names"] == ("motion_a",)
    assert env.loaded["cache_type"] == TrajectoryCacheType.SPARSE
    assert env.loaded["site_names"] == ["root_site"]
    assert "materialized" not in env.loaded


def test_runtime_amass_can_be_combined_with_other_datasets(monkeypatch):
    from musclemimic.environments.base import LocoEnv

    model = _make_model()
    amass = LoadedTrajectorySet(_make_source_trajectory(model), ("motion_a",))
    c3d = LoadedTrajectorySet(_make_source_trajectory(model), ("recording_a",))

    class FakeEnv:
        sites_for_mimic = ("root_site",)

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loaded = None

        def load_trajectory(self, **kwargs):
            self.loaded = kwargs

    monkeypatch.setitem(LocoEnv.registered_envs, "FakeEnv", FakeEnv)
    monkeypatch.setattr(
        ImitationFactory,
        "get_amass_trajectory_set",
        classmethod(lambda _cls, _env, _conf, _visualize=False: amass),
    )
    monkeypatch.setattr(
        ImitationFactory,
        "get_c3d_trajectory_set",
        classmethod(lambda _cls, _env, _conf: c3d),
    )

    env = ImitationFactory.make(
        env_name="FakeEnv",
        amass_dataset_conf={"rel_dataset_path": ["motion_a"]},
        c3d_dataset_conf={"rel_dataset_path": ["recording_a"]},
    )

    assert env.loaded["motion_names"] == ("motion_a", "recording_a")
    assert env.loaded["traj"].data.n_trajectories == 2


def test_materialized_amass_rejects_mixed_datasets_when_explicitly_enabled(monkeypatch):
    from musclemimic.environments.base import LocoEnv

    class FakeEnv:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setitem(LocoEnv.registered_envs, "FakeEnv", FakeEnv)

    with pytest.raises(ValueError, match="cannot be combined"):
        ImitationFactory.make(
            env_name="FakeEnv",
            amass_dataset_conf={
                "rel_dataset_path": ["motion_a"],
                "write_materialized_cache": True,
            },
            c3d_dataset_conf={"rel_dataset_path": ["recording_a"]},
        )


@pytest.mark.parametrize(
    "config_name",
    [
        "conf_fullbody_gmr_resnet_kit_kinesis_4gpu",
        "conf_fullbody_flashsac_4gpu",
    ],
)
def test_multi_gpu_configs_explicitly_enable_materialized_cache(config_name):
    repository_root = Path(__file__).parents[2]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(repository_root / "fullbody"),
    ):
        config = compose(config_name=config_name)

    assert config.experiment.task_factory.params.amass_dataset_conf.write_materialized_cache is True
