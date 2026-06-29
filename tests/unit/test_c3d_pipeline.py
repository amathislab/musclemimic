from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

import loco_mujoco.task_factories.imitation_factory as imitation_factory_module
import loco_mujoco.trajectory as trajectory_module
from loco_mujoco.task_factories import C3DDatasetConf, ImitationFactory
from musclemimic.web_viewer import c3d_pipeline, c3d_to_smpl


class _FakeTrajectory:
    def __init__(self, name: str, calls: dict):
        self.name = name
        self.calls = calls

    def save(self, path: str) -> None:
        self.calls.setdefault("saved_trajectory_paths", []).append(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(f"trajectory:{self.name}".encode())


class _FakeEnv:
    model = object()
    dt = 0.01


def _install_fake_retargeting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls: dict) -> None:
    default_model_path = tmp_path / "default_smplh"
    default_model_path.mkdir()
    converted_path = tmp_path / "converted"

    fake = types.ModuleType("loco_mujoco.smpl.retargeting")
    fake.OPTIMIZED_SHAPE_FILE_NAME = "shape.npz"
    fake.get_smpl_model_path = lambda: str(default_model_path)
    fake.get_converted_amass_dataset_path = lambda: str(converted_path)
    fake.load_robot_conf_file = lambda model_name: {"model_name": model_name}

    def fit_smpl_shape(model_name, robot_conf, model_path, shape_path, log):
        calls["shape_model_path"] = model_path
        Path(shape_path).parent.mkdir(parents=True, exist_ok=True)
        Path(shape_path).write_bytes(b"shape")

    def fit_smpl_motion(model_name, robot_conf, model_path, motion_data, shape_path, log):
        calls["motion_model_path"] = model_path
        calls["motion_data"] = motion_data
        return _FakeTrajectory("raw-smpl", calls), {"retarget": "smpl"}

    def fit_gmr_motion(model_name, robot_conf, motion_path, log, gmr_config):
        calls["gmr_motion_path"] = motion_path
        calls["gmr_config"] = gmr_config
        return _FakeTrajectory("raw-gmr", calls), {"retarget": "gmr"}

    def extend_motion(model_name, env_params, trajectory, log):
        calls["extend_model_name"] = model_name
        calls["extend_env_params"] = env_params
        calls["extend_input_trajectory"] = trajectory
        return _FakeTrajectory(f"extended-{trajectory.name}", calls)

    fake.fit_smpl_shape = fit_smpl_shape
    fake.fit_smpl_motion = fit_smpl_motion
    fake.fit_gmr_motion = fit_gmr_motion
    fake.extend_motion = extend_motion
    monkeypatch.setitem(sys.modules, "loco_mujoco.smpl.retargeting", fake)


def test_retarget_c3d_splits_fit_and_smpl_retarget_model_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))

    c3d_file = tmp_path / "WalkTrial01.c3d"
    c3d_file.write_bytes(b"c3d")
    c3d_model_path = tmp_path / "soma_smplx"
    retarget_model_path = tmp_path / "retarget_smplh"
    c3d_model_path.mkdir()
    retarget_model_path.mkdir()

    def fit_smpl_to_c3d_cached(c3d_path, model_path, **kwargs):
        calls["fit_c3d_path"] = c3d_path
        calls["fit_model_path"] = model_path
        calls["fit_kwargs"] = kwargs
        return {
            "pose_aa": np.zeros((1, 72), dtype=np.float32),
            "trans": np.zeros((1, 3), dtype=np.float32),
            "betas": np.zeros(16, dtype=np.float32),
            "fps": 30.0,
            "gender": "male",
        }

    monkeypatch.setattr(c3d_to_smpl, "fit_smpl_to_c3d_cached", fit_smpl_to_c3d_cached)

    trajectory, analysis = c3d_pipeline.retarget_c3d_to_trajectory(
        str(c3d_file),
        "MyoFullBody",
        retargeting_method="smpl",
        surface_model_type="smplx",
        c3d_fit_model_path=str(c3d_model_path),
        retarget_smpl_model_path=str(retarget_model_path),
        converted_c3d_name="ExampleStudy/Subject01/Walking/Trial01/WalkTrial01",
    )

    assert trajectory.name == "extended-raw-smpl"
    assert analysis["retarget"] == "smpl"
    assert analysis["source"] == "c3d"
    assert analysis["converted_c3d_dataset"] == "ExampleStudy/Subject01/Walking/Trial01/WalkTrial01"
    assert calls["fit_model_path"] == str(c3d_model_path)
    assert calls["shape_model_path"] == str(retarget_model_path)
    assert calls["motion_model_path"] == str(retarget_model_path)
    assert calls["fit_kwargs"]["surface_model_type"] == "smplx"
    assert calls["extend_model_name"] == "MyoFullBody"
    assert calls["extend_env_params"] == {}

    expected_motion_path = (
        converted_c3d_path / "MyoFullBody" / "smpl" / "ExampleStudy" / "Subject01" / "Walking" / "Trial01" / "WalkTrial01.npz"
    )
    expected_analysis_path = expected_motion_path.with_name("WalkTrial01_analysis.npz")
    assert calls["saved_trajectory_paths"] == [expected_motion_path]
    assert expected_motion_path.read_bytes() == b"trajectory:extended-raw-smpl"
    saved_analysis = np.load(expected_analysis_path)
    assert str(saved_analysis["source"]) == "c3d"
    assert str(saved_analysis["converted_c3d_path"]) == str(expected_motion_path)


def test_smpl_retargeting_rejects_reused_smplx_fit_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)

    c3d_file = tmp_path / "WalkTrial01.c3d"
    c3d_file.write_bytes(b"c3d")
    c3d_model_path = tmp_path / "soma_smplx"
    c3d_model_path.mkdir()

    with pytest.raises(ValueError, match="cannot reuse the SMPL-X C3D fitting model path"):
        c3d_pipeline.retarget_c3d_to_trajectory(
            str(c3d_file),
            "MyoFullBody",
            retargeting_method="smpl",
            surface_model_type="smplx",
            c3d_fit_model_path=str(c3d_model_path),
            retarget_smpl_model_path=str(c3d_model_path),
        )


def test_retarget_c3d_saves_gmr_trajectory_under_converted_c3d_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))

    c3d_file = tmp_path / "Project Data" / "All Steps.c3d"
    c3d_file.parent.mkdir()
    c3d_file.write_bytes(b"c3d")
    c3d_model_path = tmp_path / "soma_smplx"
    c3d_model_path.mkdir()

    monkeypatch.setattr(
        c3d_to_smpl,
        "fit_smpl_to_c3d_cached",
        lambda *args, **kwargs: {
            "pose_aa": np.zeros((1, 72), dtype=np.float32),
            "trans": np.zeros((1, 3), dtype=np.float32),
            "betas": np.zeros(16, dtype=np.float32),
            "fps": 30.0,
            "gender": "male",
        },
    )

    trajectory, analysis = c3d_pipeline.retarget_c3d_to_trajectory(
        str(c3d_file),
        "MjxMyoFullBody",
        c3d_fit_model_path=str(c3d_model_path),
        converted_c3d_name="Study A/Subject 1/All Steps.c3d",
        gmr_config={"target_fps": 30},
    )

    assert trajectory.name == "extended-raw-gmr"
    assert analysis["retarget"] == "gmr"
    assert analysis["converted_c3d_dataset"] == "Study_A/Subject_1/All_Steps"
    assert Path(calls["gmr_motion_path"]).name == "All Steps_smplh_for_gmr.npz"
    assert calls["gmr_config"] == {"target_fps": 30}

    expected_motion_path = converted_c3d_path / "MyoFullBody" / "gmr" / "Study_A" / "Subject_1" / "All_Steps.npz"
    expected_analysis_path = expected_motion_path.with_name("All_Steps_analysis.npz")
    assert calls["saved_trajectory_paths"] == [expected_motion_path]
    assert expected_motion_path.exists()
    assert expected_analysis_path.exists()


def test_retarget_c3d_loads_existing_converted_cache_without_refitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))

    c3d_file = tmp_path / "WalkTrial01.c3d"
    c3d_file.write_bytes(b"c3d")
    missing_c3d_model_path = tmp_path / "missing_soma"
    converted_name = "ExampleStudy/Subject01/Walking/Trial01/WalkTrial01"
    expected_motion_path = (
        converted_c3d_path / "MyoFullBody" / "gmr" / "ExampleStudy" / "Subject01" / "Walking" / "Trial01" / "WalkTrial01.npz"
    )
    expected_analysis_path = expected_motion_path.with_name("WalkTrial01_analysis.npz")
    expected_motion_path.parent.mkdir(parents=True)
    expected_motion_path.write_bytes(b"cached trajectory")
    np.savez(expected_analysis_path, retarget=np.array("gmr"), source=np.array("stale"), frames=np.array(12))

    loaded_trajectory = object()

    def fake_load(path, backend):
        calls["load_path"] = Path(path)
        calls["load_backend"] = backend
        return loaded_trajectory

    def fail_fit(*_args, **_kwargs):
        raise AssertionError("Final converted C3D cache should load before SMPL fitting")

    monkeypatch.setattr(trajectory_module.Trajectory, "load", staticmethod(fake_load))
    monkeypatch.setattr(c3d_to_smpl, "fit_smpl_to_c3d_cached", fail_fit)

    trajectory, analysis = c3d_pipeline.retarget_c3d_to_trajectory(
        str(c3d_file),
        "MyoFullBody",
        c3d_fit_model_path=str(missing_c3d_model_path),
        converted_c3d_name=converted_name,
    )

    assert trajectory is loaded_trajectory
    assert calls == {"load_path": expected_motion_path, "load_backend": np}
    assert analysis["retarget"] == "gmr"
    assert analysis["frames"] == 12
    assert analysis["source"] == "c3d"
    assert analysis["source_c3d_path"] == str(c3d_file.resolve())
    assert analysis["converted_c3d_path"] == str(expected_motion_path)
    assert analysis["converted_c3d_analysis_path"] == str(expected_analysis_path)
    assert analysis["converted_c3d_dataset"] == converted_name
    assert analysis["retargeting_method"] == "gmr"


def test_retarget_c3d_reports_invalid_converted_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))

    c3d_file = tmp_path / "WalkTrial01.c3d"
    c3d_file.write_bytes(b"c3d")
    motion_path = converted_c3d_path / "MyoFullBody" / "gmr" / "Study" / "Trial.npz"
    motion_path.parent.mkdir(parents=True)
    motion_path.write_bytes(b"not a trajectory")

    def fail_load(*_args, **_kwargs):
        raise ValueError("invalid trajectory")

    monkeypatch.setattr(trajectory_module.Trajectory, "load", staticmethod(fail_load))

    with pytest.raises(RuntimeError, match="--clear-c3d-cache"):
        c3d_pipeline.retarget_c3d_to_trajectory(
            str(c3d_file),
            "MyoFullBody",
            c3d_fit_model_path=str(tmp_path / "missing_soma"),
            converted_c3d_name="Study/Trial",
        )


def test_retarget_c3d_clear_cache_forces_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))

    c3d_file = tmp_path / "WalkTrial01.c3d"
    c3d_file.write_bytes(b"c3d")
    c3d_model_path = tmp_path / "soma_smplx"
    c3d_model_path.mkdir()
    expected_motion_path = converted_c3d_path / "MyoFullBody" / "gmr" / "Study" / "Subject" / "Trial.npz"
    expected_motion_path.parent.mkdir(parents=True)
    expected_motion_path.write_bytes(b"old converted cache")

    def fail_load(*_args, **_kwargs):
        raise AssertionError("clear_cache=True should not load the final converted C3D cache")

    def fit_smpl_to_c3d_cached(c3d_path, model_path, **kwargs):
        calls["fit_c3d_path"] = c3d_path
        calls["fit_model_path"] = model_path
        calls["fit_kwargs"] = kwargs
        return {
            "pose_aa": np.zeros((1, 72), dtype=np.float32),
            "trans": np.zeros((1, 3), dtype=np.float32),
            "betas": np.zeros(16, dtype=np.float32),
            "fps": 30.0,
            "gender": "male",
        }

    monkeypatch.setattr(trajectory_module.Trajectory, "load", staticmethod(fail_load))
    monkeypatch.setattr(c3d_to_smpl, "fit_smpl_to_c3d_cached", fit_smpl_to_c3d_cached)

    trajectory, analysis = c3d_pipeline.retarget_c3d_to_trajectory(
        str(c3d_file),
        "MyoFullBody",
        c3d_fit_model_path=str(c3d_model_path),
        converted_c3d_name="Study/Subject/Trial",
        clear_cache=True,
    )

    assert trajectory.name == "extended-raw-gmr"
    assert analysis["source"] == "c3d"
    assert calls["fit_c3d_path"] == str(c3d_file)
    assert calls["fit_model_path"] == str(c3d_model_path)
    assert calls["fit_kwargs"]["clear_cache"] is True
    assert calls["saved_trajectory_paths"] == [expected_motion_path]
    assert expected_motion_path.read_bytes() == b"trajectory:extended-raw-gmr"


def test_converted_c3d_name_rejects_unsafe_paths() -> None:
    for unsafe in ("../trial", "/tmp/trial", "subject/../trial", "", "."):
        with pytest.raises(ValueError, match="C3D dataset name"):
            c3d_pipeline._normalize_c3d_dataset_name(unsafe)


def test_converted_c3d_path_uses_dedicated_env_before_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUSCLEMIMIC_CONFIG_PATH", str(tmp_path / "missing_config.yaml"))
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(tmp_path / "explicit_c3d"))
    monkeypatch.setenv("MUSCLEMIMIC_HOME", str(tmp_path / "home"))

    assert c3d_pipeline.get_converted_c3d_dataset_path() == tmp_path / "explicit_c3d"

    monkeypatch.delenv("MUSCLEMIMIC_CONVERTED_C3D_PATH")
    assert c3d_pipeline.get_converted_c3d_dataset_path() == tmp_path / "home" / "caches" / "C3D"


def test_c3d_dataset_conf_loads_from_converted_c3d_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))
    trajectory_path = converted_c3d_path / "_FakeEnv" / "gmr" / "Study_A" / "Subject_1" / "Trial_05.npz"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_bytes(b"trajectory")

    calls: dict = {}
    loaded_trajectory = object()

    def fake_load(path, backend):
        calls["load_path"] = Path(path)
        calls["load_backend"] = backend
        return loaded_trajectory

    monkeypatch.setattr(imitation_factory_module.Trajectory, "load", staticmethod(fake_load))

    traj = ImitationFactory.get_c3d_traj(_FakeEnv(), C3DDatasetConf(rel_dataset_path="Study A/Subject 1/Trial 05"))

    assert traj is loaded_trajectory
    assert calls["load_path"] == trajectory_path
    assert calls["load_backend"] is np


def test_c3d_dataset_conf_reports_missing_converted_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(tmp_path / "converted_c3d"))

    with pytest.raises(FileNotFoundError, match="Converted C3D trajectory not found"):
        ImitationFactory.get_c3d_traj(_FakeEnv(), C3DDatasetConf(rel_dataset_path="Missing/Trial"))
