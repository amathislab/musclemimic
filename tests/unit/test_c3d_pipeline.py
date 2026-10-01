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


def test_retarget_c3d_validates_environment_before_processing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown MuscleMimic environment"):
        c3d_pipeline.retarget_c3d_to_trajectory(
            str(tmp_path / "missing.c3d"),
            "NotARegisteredEnvironment",
        )


def test_c3d_analysis_round_trips_structured_metadata_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "analysis.npz"
    analysis = {
        "terrain": {"boxes": [{"name": "ramp", "yaw": 0.25}]},
        "active_terms": ["interaction_mesh", "direct_environment_nonpenetration"],
    }

    c3d_pipeline._save_analysis_npz(path, analysis)

    assert c3d_pipeline._load_analysis_npz(path) == analysis


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

    c3d_file = tmp_path / "AllSteps.c3d"
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
        converted_c3d_name="EMG_Boo/S002/LevelWalking/Trial05/AllSteps",
    )

    assert trajectory.name == "extended-raw-smpl"
    assert analysis["retarget"] == "smpl"
    assert analysis["source"] == "c3d"
    assert analysis["converted_c3d_dataset"] == "EMG_Boo/S002/LevelWalking/Trial05/AllSteps"
    assert analysis["c3d_fit_model_path"] == str(c3d_model_path.resolve())
    assert analysis["c3d_surface_model_type"] == "smplx"
    assert analysis["c3d_gender"] == "male"
    assert analysis["smpl_model_path"] == str(retarget_model_path.resolve())
    assert len(analysis["fitted_shape_sha256"]) == 64
    assert calls["fit_model_path"] == str(c3d_model_path)
    assert calls["shape_model_path"] == str(retarget_model_path)
    assert calls["motion_model_path"] == str(retarget_model_path)
    assert calls["fit_kwargs"]["surface_model_type"] == "smplx"
    assert calls["extend_model_name"] == "MyoFullBody"
    assert calls["extend_env_params"] == {}

    expected_motion_path = (
        converted_c3d_path / "MyoFullBody" / "smpl" / "EMG_Boo" / "S002" / "LevelWalking" / "Trial05" / "AllSteps.npz"
    )
    expected_analysis_path = expected_motion_path.with_name("AllSteps_analysis.npz")
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

    c3d_file = tmp_path / "AllSteps.c3d"
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
    gmr_motion_path = Path(calls["gmr_motion_path"])
    assert gmr_motion_path.name == "All Steps_smplh_for_gmr.npz"
    assert converted_c3d_path in gmr_motion_path.parents
    assert not (c3d_file.parent / ".smpl_cache").exists()
    assert calls["gmr_config"] == {"target_fps": 30}

    expected_motion_path = converted_c3d_path / "MyoFullBody" / "gmr" / "Study_A" / "Subject_1" / "All_Steps.npz"
    expected_analysis_path = expected_motion_path.with_name("All_Steps_analysis.npz")
    assert calls["saved_trajectory_paths"] == [expected_motion_path]
    assert expected_motion_path.exists()
    assert expected_analysis_path.exists()


def test_retarget_c3d_keeps_managed_fit_cache_under_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    source_dir = tmp_path / "read_only_captures"
    source_dir.mkdir()
    c3d_file = source_dir / "Trial.c3d"
    c3d_file.write_bytes(b"unique c3d capture")
    c3d_model_path = tmp_path / "soma_smplx"
    c3d_model_path.mkdir()
    c3d_model_file = c3d_model_path / "SMPLX_NEUTRAL.pkl"
    c3d_model_file.write_bytes(b"model parameters")
    cache_root = tmp_path / "cache"
    stage1_state = tmp_path / "subject_stage1.npz"
    stage1_state.write_bytes(b"calibration")

    def fit_smpl_to_c3d_cached(c3d_path, model_path, **kwargs):
        calls["fit_kwargs"] = kwargs
        return {
            "pose_aa": np.zeros((1, 72), dtype=np.float32),
            "trans": np.zeros((1, 3), dtype=np.float32),
            "betas": np.zeros(16, dtype=np.float32),
            "fps": 30.0,
            "gender": "neutral",
            "debug": {
                "surface_model_type": "smplx",
                "canonical_labels": ["LHEE", "RHEE"],
                "fit_pose_dim": 165,
                "stagei": {"reused": True, "state_path": str(stage1_state)},
                "stage1_shape_solver": "joint_dogleg_jax",
                "stage2_solver": "batched_lbfgs",
                "enforce_knee_hinge": True,
                "stage2_marker_weight_overrides": {"LHEE": 2.0},
                "stage2_torso_frame_weight": 3.0,
                "marker_error": {"mean_mm": np.float32(8.5), "max_mm": 20.0},
            },
        }

    monkeypatch.setattr(c3d_to_smpl, "fit_smpl_to_c3d_cached", fit_smpl_to_c3d_cached)

    _, analysis = c3d_pipeline.retarget_c3d_to_trajectory(
        str(c3d_file),
        "MyoFullBody",
        c3d_fit_model_path=str(c3d_model_path),
        cache_root=cache_root,
        converted_c3d_name="Study/Trial",
        device="cpu",
        enforce_knee_hinge=True,
        gender="neutral",
        least_avail_markers=0.8,
        n_ref_frames=16,
        seed=7,
        stage1_state_path=str(stage1_state),
        stage2_marker_weight_overrides={"LHEE": 2.0},
        stage2_solver="batched_lbfgs",
        stage2_torso_frame_weight=3.0,
        strict_frame_picking=False,
        target_fps=50.0,
        wrist_markers_on_stick=True,
    )

    source_hash = c3d_pipeline._file_sha256(c3d_file)
    expected_fit_cache = cache_root.resolve() / ".c3d_smpl_cache" / source_hash
    assert Path(calls["fit_kwargs"]["cache_dir"]) == expected_fit_cache
    assert Path(calls["gmr_motion_path"]).parent == expected_fit_cache
    assert analysis["c3d_fit_cache_dir"] == str(expected_fit_cache)
    assert analysis["c3d_fit_model_file"] == str(c3d_model_file)
    assert analysis["c3d_fit_model_sha256"] == c3d_pipeline._file_sha256(c3d_model_file)
    assert analysis["c3d_fit_options"]["stage2_solver"] == "batched_lbfgs"
    assert analysis["c3d_fit"]["marker_error"] == {"mean_mm": 8.5, "max_mm": 20.0}
    assert analysis["c3d_fit"]["stage1"] == {
        "reused": True,
        "state_path": str(stage1_state),
    }
    assert calls["fit_kwargs"]["target_fps"] == 50.0
    assert calls["fit_kwargs"]["stage1_state_path"] == str(stage1_state.resolve())
    assert calls["fit_kwargs"]["stage2_marker_weight_overrides"] == {"LHEE": 2.0}
    assert calls["fit_kwargs"]["stage2_torso_frame_weight"] == 3.0
    assert not (source_dir / ".smpl_cache").exists()


@pytest.mark.parametrize("method", ["terra", "omniretarget"])
def test_retarget_c3d_dispatches_terra_adapter_methods_truthfully(
    method: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))

    c3d_file = tmp_path / "ramp.c3d"
    c3d_file.write_bytes(b"c3d")
    c3d_model_path = tmp_path / "soma_smplx"
    retarget_model_path = tmp_path / "retarget_smplh"
    c3d_model_path.mkdir()
    retarget_model_path.mkdir()
    motion_data = {
        "pose_aa": np.zeros((1, 72), dtype=np.float32),
        "trans": np.zeros((1, 3), dtype=np.float32),
        "betas": np.zeros(16, dtype=np.float32),
        "fps": 30.0,
        "gender": "male",
    }
    monkeypatch.setattr(c3d_to_smpl, "fit_smpl_to_c3d_cached", lambda *args, **kwargs: motion_data)

    fake_terra = types.ModuleType("terra")

    def resolve_method_profile(profile, config):
        calls["profile"] = profile
        return dict(config or {}) | {"method_profile": profile}

    def fit_terra_motion(
        model_name,
        robot_conf,
        fitted_motion,
        log,
        config,
        *,
        smpl_model_path,
        fitted_shape_path,
    ):
        calls["terra"] = {
            "model_name": model_name,
            "robot_conf": robot_conf,
            "motion_data": fitted_motion,
            "config": config,
            "smpl_model_path": smpl_model_path,
            "fitted_shape_path": fitted_shape_path,
        }
        return _FakeTrajectory(f"raw-{method}", calls), {"retarget": method}

    fake_terra.resolve_method_profile = resolve_method_profile
    fake_terra.fit_terra_motion = fit_terra_motion
    monkeypatch.setitem(sys.modules, "terra", fake_terra)

    trajectory, analysis = c3d_pipeline.retarget_c3d_to_trajectory(
        str(c3d_file),
        "MyoFullBody",
        retargeting_method=method,
        terra_config={"terrain": "auto"},
        surface_model_type="smplx",
        c3d_fit_model_path=str(c3d_model_path),
        retarget_smpl_model_path=str(retarget_model_path),
        converted_c3d_name=f"Study/{method}/ramp",
    )

    assert trajectory.name == f"extended-raw-{method}"
    assert analysis["retarget"] == method
    assert analysis["retargeting_method"] == method
    assert calls["profile"] == method
    assert calls["terra"]["motion_data"] is motion_data
    assert calls["terra"]["config"] == {"terrain": "auto", "method_profile": method}
    assert calls["terra"]["smpl_model_path"] == str(retarget_model_path)
    assert calls["terra"]["fitted_shape_path"].endswith("MyoFullBody/shape.npz")
    assert calls["shape_model_path"] == str(retarget_model_path)
    assert "motion_model_path" not in calls


def test_retarget_c3d_rejects_unknown_method_before_fitting(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported C3D retargeting method"):
        c3d_pipeline.retarget_c3d_to_trajectory(
            str(tmp_path / "missing.c3d"),
            "MyoFullBody",
            retargeting_method="not-a-method",
        )


def test_retarget_c3d_loads_existing_converted_cache_without_refitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))

    c3d_file = tmp_path / "AllSteps.c3d"
    c3d_file.write_bytes(b"c3d")
    missing_c3d_model_path = tmp_path / "missing_soma"
    converted_name = "EMG_Boo/S002/LevelWalking/Trial05/AllSteps"
    expected_motion_path = (
        converted_c3d_path / "MyoFullBody" / "gmr" / "EMG_Boo" / "S002" / "LevelWalking" / "Trial05" / "AllSteps.npz"
    )
    expected_analysis_path = expected_motion_path.with_name("AllSteps_analysis.npz")
    expected_motion_path.parent.mkdir(parents=True)
    expected_motion_path.write_bytes(b"cached trajectory")
    monkeypatch.setattr(c3d_pipeline, "_c3d_cache_key", lambda *_args, **_kwargs: "expected key")
    np.savez(
        expected_analysis_path,
        retarget=np.array("gmr"),
        frames=np.array(12),
        source_c3d_sha256=np.asarray(c3d_pipeline._file_sha256(c3d_file)),
        c3d_cache_key_sha256=np.asarray(c3d_pipeline._cache_key_sha256("expected key")),
    )

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


def test_retarget_c3d_rejects_cache_without_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_root = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_root))
    c3d_file = tmp_path / "AllSteps.c3d"
    c3d_file.write_bytes(b"c3d")
    motion_path = converted_root / "MyoFullBody" / "gmr" / "Study" / "Trial.npz"
    motion_path.parent.mkdir(parents=True)
    motion_path.write_bytes(b"legacy cache")

    with pytest.raises(RuntimeError, match="no source checksum"):
        c3d_pipeline.retarget_c3d_to_trajectory(
            str(c3d_file),
            "MyoFullBody",
            c3d_fit_model_path=str(tmp_path / "missing_soma"),
            converted_c3d_name="Study/Trial",
        )


def test_retarget_c3d_rejects_cache_from_changed_source_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))

    c3d_file = tmp_path / "AllSteps.c3d"
    c3d_file.write_bytes(b"new source contents")
    motion_path = converted_c3d_path / "MyoFullBody" / "gmr" / "Study" / "Trial.npz"
    analysis_path = motion_path.with_name("Trial_analysis.npz")
    motion_path.parent.mkdir(parents=True)
    motion_path.write_bytes(b"cached trajectory")
    np.savez(analysis_path, source_c3d_sha256=np.asarray("0" * 64))
    monkeypatch.setattr(trajectory_module.Trajectory, "load", staticmethod(lambda *_args, **_kwargs: object()))

    with pytest.raises(RuntimeError, match="different source contents"):
        c3d_pipeline.retarget_c3d_to_trajectory(
            str(c3d_file),
            "MyoFullBody",
            c3d_fit_model_path=str(tmp_path / "missing_soma"),
            converted_c3d_name="Study/Trial",
        )


def test_retarget_c3d_rejects_cache_from_changed_fitting_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_root = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_root))
    c3d_file = tmp_path / "AllSteps.c3d"
    c3d_file.write_bytes(b"same source")
    motion_path = converted_root / "MyoFullBody" / "gmr" / "Study" / "Trial.npz"
    analysis_path = motion_path.with_name("Trial_analysis.npz")
    motion_path.parent.mkdir(parents=True)
    motion_path.write_bytes(b"cached trajectory")
    np.savez(
        analysis_path,
        source_c3d_sha256=np.asarray(c3d_pipeline._file_sha256(c3d_file)),
        c3d_cache_key_sha256=np.asarray("0" * 64),
    )

    with pytest.raises(RuntimeError, match="different fitting or retargeting configuration"):
        c3d_pipeline.retarget_c3d_to_trajectory(
            str(c3d_file),
            "MyoFullBody",
            c3d_fit_model_path=str(tmp_path / "missing_soma"),
            converted_c3d_name="Study/Trial",
        )


def test_retarget_c3d_reports_invalid_converted_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    _install_fake_retargeting(monkeypatch, tmp_path, calls)
    converted_c3d_path = tmp_path / "converted_c3d"
    monkeypatch.setenv("MUSCLEMIMIC_CONVERTED_C3D_PATH", str(converted_c3d_path))

    c3d_file = tmp_path / "AllSteps.c3d"
    c3d_file.write_bytes(b"c3d")
    motion_path = converted_c3d_path / "MyoFullBody" / "gmr" / "Study" / "Trial.npz"
    analysis_path = motion_path.with_name("Trial_analysis.npz")
    motion_path.parent.mkdir(parents=True)
    motion_path.write_bytes(b"not a trajectory")
    monkeypatch.setattr(c3d_pipeline, "_c3d_cache_key", lambda *_args, **_kwargs: "expected key")
    np.savez(
        analysis_path,
        source_c3d_sha256=np.asarray(c3d_pipeline._file_sha256(c3d_file)),
        c3d_cache_key_sha256=np.asarray(c3d_pipeline._cache_key_sha256("expected key")),
    )

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

    c3d_file = tmp_path / "AllSteps.c3d"
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
