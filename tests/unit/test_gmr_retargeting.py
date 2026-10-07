from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import loco_mujoco.smpl.retargeting as retargeting_module
from loco_mujoco.smpl.motion_spec import AMASSMotionSpec
from loco_mujoco.smpl.retargeting import (
    GMR_EQUALITY_CONSTRAINT_COST,
    GMR_FLOATING_CLEARANCE_THRESHOLD_M,
    GMRQualityGateError,
)
from loco_mujoco.trajectory import Trajectory, TrajectoryData
from musclemimic.utils import gmr_cache

GMR_CONFIG_PATH = Path(__file__).parents[2] / "loco_mujoco" / "smpl" / "gmr_configs" / "smplh_to_myofullbody.json"


def test_default_floating_clearance_threshold_is_eleven_centimeters() -> None:
    assert GMR_FLOATING_CLEARANCE_THRESHOLD_M == pytest.approx(0.11)


def test_clear_gmr_terminal_artifacts_removes_complete_triplet(tmp_path: Path) -> None:
    cache_path = tmp_path / "motion.npz"
    analysis_path = tmp_path / "motion_analysis.npz"
    rejection_path = tmp_path / "motion_quality_rejected.json"
    for path in (cache_path, analysis_path, rejection_path):
        path.touch()

    retargeting_module._clear_gmr_terminal_artifacts(cache_path)

    assert not cache_path.exists()
    assert not analysis_path.exists()
    assert not rejection_path.exists()


def test_gmr_cache_path_isolates_behavior_affecting_configuration(tmp_path: Path) -> None:
    motion = AMASSMotionSpec("CMU/01/01_01_poses")

    canonical_path = retargeting_module.get_gmr_motion_cache_path(
        "MyoFullBody",
        motion,
        tmp_path,
    )
    explicit_defaults_path = retargeting_module.get_gmr_motion_cache_path(
        "MyoFullBody",
        motion,
        tmp_path,
        gmr_config={"target_fps": 30, "solver": "daqp", "verbose": True},
    )
    changed_path = retargeting_module.get_gmr_motion_cache_path(
        "MyoFullBody",
        motion,
        tmp_path,
        gmr_config={"target_fps": 60},
    )

    assert canonical_path == tmp_path / "MyoFullBody/gmr_v6/CMU/01/01_01_poses.npz"
    assert explicit_defaults_path == canonical_path
    assert changed_path.parents[3].name == "_variants"
    assert len(changed_path.parents[2].name) == 64
    assert changed_path != canonical_path


def _model_with_joint_equalities() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint name="joint_a" type="hinge"/>
              <geom type="sphere" size="0.01"/>
              <body>
                <joint name="joint_b" type="hinge"/>
                <geom type="sphere" size="0.01"/>
                <body>
                  <joint name="joint_c" type="hinge"/>
                  <geom type="sphere" size="0.01"/>
                </body>
              </body>
            </body>
          </worldbody>
          <equality>
            <joint joint1="joint_a" joint2="joint_b" polycoef="0 1 0 0 0"/>
            <joint joint1="joint_b" joint2="joint_c" polycoef="0 1 0 0 0"/>
          </equality>
        </mujoco>
        """
    )


def test_configure_gmr_equality_tasks_consolidates_and_weights_constraints() -> None:
    mink = pytest.importorskip("mink")
    pytest.importorskip("general_motion_retargeting")
    from mink.tasks.equality_constraint_task import EqualityConstraintTask

    model = _model_with_joint_equalities()
    frame_task_1 = object()
    frame_task_2 = object()
    retarget = SimpleNamespace(
        configuration=mink.Configuration(model),
        tasks1=[EqualityConstraintTask(model, cost=5.0), frame_task_1],
        tasks2=[
            EqualityConstraintTask(model, cost=5.0),
            EqualityConstraintTask(model, cost=10.0),
            frame_task_2,
        ],
    )

    retargeting_module._configure_gmr_equality_tasks(retarget)

    for tasks, frame_task in ((retarget.tasks1, frame_task_1), (retarget.tasks2, frame_task_2)):
        equality_tasks = [task for task in tasks if isinstance(task, EqualityConstraintTask)]
        assert len(equality_tasks) == 1
        np.testing.assert_allclose(equality_tasks[0].cost, GMR_EQUALITY_CONSTRAINT_COST)
        assert frame_task in tasks


def test_configure_gmr_stage2_position_costs_applies_named_override() -> None:
    mink = pytest.importorskip("mink")
    pytest.importorskip("general_motion_retargeting")
    frame_task = mink.FrameTask(
        frame_name="ulna_l",
        frame_type="body",
        position_cost=0.0,
        orientation_cost=5.0,
    )
    retarget = SimpleNamespace(tasks2=[frame_task])

    retargeting_module._configure_gmr_stage2_position_costs(retarget, {"ulna_l": 40.0})

    np.testing.assert_allclose(frame_task.cost[:3], 40.0)


def test_configure_gmr_stage2_position_costs_rejects_unknown_frame() -> None:
    pytest.importorskip("mink")
    pytest.importorskip("general_motion_retargeting")
    retarget = SimpleNamespace(tasks2=[])

    with pytest.raises(ValueError, match="Unknown GMR stage-two frame task"):
        retargeting_module._configure_gmr_stage2_position_costs(retarget, {"missing": 40.0})


def test_myofullbody_stage2_arm_position_costs_are_bilaterally_symmetric() -> None:
    with GMR_CONFIG_PATH.open(encoding="utf-8") as config_file:
        stage2 = json.load(config_file)["ik_match_table2"]

    for left_frame, right_frame in (("ulna_l", "ulna_r"), ("lunate_l", "lunate_r")):
        assert stage2[left_frame][1] == stage2[right_frame][1] == 40


def test_download_gmr_cache_uses_versioned_namespace(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloaded_file = tmp_path / "downloaded.npz"
    downloaded_file.write_bytes(b"cache")
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        return str(downloaded_file)

    monkeypatch.setattr(gmr_cache, "hf_hub_download", fake_download)
    cache_root = tmp_path / "cache"
    result = gmr_cache.download_gmr_cache(
        "Dataset/Subject/Motion",
        env_name="MyoFullBody",
        cache_dir=cache_root,
    )

    expected_relative_path = "MyoFullBody/gmr_v6/Dataset/Subject/Motion.npz"
    assert calls["filename"] == expected_relative_path
    assert result == cache_root / expected_relative_path
    assert result.read_bytes() == b"cache"


class _FakeConfiguration:
    def __init__(self) -> None:
        self.updated_qpos = None

    def update(self, *, q) -> None:
        self.updated_qpos = np.asarray(q).copy()


class _FakeRetarget:
    def __init__(self) -> None:
        self.configuration = _FakeConfiguration()
        self.visited_frames = []

    def retarget(self, frame, *, offset_to_ground):
        del offset_to_ground
        frame_index = int(frame)
        self.visited_frames.append(frame_index)
        return np.array([100.0 + frame_index]), np.array([0.02, 0.03])


class _ConvergingRetarget:
    def __init__(self, qpos_values: list[float]) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body>
                  <joint name="joint" type="hinge"/>
                  <geom type="sphere" size="0.01"/>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        self.configuration = SimpleNamespace(model=model)
        self._qpos_values = iter(qpos_values)
        self.num_calls = 0

    def retarget(self, frame, *, offset_to_ground):
        del frame, offset_to_ground
        self.num_calls += 1
        return np.array([next(self._qpos_values)]), np.array([0.02, 0.03])


def test_solve_gmr_initial_frame_waits_for_consecutive_qpos_convergence() -> None:
    retarget = _ConvergingRetarget([0.4, 0.2, 0.1, 0.05, 0.025, 0.0255, 0.0257])

    qpos, distances_m, metrics = retargeting_module._solve_gmr_initial_frame(
        retarget,
        frame=object(),
        offset_to_ground=False,
        minimum_passes=5,
        maximum_passes=10,
        qpos_delta_tolerance=1e-3,
        consecutive_converged_passes=2,
    )

    assert retarget.num_calls == 7
    np.testing.assert_allclose(qpos, [0.0257])
    np.testing.assert_allclose(distances_m, [0.02, 0.03])
    assert metrics["initialization_converged"]
    assert metrics["initialization_passes_used"] == 7
    assert metrics["initialization_max_qpos_delta"] == pytest.approx(0.0002)


def test_solve_gmr_initial_frame_reports_nonconverged_initialization() -> None:
    retarget = _ConvergingRetarget([0.1, 0.2, 0.3, 0.4, 0.5])

    qpos, distances_m, metrics = retargeting_module._solve_gmr_initial_frame(
        retarget,
        frame=object(),
        offset_to_ground=False,
        minimum_passes=2,
        maximum_passes=5,
        qpos_delta_tolerance=1e-3,
        consecutive_converged_passes=2,
    )

    np.testing.assert_allclose(qpos, [0.5])
    np.testing.assert_allclose(distances_m, [0.02, 0.03])
    assert metrics["initialization_converged"] is False
    assert metrics["initialization_passes_used"] == 5
    assert metrics["initialization_max_qpos_delta"] == pytest.approx(0.1)


def test_refine_bad_gmr_prefix_solves_backward_from_stable_anchor() -> None:
    retarget = _FakeRetarget()
    qpos = np.arange(5.0)[:, None]
    distances_m = np.array(
        [
            [0.20, 0.30],
            [0.18, 0.22],
            [0.02, 0.03],
            [0.03, 0.04],
            [0.04, 0.05],
        ]
    )

    refined_qpos, refined_distances_m, refined = retargeting_module._refine_bad_gmr_prefix(
        retarget,
        np.arange(5),
        qpos,
        distances_m,
        aligned_fps=4.0,
        offset_to_ground=False,
        logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    )

    assert refined
    np.testing.assert_array_equal(retarget.configuration.updated_qpos, qpos[2])
    assert retarget.visited_frames == [1, 0]
    np.testing.assert_array_equal(refined_qpos[:, 0], [100.0, 101.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(refined_distances_m[:2], [[0.02, 0.03], [0.02, 0.03]])
    np.testing.assert_array_equal(refined_distances_m[2:], distances_m[2:])


def test_refine_bad_gmr_prefix_repairs_first_retained_frame() -> None:
    retarget = _FakeRetarget()
    qpos = np.arange(5.0)[:, None]
    distances_m = np.array(
        [
            [0.01, 0.11],
            [0.02, 0.03],
            [0.02, 0.03],
            [0.02, 0.03],
            [0.02, 0.03],
        ]
    )

    refined_qpos, _, refined = retargeting_module._refine_bad_gmr_prefix(
        retarget,
        np.arange(5),
        qpos,
        distances_m,
        aligned_fps=4.0,
        offset_to_ground=False,
        logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    )

    assert refined
    np.testing.assert_array_equal(retarget.configuration.updated_qpos, qpos[2])
    assert retarget.visited_frames == [1, 0]
    assert refined_qpos[1, 0] == pytest.approx(101.0)


def test_refine_bad_gmr_prefix_forces_recovery_after_nonconverged_initialization() -> None:
    retarget = _FakeRetarget()
    qpos = np.arange(5.0)[:, None]
    distances_m = np.full((5, 2), 0.05)

    refined_qpos, refined_distances_m, refined = retargeting_module._refine_bad_gmr_prefix(
        retarget,
        np.arange(5),
        qpos,
        distances_m,
        aligned_fps=4.0,
        offset_to_ground=False,
        logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
        force_refinement=True,
    )

    assert refined
    np.testing.assert_array_equal(retarget.configuration.updated_qpos, qpos[2])
    assert retarget.visited_frames == [1, 0]
    np.testing.assert_array_equal(refined_qpos[:, 0], [100.0, 101.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(refined_distances_m[:2], [[0.02, 0.03], [0.02, 0.03]])


def test_refine_bad_gmr_prefix_leaves_valid_motion_unchanged() -> None:
    retarget = _FakeRetarget()
    qpos = np.arange(3.0)[:, None]
    distances_m = np.full((3, 2), 0.05)

    result_qpos, result_distances_m, refined = retargeting_module._refine_bad_gmr_prefix(
        retarget,
        np.arange(3),
        qpos,
        distances_m,
        aligned_fps=30.0,
        offset_to_ground=False,
        logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    )

    assert not refined
    assert result_qpos is qpos
    assert result_distances_m is distances_m
    assert not retarget.visited_frames


def test_validate_gmr_tracking_quality_rejects_bad_first_frame() -> None:
    distances_m = np.full((10, 2), 0.02)
    distances_m[1] = 0.20

    with pytest.raises(GMRQualityGateError, match=r"first retained-frame mean=0\.2000 m") as error:
        retargeting_module._validate_gmr_tracking_quality(distances_m)

    assert error.value.metrics["first_retained_frame_mean_position_error_m"] == pytest.approx(0.20)
    assert error.value.metrics["max_first_retained_frame_mean_position_error_m"] == pytest.approx(0.10)


def test_validate_gmr_tracking_quality_rejects_bad_trajectory_mean() -> None:
    distances_m = np.full((10, 2), 0.12)
    distances_m[1] = 0.02

    with pytest.raises(GMRQualityGateError, match=r"retained trajectory mean=0\.1075 m") as error:
        retargeting_module._validate_gmr_tracking_quality(distances_m)

    assert error.value.metrics["retained_trajectory_mean_position_error_m"] == pytest.approx(0.1075)
    assert error.value.metrics["max_retained_trajectory_mean_position_error_m"] == pytest.approx(0.10)


def test_validate_gmr_floor_contact_quality_accepts_exact_ten_percent() -> None:
    distances_m = np.array([0.0] * 9 + [0.021])

    metrics = retargeting_module._validate_gmr_floor_contact_quality(
        distances_m,
        distances_m,
        source_contact_frame_fraction=1.0,
        floating_clearance_threshold_m=0.02,
        max_floating_frame_fraction=0.10,
    )

    assert metrics["floating_frame_fraction"] == pytest.approx(0.10)


def test_validate_gmr_floor_contact_quality_rejects_above_ten_percent() -> None:
    distances_m = np.array([0.0] * 8 + [0.021] * 2)

    with pytest.raises(
        GMRQualityGateError,
        match=r"source-contact mismatch fraction=0\.200000",
    ) as error:
        retargeting_module._validate_gmr_floor_contact_quality(
            distances_m,
            distances_m,
            source_contact_frame_fraction=1.0,
            floating_clearance_threshold_m=0.02,
            max_floating_frame_fraction=0.10,
        )

    assert error.value.metrics["maximum_surface_clearance_m"] == pytest.approx(0.021)


def test_validate_gmr_floor_contact_quality_excludes_intended_flight() -> None:
    metrics = retargeting_module._validate_gmr_floor_contact_quality(
        np.array([0.0, 0.20]),
        np.array([0.0, np.nan]),
        source_contact_frame_fraction=0.50,
        floating_clearance_threshold_m=0.02,
        max_floating_frame_fraction=0.50,
    )

    assert metrics["floating_frame_fraction"] == pytest.approx(0.0)
    assert metrics["global_floating_frame_fraction"] == pytest.approx(0.50)


def test_validate_gmr_retained_continuity_accepts_calibrated_motion() -> None:
    qvel = np.zeros((100, 10), dtype=np.float64)
    qvel[:10, 0] = 4.0
    qvel[:10, 3] = 7.0

    metrics = retargeting_module._validate_gmr_retained_continuity(qvel, frequency_hz=100.0)

    assert metrics["leading_continuity_frames"] == 10
    assert metrics["leading_maximum_root_linear_speed_m_s"] == pytest.approx(4.0)
    assert metrics["leading_maximum_root_angular_speed_rad_s"] == pytest.approx(7.0)


def test_validate_gmr_retained_continuity_rejects_leading_root_spin() -> None:
    qvel = np.zeros((100, 10), dtype=np.float64)
    qvel[0, 3] = 15.58

    with pytest.raises(GMRQualityGateError, match=r"root angular speed=15\.5800 rad/s"):
        retargeting_module._validate_gmr_retained_continuity(qvel, frequency_hz=100.0)


def test_resampled_floor_guard_lifts_qpos_without_changing_qvel() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 0.1" contype="1" conaffinity="1"/>
            <body>
              <freejoint name="root"/>
              <geom type="sphere" size="0.1" contype="1" conaffinity="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    qpos = np.repeat(model.qpos0[None], 3, axis=0)
    qpos[:, 2] = [0.20, 0.05, 0.30]
    qvel = np.arange(3 * model.nv, dtype=np.float64).reshape(3, model.nv)
    trajectory = Trajectory(
        info=SimpleNamespace(frequency=100.0),
        data=TrajectoryData(qpos=qpos, qvel=qvel, split_points=np.array([0, 3])),
    )

    guarded, metrics = retargeting_module._guard_gmr_resampled_floor_clearance(
        trajectory,
        model,
        "root",
        excluded_body_subtree_names=(),
    )

    assert metrics["output_floor_z_translation_m"] == pytest.approx(0.050001)
    assert metrics["output_minimum_surface_clearance_before_m"] == pytest.approx(-0.05)
    assert metrics["output_minimum_surface_clearance_after_m"] == pytest.approx(1e-6)
    np.testing.assert_allclose(np.asarray(guarded.data.qpos)[:, 2], qpos[:, 2] + 0.050001)
    np.testing.assert_array_equal(np.asarray(guarded.data.qvel), qvel)
