import json
from pathlib import Path

import numpy as np
import pytest

import scripts.generate_gmr_quality_filtered_phc_split as generate_gmr_quality_filtered_phc_split_module
from loco_mujoco.smpl.motion_spec import AMASSMotionSpec
from scripts.generate_gmr_quality_filtered_phc_split import motion_cache_path


def test_nested_train_counts_use_final_accepted_population() -> None:
    assert generate_gmr_quality_filtered_phc_split_module._nested_train_counts(7_944) == {
        "0.05": 397,
        "0.3": 2_383,
        "0.5": 3_972,
        "1.0": 7_944,
    }


def test_nested_train_counts_require_nonempty_train_set() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_gmr_quality_filtered_phc_split_module._nested_train_counts(0)


def test_classify_motion_cache_requires_trajectory_and_analysis(tmp_path: Path) -> None:
    cache_path = tmp_path / "motion.npz"
    analysis_path = tmp_path / "motion_analysis.npz"
    rejection_path = tmp_path / "motion_quality_rejected.json"

    assert generate_gmr_quality_filtered_phc_split_module._classify_motion_cache(cache_path) == "missing"
    cache_path.touch()
    assert generate_gmr_quality_filtered_phc_split_module._classify_motion_cache(cache_path) == "incomplete"
    analysis_path.touch()
    assert generate_gmr_quality_filtered_phc_split_module._classify_motion_cache(cache_path) == "accepted"
    rejection_path.write_text("{}")
    assert generate_gmr_quality_filtered_phc_split_module._classify_motion_cache(cache_path) == "conflict"

    cache_path.unlink()
    analysis_path.unlink()
    assert generate_gmr_quality_filtered_phc_split_module._classify_motion_cache(cache_path) == "rejected"


def test_rejection_kind_preserves_every_production_gate(tmp_path: Path) -> None:
    rejection_path = tmp_path / "motion_quality_rejected.json"
    for reason, expected in (
        ("GMR floor contact quality check failed", "floor"),
        ("GMR tracking quality check failed", "tracking"),
        ("GMR retained-output continuity quality check failed", "continuity"),
        ("GMR first-frame initialization did not converge", "initialization"),
    ):
        rejection_path.write_text(json.dumps({"reason": reason}))
        assert generate_gmr_quality_filtered_phc_split_module._rejection_kind(rejection_path) == expected


def test_motion_cache_path_preserves_clip_identity_without_mujoco_import(tmp_path: Path) -> None:
    motion = AMASSMotionSpec("CMU/01/01_01_poses", 1.0, 2.0)

    path = motion_cache_path(tmp_path, "MjxMyoFullBody", "gmr_v6", motion)

    assert path == (tmp_path / "MyoFullBody" / "gmr_v6" / "_clips" / f"CMU/01/01_01_poses__{motion.cache_suffix}.npz")


def test_accepted_analysis_is_recomputed_at_requested_threshold(tmp_path: Path) -> None:
    analysis_path = tmp_path / "motion_analysis.npz"
    np.savez(
        analysis_path,
        floor_surface_clearance_m=np.asarray([0.0, 0.12]),
        floating_clearance_threshold_m=np.asarray(0.05),
        floating_frame_fraction=np.asarray(0.5),
    )

    stored_threshold_m, floating_fraction = generate_gmr_quality_filtered_phc_split_module._validate_accepted_analysis(
        analysis_path,
        clearance_threshold_m=0.11,
        max_floating_frame_fraction=0.5,
    )

    assert stored_threshold_m == pytest.approx(0.05)
    assert floating_fraction == pytest.approx(0.5)
    with pytest.raises(ValueError, match="recomputed floating fraction"):
        generate_gmr_quality_filtered_phc_split_module._validate_accepted_analysis(
            analysis_path,
            clearance_threshold_m=0.11,
            max_floating_frame_fraction=0.1,
        )


def test_accepted_analysis_ignores_clearance_outside_source_contact(tmp_path: Path) -> None:
    analysis_path = tmp_path / "motion_analysis.npz"
    np.savez(
        analysis_path,
        floor_surface_clearance_m=np.asarray([0.0, 0.25]),
        source_contact_frame_clearance_m=np.asarray([0.01, np.nan]),
        floating_clearance_threshold_m=np.asarray(0.11),
        floating_frame_fraction=np.asarray(0.0),
    )

    _stored_threshold_m, floating_fraction = generate_gmr_quality_filtered_phc_split_module._validate_accepted_analysis(
        analysis_path,
        clearance_threshold_m=0.11,
        max_floating_frame_fraction=0.1,
    )

    assert floating_fraction == 0.0


def test_accepted_analysis_allows_no_source_contact_with_zero_fraction(tmp_path: Path) -> None:
    analysis_path = tmp_path / "motion_analysis.npz"
    np.savez(
        analysis_path,
        source_contact_frame_clearance_m=np.asarray([np.nan, np.nan]),
        floating_clearance_threshold_m=np.asarray(0.11),
        floating_frame_fraction=np.asarray(0.0),
    )

    _stored_threshold_m, floating_fraction = generate_gmr_quality_filtered_phc_split_module._validate_accepted_analysis(
        analysis_path,
        clearance_threshold_m=0.11,
        max_floating_frame_fraction=0.1,
    )

    assert floating_fraction == 0.0


def test_accepted_legacy_analysis_uses_stricter_scalar_evidence(tmp_path: Path) -> None:
    analysis_path = tmp_path / "motion_analysis.npz"
    np.savez(
        analysis_path,
        floating_clearance_threshold_m=np.asarray(0.02),
        floating_frame_fraction=np.asarray(0.08),
    )

    stored_threshold_m, floating_fraction = generate_gmr_quality_filtered_phc_split_module._validate_accepted_analysis(
        analysis_path,
        clearance_threshold_m=0.11,
        max_floating_frame_fraction=0.1,
    )

    assert stored_threshold_m == pytest.approx(0.02)
    assert floating_fraction == pytest.approx(0.08)


def test_floor_rejection_marker_must_match_frozen_gate(tmp_path: Path) -> None:
    motion = AMASSMotionSpec("CMU/01/01_01_poses", 1.0, 2.0)
    rejection_path = tmp_path / "motion_quality_rejected.json"
    payload = {
        "cache_namespace": "gmr_v6",
        "cache_key": motion.cache_key,
        "reason": "GMR floor contact quality check failed",
        "metrics": {
            "floating_clearance_threshold_m": 0.11,
            "max_floating_frame_fraction": 0.1,
            "floating_frame_fraction": 0.2,
        },
    }
    rejection_path.write_text(json.dumps(payload))

    assert (
        generate_gmr_quality_filtered_phc_split_module._validate_rejection_marker(
            rejection_path,
            motion,
            cache_namespace="gmr_v6",
            clearance_threshold_m=0.11,
            max_floating_frame_fraction=0.1,
        )
        == "floor"
    )
