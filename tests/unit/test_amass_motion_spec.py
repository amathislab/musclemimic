import numpy as np
import pytest

from loco_mujoco.smpl import retargeting
from loco_mujoco.smpl.motion_spec import AMASSMotionSpec, as_amass_motion_spec


def test_motion_spec_normalizes_paths_and_converts_time_bounds_to_frames():
    spec = AMASSMotionSpec("CMU/01/walk_poses.npz", start_time_s=0.5, end_time_s=2.0)

    assert spec.path == "CMU/01/walk_poses"
    assert spec.frame_bounds(frequency_hz=60.0, num_frames=200) == (30, 120)
    assert not spec.is_full_motion
    assert spec.cache_key.startswith("CMU/01/walk_poses#t=")
    assert len(spec.cache_suffix) == 16


def test_motion_spec_full_motion_keeps_the_legacy_cache_identity():
    spec = as_amass_motion_spec("KIT/1/walk_poses")

    assert spec.is_full_motion
    assert spec.cache_key == spec.path
    assert spec.frame_bounds(frequency_hz=30.0, num_frames=90) == (0, 90)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"path": "/absolute/path"},
        {"path": "../outside"},
        {"path": "motion", "start_time_s": -1.0},
        {"path": "motion", "start_time_s": 1.0, "end_time_s": 1.0},
    ),
)
def test_motion_spec_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        AMASSMotionSpec(**kwargs)


def test_motion_spec_rejects_an_empty_frame_interval():
    spec = AMASSMotionSpec("motion", start_time_s=3.0, end_time_s=4.0)

    with pytest.raises(ValueError, match="selects no frames"):
        spec.frame_bounds(frequency_hz=30.0, num_frames=60)


def test_amass_loader_applies_motion_time_bounds(tmp_path, monkeypatch):
    motion_path = tmp_path / "CMU" / "01" / "walk_poses.npz"
    motion_path.parent.mkdir(parents=True)
    poses = np.arange(10 * 72, dtype=np.float32).reshape(10, 72)
    translations = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
    np.savez(
        motion_path,
        poses=poses,
        trans=translations,
        betas=np.zeros(16, dtype=np.float32),
        gender="neutral",
        mocap_framerate=10.0,
    )
    monkeypatch.setattr(retargeting, "get_amass_dataset_path", lambda: str(tmp_path))

    loaded = retargeting.load_amass_data(
        "CMU/01/walk_poses",
        motion_spec=AMASSMotionSpec(
            "CMU/01/walk_poses",
            start_time_s=0.2,
            end_time_s=0.7,
        ),
    )

    np.testing.assert_array_equal(loaded["trans"], translations[2:7])
    np.testing.assert_array_equal(loaded["pose_aa"][:, :66], poses[2:7, :66])
    assert loaded["pose_aa"].shape == (5, 72)
