"""Regression coverage for retargeting integration and cache compatibility."""

import inspect
import json
import logging

import pytest

from loco_mujoco.smpl import retargeting
from loco_mujoco.smpl.motion_spec import AMASSMotionSpec
from loco_mujoco.task_factories import AMASSDatasetConf, dataset_confs
from musclemimic.utils import gmr_cache


@pytest.mark.parametrize(
    "loader",
    (retargeting.load_retargeted_amass_trajectory, retargeting.load_retargeted_amass_trajectory_set),
)
def test_legacy_loader_positional_clear_cache(loader):
    arguments = inspect.signature(loader).bind("MyoFullBody", "motion", None, "gmr", None, True)
    assert arguments.arguments["clear_cache"] is True


def test_legacy_dataset_config_positional_sampling():
    config = AMASSDatasetConf("motion", None, "gmr", None, 7, 42, True)
    assert config.max_motions == 7
    assert config.max_motions_seed == 42
    assert config.clear_cache is True


def test_phc_group_download_preserves_clip_identity(tmp_path, monkeypatch):
    motion = AMASSMotionSpec("CMU/walk", start_time_s=0.5, end_time_s=2.0)
    monkeypatch.setattr(dataset_confs, "get_amass_dataset_groups", lambda: {"CLIPS": [motion.path]})
    monkeypatch.setattr(dataset_confs, "get_amass_motion_spec_groups", lambda: {"CLIPS": [motion]})
    source = tmp_path / "download.npz"
    source.write_bytes(b"bounded trajectory")
    filenames = []

    def download(**kwargs):
        filenames.append(kwargs["filename"])
        return source

    monkeypatch.setattr(gmr_cache, "hf_hub_download", download)
    cache_root = tmp_path / "cache"
    paths = gmr_cache.download_gmr_dataset_group("CLIPS", cache_dir=cache_root)
    expected = retargeting.get_gmr_motion_cache_path("MyoFullBody", motion, cache_root)
    assert paths == [expected]
    assert filenames == [expected.relative_to(cache_root).as_posix()]
    assert expected.read_bytes() == source.read_bytes()


def test_persistent_rejection_does_not_trigger_remote_download(tmp_path, monkeypatch):
    motion = AMASSMotionSpec("CMU/walk")
    cache_path = retargeting.get_gmr_motion_cache_path("MyoFullBody", motion, tmp_path)
    rejection = retargeting.gmr_quality_rejection_path(cache_path)
    rejection.parent.mkdir(parents=True)
    rejection.write_text(json.dumps({"reason": "GMR tracking quality check failed", "metrics": {}}))
    monkeypatch.setattr(retargeting, "get_converted_amass_dataset_path", lambda: str(tmp_path))

    def unexpected_download(**_kwargs):
        pytest.fail("Persistent rejection must be checked before any download")

    monkeypatch.setattr(retargeting, "try_download_gmr_cache", unexpected_download)
    with pytest.raises(retargeting.GMRQualityGateError, match="Previously rejected"):
        retargeting.load_retargeted_amass_trajectory_set("MyoFullBody", motion, robot_conf={}, retargeting_method="gmr")


def test_batch_loader_does_not_accept_rejected_cached_motion(tmp_path, monkeypatch):
    motion = AMASSMotionSpec("CMU/walk")
    cache_path = retargeting.get_gmr_motion_cache_path("MyoFullBody", motion, tmp_path)
    cache_path.parent.mkdir(parents=True)
    cache_path.touch()
    retargeting.gmr_quality_rejection_path(cache_path).write_text(
        json.dumps({"reason": "GMR tracking quality check failed", "metrics": {}})
    )
    monkeypatch.setattr(retargeting, "get_converted_amass_dataset_path", lambda: str(tmp_path))

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("Rejected cache must not enter a loaded trajectory set")

    monkeypatch.setattr(retargeting.Trajectory, "load", unexpected_load)
    with pytest.raises(RuntimeError, match=r"no valid .* trajectories"):
        retargeting._load_trajectories_individually(
            "MyoFullBody", [motion], {}, logging.getLogger(__name__), retargeting_method="gmr"
        )


def test_bimanual_persists_source_rejection_and_retries_only_when_cleared(tmp_path, monkeypatch):
    motion = AMASSMotionSpec("CMU/walk", end_time_s=2.0)
    monkeypatch.setattr(retargeting, "get_converted_amass_dataset_path", lambda: str(tmp_path))
    monkeypatch.setattr(retargeting, "detect_headless_environment", lambda: False)
    calls = []

    def reject_source(**kwargs):
        assert kwargs["dataset_name"] == motion
        calls.append(kwargs.get("clear_cache", False))
        raise retargeting.GMRQualityGateError("GMR tracking quality check failed", {})

    monkeypatch.setattr(retargeting, "load_retargeted_amass_trajectory", reject_source)
    for clear_cache, expected_calls in ((False, [False]), (False, [False]), (True, [False, True])):
        with pytest.raises(RuntimeError, match="no valid trajectories"):
            retargeting.retarget_smpl_to_bimanual_via_intermediate_set(
                motion,
                robot_conf_skeleton={},
                robot_conf_bimanual={},
                retargeting_method="gmr",
                clear_cache=clear_cache,
            )
        rejection = retargeting.gmr_quality_rejection_path(
            retargeting.get_gmr_motion_cache_path("MyoBimanualArm", motion, tmp_path)
        )
        assert rejection.exists()
        assert json.loads(rejection.read_text())["cache_key"] == motion.cache_key
        assert calls == expected_calls
