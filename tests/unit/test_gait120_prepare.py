from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from musclemimic.web_viewer.c3d.markers import load_marker_archive, load_marker_trajectory
from scripts.gait120.merge_staging import merge_shard
from scripts.gait120.prepare_smplh import (
    EXPECTED_EMG_CHANNELS,
    ClipRecord,
    _combine_paired_steps,
    _filter_clips_by_selection_manifest,
    _summarize_marker_fit_quality,
    _validate_emg_file,
    load_gait120_trc,
)


def _write_trc(path: Path, frames: list[tuple[int, float, list[float]]]) -> None:
    rows = [
        "PathFileType\t4\t(X/Y/Z)\ttest.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"100\t100\t{len(frames)}\t2\tmm\t100\t{frames[0][0]}\t{len(frames)}",
        "Frame#\tTime\tLFHD\t\t\tRFHD\t\t",
        "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
    ]
    rows.extend(f"{frame}\t{time:.2f}\t" + "\t".join(map(str, values)) for frame, time, values in frames)
    path.write_text("\n".join(rows) + "\n")


def test_load_gait120_trc_converts_y_up_mm_to_z_up_metres(tmp_path: Path) -> None:
    path = tmp_path / "Step01.trc"
    _write_trc(path, [(10, 0.1, [100, 2000, 300, -100, 1500, -400])])

    trc = load_gait120_trc(path)

    assert trc.labels == ("LFHD", "RFHD")
    assert trc.fps == 100.0
    np.testing.assert_array_equal(trc.frame_numbers, [10])
    np.testing.assert_allclose(
        trc.positions[0],
        np.array([[0.1, -0.3, 2.0], [-0.1, 0.4, 1.5]], dtype=np.float32),
    )


def test_combine_paired_steps_deduplicates_shared_boundary_frame(tmp_path: Path) -> None:
    first = tmp_path / "Step01.trc"
    second = tmp_path / "Step02.trc"
    _write_trc(
        first,
        [
            (10, 0.10, [100, 2000, 300, -100, 1500, -400]),
            (11, 0.11, [110, 2000, 310, -110, 1500, -410]),
        ],
    )
    _write_trc(
        second,
        [
            (11, 0.11, [110, 2000, 310, -110, 1500, -410]),
            (12, 0.12, [120, 2000, 320, -120, 1500, -420]),
        ],
    )

    positions, labels, fps, frames, steps = _combine_paired_steps([first, second])

    assert positions.shape == (3, 2, 3)
    assert labels == ("LFHD", "RFHD")
    assert fps == 100.0
    np.testing.assert_array_equal(frames, [10, 11, 12])
    np.testing.assert_array_equal(steps, [1, 1, 2])


def test_marker_archive_is_accepted_by_generic_fitter_loader(tmp_path: Path) -> None:
    path = tmp_path / "AllSteps_markers.npz"
    positions = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    np.savez(path, positions=positions, labels=np.array(["LFHD", "RFHD"]), fps=np.array(100.0))

    loaded_positions, labels, fps = load_marker_archive(path)
    generic_positions, generic_labels, generic_fps = load_marker_trajectory(path)

    np.testing.assert_array_equal(loaded_positions, positions)
    np.testing.assert_array_equal(generic_positions, positions)
    assert labels == generic_labels == ["LFHD", "RFHD"]
    assert fps == generic_fps == 100.0


def test_validate_emg_archive_checks_pairing_and_shape(tmp_path: Path) -> None:
    path = tmp_path / "AllSteps_emg.npz"
    np.savez(
        path,
        emg=np.zeros((2, 101, len(EXPECTED_EMG_CHANNELS)), dtype=np.float32),
        channels=np.asarray(EXPECTED_EMG_CHANNELS),
        steps=np.array([1, 2], dtype=np.int8),
        samples_per_step=np.array(101),
        source_mat=np.array("ConvertedData.mat"),
    )

    metadata = _validate_emg_file(path, expected_steps="1;2")

    assert metadata == {"steps": 2, "samples": 202}


def test_marker_fit_quality_flags_robust_outlier_and_missing_metric() -> None:
    clips = [
        SimpleNamespace(motion="a", marker_error_mean_mm=9.8),
        SimpleNamespace(motion="b", marker_error_mean_mm=10.0),
        SimpleNamespace(motion="c", marker_error_mean_mm=10.2),
        SimpleNamespace(motion="outlier", marker_error_mean_mm=25.0),
        SimpleNamespace(motion="missing", marker_error_mean_mm=None),
    ]

    quality = _summarize_marker_fit_quality(clips)

    assert quality["clips_with_metrics"] == 4
    assert quality["missing_metrics"] == ["missing"]
    assert [row["motion"] for row in quality["outliers"]] == ["outlier"]
    assert not quality["ready"]


def test_selection_manifest_keeps_retargets_and_calibrations(tmp_path: Path) -> None:
    def clip(motion: str, subject: int, movement: str, trial: int) -> ClipRecord:
        return ClipRecord(subject, movement, trial, "1;2", "m", "e", "o", motion, "e", "t", "m")

    clips = [
        clip("Gait120/S001/StairAscent/Trial02/AllSteps_stageii", 1, "StairAscent", 2),
        clip("Gait120/S001/LevelWalking/Trial01/AllSteps_stageii", 1, "LevelWalking", 1),
        clip("Gait120/S002/LevelWalking/Trial03/AllSteps_stageii", 2, "LevelWalking", 3),
    ]
    manifest = tmp_path / "selection.csv"
    manifest.write_text(
        "dataset,motion,calibration_motion\n"
        "gait120,Gait120/S001/StairAscent/Trial02/AllSteps_stageii,"
        "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii\n"
        "vielemeyer,Vielemeyer/ignored,\n"
    )

    selected = _filter_clips_by_selection_manifest(clips, manifest)

    assert {item.motion for item in selected} == {
        "Gait120/S001/StairAscent/Trial02/AllSteps_stageii",
        "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii",
    }


def _write_staging_shard(root: Path, destination: Path) -> None:
    subject = root / "Gait120" / "S061" / "LevelWalking" / "Trial01"
    subject.mkdir(parents=True)
    (subject / "AllSteps_stageii.npz").write_bytes(b"motion")
    marker = destination / ".markers" / "Gait120" / "S061" / "LevelWalking" / "Trial01"
    marker.mkdir(parents=True)
    (marker / "AllSteps_markers.npz").write_bytes(b"markers")
    state = root / ".stage1" / "S061.npz"
    state.parent.mkdir()
    state.write_bytes(b"state")
    metadata = {
        "source_marker_archive": str(
            root / ".markers" / "Gait120" / "S061" / "LevelWalking" / "Trial01"
            / "AllSteps_markers.npz"
        ),
        "stage1_state": str(state),
    }
    (subject / "AllSteps_stageii.fit.json").write_text(json.dumps(metadata))
    (root / "validation.json").write_text(json.dumps({"paired_dataset_ready": True}))


def test_merge_staging_augments_emg_subject_and_refuses_motion_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "canonical"
    _write_staging_shard(source, destination)
    emg = destination / "Gait120" / "S061" / "LevelWalking" / "Trial01" / "AllSteps_emg.npz"
    emg.parent.mkdir(parents=True)
    emg.write_bytes(b"canonical-emg")

    assert merge_shard(source, destination) == ["S061"]
    metadata_path = (
        destination
        / "Gait120"
        / "S061"
        / "LevelWalking"
        / "Trial01"
        / "AllSteps_stageii.fit.json"
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source_marker_archive"].startswith(str(destination.resolve()))
    assert metadata["stage1_state"] == str((destination / ".stage1" / "S061.npz").resolve())
    assert (destination / ".stage1" / "S061.npz").read_bytes() == b"state"
    assert emg.read_bytes() == b"canonical-emg"

    with pytest.raises(FileExistsError, match="S061"):
        merge_shard(source, destination)

    source_motion = (
        source / "Gait120" / "S061" / "LevelWalking" / "Trial01" / "AllSteps_stageii.npz"
    )
    source_motion.write_bytes(b"improved-motion")
    assert merge_shard(source, destination, replace_existing=True) == ["S061"]
    assert metadata_path.with_name("AllSteps_stageii.npz").read_bytes() == b"improved-motion"
    assert emg.read_bytes() == b"canonical-emg"
