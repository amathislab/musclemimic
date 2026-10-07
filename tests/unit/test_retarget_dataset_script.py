import csv
import os
from pathlib import Path


from loco_mujoco.smpl.motion_spec import AMASSMotionSpec

_SCRIPT_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "MUJOCO_GL",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)
_ORIGINAL_ENV = {key: os.environ.get(key) for key in _SCRIPT_ENV_KEYS}
from scripts.retarget_dataset import (  # noqa: E402
    BIMANUAL_TRAINING_AND_TESTING,
    build_gmr_params,
    get_gmr_motion_cache_path,
    parse_arguments,
    write_failure_report,
)

for _key, _value in _ORIGINAL_ENV.items():
    if _value is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _value


def test_get_gmr_motion_cache_path_for_full_motion(tmp_path: Path) -> None:
    motion = AMASSMotionSpec("CMU/CMU/01/01_01_poses")

    cache_path = get_gmr_motion_cache_path(motion, "MyoFullBody", cache_root=tmp_path)

    assert cache_path == tmp_path / "MyoFullBody/gmr_v6/CMU/CMU/01/01_01_poses.npz"


def test_get_gmr_motion_cache_path_for_bounded_motion(tmp_path: Path) -> None:
    motion = AMASSMotionSpec("CMU/CMU/01/01_01_poses", start_time_s=1.0, end_time_s=3.0)

    cache_path = get_gmr_motion_cache_path(motion, "MjxMyoFullBody", cache_root=tmp_path)

    assert cache_path == (tmp_path / f"MyoFullBody/gmr_v6/_clips/CMU/CMU/01/01_01_poses__{motion.cache_suffix}.npz")


def test_get_gmr_motion_cache_path_can_resolve_previous_namespace(tmp_path: Path) -> None:
    from loco_mujoco.smpl.retargeting import get_gmr_motion_cache_path as resolve_cache_path

    motion = AMASSMotionSpec("CMU/CMU/01/01_01_poses")

    cache_path = resolve_cache_path("MyoFullBody", motion, tmp_path, cache_namespace="gmr_v4")

    assert cache_path == tmp_path / "MyoFullBody/gmr_v4/CMU/CMU/01/01_01_poses.npz"


def test_gmr_params_enable_shape_fitting_on_request() -> None:
    args = parse_arguments(["--retargeting-method", "gmr", "--gmr-src-human", "smplh", "--gmr-use-fitted-shape"])

    assert build_gmr_params(args)["use_fitted_shape"] is True


def test_gmr_cli_keeps_existing_fitting_defaults() -> None:
    args = parse_arguments(["--retargeting-method", "gmr", "--gmr-src-human", "smplh"])

    assert build_gmr_params(args)["use_fitted_shape"] is False


def test_training_and_testing_batch_groups_are_exposed() -> None:
    assert BIMANUAL_TRAINING_AND_TESTING == (
        "AMASS_BIMANUAL_TRAIN_MOTIONS",
        "AMASS_BIMANUAL_TEST_MOTIONS",
    )

    bimanual_args = parse_arguments(["--dataset", *BIMANUAL_TRAINING_AND_TESTING])

    assert tuple(bimanual_args.dataset) == BIMANUAL_TRAINING_AND_TESTING


def test_write_failure_report_records_failed_motion_details(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "failed.csv"

    failure_count = write_failure_report(
        [
            (True, "KIT/3/success_poses", None),
            (False, "KIT/3/failure_poses", "FileNotFoundError: missing.npz"),
        ],
        output_path,
        model="MyoBimanualArm",
        dataset_groups=BIMANUAL_TRAINING_AND_TESTING,
        retargeting_method="gmr",
    )

    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert failure_count == 1
    assert rows == [
        {
            "motion": "KIT/3/failure_poses",
            "error": "FileNotFoundError: missing.npz",
            "model": "MyoBimanualArm",
            "dataset_groups": "AMASS_BIMANUAL_TRAIN_MOTIONS AMASS_BIMANUAL_TEST_MOTIONS",
            "retargeting_method": "gmr",
        }
    ]


def test_parse_arguments_accepts_failure_report_path(tmp_path: Path) -> None:
    output_path = tmp_path / "failed.csv"

    args = parse_arguments(["--failed-motions-csv", str(output_path)])

    assert args.failed_motions_csv == output_path
