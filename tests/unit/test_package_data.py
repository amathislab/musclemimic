"""Required runtime resources in the package configuration."""

import tomllib
from pathlib import Path


def test_amass_motion_manifest_is_declared_as_package_data():
    repository = Path(__file__).resolve().parents[2]
    with (repository / "pyproject.toml").open("rb") as stream:
        config = tomllib.load(stream)

    resources = config["tool"]["setuptools"].get("package-data", {}).get("loco_mujoco.smpl", [])
    for name in (
        "amass_cleaned_training_motions.txt",
        "amass_phc_cleaned_split_seed42.tsv",
        "amass_phc_gmr_v6_quality_split.tsv",
        "gmr_configs/smplh_to_myofullbody.json",
    ):
        assert name in resources
        assert (repository / "loco_mujoco/smpl" / name).is_file()
