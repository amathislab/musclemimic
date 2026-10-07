"""Distribution smoke coverage requiring uv and the declared build dependencies."""

import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.mark.integration
def test_motion_resources_survive_sdist_and_wheel_builds(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build distributions")

    repository = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repository / name, source / name)
    for name in ("loco_mujoco", "musclemimic", "fullbody", "bimanual"):
        shutil.copytree(repository / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    distributions = tmp_path / "dist"
    build = subprocess.run(
        [uv, "build", "--sdist", "--out-dir", str(distributions), str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    (sdist,) = distributions.glob("*.tar.gz")
    manifests = {
        "AMASS_CLEANED_TRAINING_MOTIONS": "loco_mujoco/smpl/amass_cleaned_training_motions.txt",
    }
    ik_config_path = "loco_mujoco/smpl/gmr_configs/smplh_to_myofullbody.json"
    resources = {
        path: (repository / path).read_bytes()
        for path in [
            *manifests.values(),
            ik_config_path,
            "loco_mujoco/smpl/amass_phc_cleaned_split_seed42.tsv",
            "loco_mujoco/smpl/amass_phc_gmr_v6_quality_split.tsv",
            "loco_mujoco/smpl/robot_confs/defaults.yaml",
            "loco_mujoco/smpl/robot_confs/MyoFullBody.yaml",
        ]
    }
    with tarfile.open(sdist) as archive:
        for path, expected in resources.items():
            member = archive.extractfile(f"{sdist.name.removesuffix('.tar.gz')}/{path}")
            assert member is not None
            assert member.read() == expected

    build = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(distributions), str(sdist)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    (wheel,) = distributions.glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        for path, expected in resources.items():
            assert archive.read(path) == expected
        archive.extractall(installed)

    # Execute the installed constants module without editable imports or optional SMPL dependencies.
    smoke = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json, runpy, sys; constants = runpy.run_path(sys.argv[1]); "
            "print(json.dumps({name: constants[name] for name in sys.argv[2:]}))",
            str(installed / "loco_mujoco/smpl/const.py"),
            *manifests,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
    expected_motions = {
        name: [
            line.strip()
            for line in resources[path].decode().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for name, path in manifests.items()
    }
    assert all(expected_motions.values())
    assert json.loads(smoke.stdout) == expected_motions
    assert json.loads((installed / ik_config_path).read_text())["human_root_name"] == "pelvis"

    # Load the installed PHC resources with only the standard library. Stub the
    # package initializers to isolate resource loading from optional simulators.
    phc_smoke = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib, json, pathlib, sys, types
root = pathlib.Path(sys.argv[1])
for name in ("loco_mujoco", "loco_mujoco.smpl"):
    package = types.ModuleType(name)
    package.__path__ = [str(root.joinpath(*name.split(".")))]
    sys.modules[name] = package
manifest = importlib.import_module("loco_mujoco.smpl.motion_manifest")
source = manifest.load_amass_phc_motion_groups()
quality = manifest.load_amass_phc_quality_motion_groups()
print(json.dumps([len(source[manifest.AMASS_PHC_CLEANED_GROUP]),
                  len(quality[manifest.AMASS_PHC_QUALITY_CLEANED_GROUP])]))
""",
            str(installed),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert phc_smoke.returncode == 0, phc_smoke.stdout + phc_smoke.stderr
    assert json.loads(phc_smoke.stdout) == [11776, 11479]
