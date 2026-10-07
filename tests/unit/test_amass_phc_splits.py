import csv
import hashlib
import sys
from pathlib import Path

import yaml

from loco_mujoco.smpl.motion_manifest import (
    AMASS_PHC_CLEANED_GROUP,
    AMASS_PHC_FINAL_TEST_GROUP,
    AMASS_PHC_HELDOUT_SUBSPLIT_SEED,
    AMASS_PHC_MANIFEST_FILENAME,
    AMASS_PHC_QUALITY_CLEANED_GROUP,
    AMASS_PHC_QUALITY_FINAL_TEST_GROUP,
    AMASS_PHC_QUALITY_MANIFEST_FILENAME,
    AMASS_PHC_QUALITY_TRAIN_5_GROUP,
    AMASS_PHC_QUALITY_TRAIN_30_GROUP,
    AMASS_PHC_QUALITY_TRAIN_50_GROUP,
    AMASS_PHC_QUALITY_TRAIN_100_GROUP,
    AMASS_PHC_QUALITY_VALIDATION_GROUP,
    AMASS_PHC_TEST_GROUP,
    AMASS_PHC_TRAIN_30_GROUP,
    AMASS_PHC_TRAIN_50_GROUP,
    AMASS_PHC_TRAIN_100_GROUP,
    AMASS_PHC_VALIDATION_GROUP,
    load_amass_phc_motion_groups,
    load_amass_phc_quality_motion_groups,
)
from loco_mujoco.task_factories.dataset_confs import (
    get_amass_dataset_groups,
    get_amass_motion_spec_groups,
)
from scripts import generate_amass_phc_splits

EXPECTED_MANIFEST_SHA256 = "0821fb81eed72be8a0df67f74accb053636f5428c02a51698a00267f98d191a8"
EXPECTED_QUALITY_MANIFEST_SHA256 = "d75528d5efcd4988791b09f69d57aba79e672a061ce305bf87e6d7f39d2b9d23"


def test_split_generator_excludes_requested_motions(tmp_path: Path, monkeypatch) -> None:
    amass_root = tmp_path / "amass"
    paths = [
        "KIT/987/Experiment3a_03_poses",
        "BMLmovi/Subject_6_F_MoSh/Subject_6_F_17_poses",
        "TCD_handMocap/ExperimentDatabase/OK_B_poses",
        "KIT/1/retained_a_poses",
        "KIT/2/retained_b_poses",
    ]
    for relative_path in paths:
        path = amass_root / f"{relative_path}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        yaml.safe_dump(
            {
                "motions": [
                    {"file": f"{path}.npz", "sub_motions": [{"timings": {"start": 0.0, "end": 1.0}}]} for path in paths
                ]
            }
        )
    )
    validation_yaml = tmp_path / "validation.yaml"
    test_yaml = tmp_path / "test.yaml"
    for source_yaml in (validation_yaml, test_yaml):
        source_yaml.write_text(yaml.safe_dump({"motions": []}))
    output = tmp_path / "split.tsv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_amass_phc_splits",
            "--amass-root",
            str(amass_root),
            "--train-yaml",
            str(train_yaml),
            "--validation-yaml",
            str(validation_yaml),
            "--test-yaml",
            str(test_yaml),
            "--output",
            str(output),
        ],
    )

    assert generate_amass_phc_splits.main() == 0
    with output.open() as file:
        rows = list(csv.DictReader((line for line in file if not line.startswith("#")), delimiter="\t"))
    assert {row["path"] for row in rows} == set(paths[-2:])
    train_ranks = sorted(int(row["train_rank"]) for row in rows if row["partition"] == "train")
    assert train_ranks == list(range(len(train_ranks)))


def test_phc_cleaned_splits_are_complete_disjoint_and_nested():
    groups = load_amass_phc_motion_groups()
    universe = groups[AMASS_PHC_CLEANED_GROUP]
    heldout = groups[AMASS_PHC_TEST_GROUP]
    validation = groups[AMASS_PHC_VALIDATION_GROUP]
    final_test = groups[AMASS_PHC_FINAL_TEST_GROUP]
    train_30 = groups[AMASS_PHC_TRAIN_30_GROUP]
    train_50 = groups[AMASS_PHC_TRAIN_50_GROUP]
    train_100 = groups[AMASS_PHC_TRAIN_100_GROUP]

    assert len(universe) == 11_776
    assert len(heldout) == 1_175
    assert len(validation) == 294
    assert len(final_test) == 881
    assert len(train_30) == 3_180
    assert len(train_50) == 5_300
    assert len(train_100) == 10_601

    universe_set = set(universe)
    heldout_set = set(heldout)
    validation_set = set(validation)
    final_test_set = set(final_test)
    train_100_set = set(train_100)
    assert len(universe_set) == len(universe)
    assert heldout_set.isdisjoint(train_100_set)
    assert heldout_set | train_100_set == universe_set
    assert validation_set.isdisjoint(final_test_set)
    assert validation_set | final_test_set == heldout_set
    assert len({spec.path.split("/", maxsplit=1)[0] for spec in final_test}) == 18
    assert train_30 == train_50[: len(train_30)]
    assert train_50 == train_100[: len(train_50)]


def test_phc_heldout_subsplit_is_stable():
    groups = load_amass_phc_motion_groups()
    validation = groups[AMASS_PHC_VALIDATION_GROUP]
    final_test = groups[AMASS_PHC_FINAL_TEST_GROUP]

    assert AMASS_PHC_HELDOUT_SUBSPLIT_SEED == 43
    validation_digest = hashlib.sha256("\n".join(spec.cache_key for spec in validation).encode()).hexdigest()
    final_test_digest = hashlib.sha256("\n".join(spec.cache_key for spec in final_test).encode()).hexdigest()
    assert validation_digest == "c430a6b42120ec0661b264084d8ece06fc52888e3becf4dc4f0cd4fa0073e9de"
    assert final_test_digest == "650ceff6d1ebca2c949befedeea4f34b15a11fdd18a2fe16c67f6a34b99fada8"


def test_phc_manifest_and_legacy_group_views_are_stable():
    manifest_path = Path(__file__).parents[2] / "loco_mujoco" / "smpl" / AMASS_PHC_MANIFEST_FILENAME
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    spec_groups = get_amass_motion_spec_groups()
    path_groups = get_amass_dataset_groups()

    assert manifest_digest == EXPECTED_MANIFEST_SHA256
    for group_name in (
        AMASS_PHC_CLEANED_GROUP,
        AMASS_PHC_TEST_GROUP,
        AMASS_PHC_VALIDATION_GROUP,
        AMASS_PHC_FINAL_TEST_GROUP,
        AMASS_PHC_TRAIN_30_GROUP,
        AMASS_PHC_TRAIN_50_GROUP,
        AMASS_PHC_TRAIN_100_GROUP,
    ):
        specs = spec_groups[group_name]
        assert path_groups[group_name] == [spec.path for spec in specs]
        assert all(spec.end_time_s is not None for spec in specs)
        assert all(spec.end_time_s > spec.start_time_s for spec in specs)


def test_quality_filtered_phc_splits_preserve_original_partitions_and_nesting():
    source_groups = load_amass_phc_motion_groups()
    groups = load_amass_phc_quality_motion_groups()
    universe = groups[AMASS_PHC_QUALITY_CLEANED_GROUP]
    train_5 = groups[AMASS_PHC_QUALITY_TRAIN_5_GROUP]
    train_30 = groups[AMASS_PHC_QUALITY_TRAIN_30_GROUP]
    train_50 = groups[AMASS_PHC_QUALITY_TRAIN_50_GROUP]
    train_100 = groups[AMASS_PHC_QUALITY_TRAIN_100_GROUP]
    validation = groups[AMASS_PHC_QUALITY_VALIDATION_GROUP]
    final_test = groups[AMASS_PHC_QUALITY_FINAL_TEST_GROUP]

    assert len(universe) == 11_479
    assert len(train_5) == 517
    assert len(train_30) == 3_103
    assert len(train_50) == 5_173
    assert len(train_100) == 10_346
    assert len(validation) == 283
    assert len(final_test) == 850
    assert train_5 == train_30[: len(train_5)]
    assert train_30 == train_50[: len(train_30)]
    assert train_50 == train_100[: len(train_50)]
    assert set(train_100).issubset(source_groups[AMASS_PHC_TRAIN_100_GROUP])
    assert set(validation).issubset(source_groups[AMASS_PHC_VALIDATION_GROUP])
    assert set(final_test).issubset(source_groups[AMASS_PHC_FINAL_TEST_GROUP])
    assert set(train_100).isdisjoint(validation)
    assert set(train_100).isdisjoint(final_test)
    assert set(validation).isdisjoint(final_test)


def test_removed_amass_motions_are_absent_from_dataset_groups():
    excluded_paths = {
        "KIT/987/Experiment3a_03_poses",
        "BMLmovi/Subject_6_F_MoSh/Subject_6_F_17_poses",
    }
    for motions in get_amass_dataset_groups().values():
        assert excluded_paths.isdisjoint(motions)
        assert all(not path.startswith("TCD_handMocap/") for path in motions)


def test_quality_filtered_manifest_and_group_views_are_stable():
    manifest_path = Path(__file__).parents[2] / "loco_mujoco" / "smpl" / AMASS_PHC_QUALITY_MANIFEST_FILENAME
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    spec_groups = get_amass_motion_spec_groups()
    path_groups = get_amass_dataset_groups()

    assert manifest_digest == EXPECTED_QUALITY_MANIFEST_SHA256
    for group_name, specs in load_amass_phc_quality_motion_groups().items():
        assert spec_groups[group_name] == list(specs)
        assert path_groups[group_name] == [spec.path for spec in specs]
