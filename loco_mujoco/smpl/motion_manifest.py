"""Load the reproducible PHC-cleaned AMASS split manifest."""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

from .motion_spec import AMASSMotionSpec

AMASS_PHC_MANIFEST_FILENAME = "amass_phc_cleaned_split_seed42.tsv"
AMASS_PHC_QUALITY_MANIFEST_FILENAME = "amass_phc_gmr_v6_quality_split.tsv"

AMASS_PHC_CLEANED_GROUP = "AMASS_PHC_CLEANED_MOTIONS"
# The heldout group supports checkpoint and pilot-run evaluation. Formal
# experiments use the disjoint validation and final-test groups below.
AMASS_PHC_TEST_GROUP = "AMASS_PHC_TEST_MOTIONS"
AMASS_PHC_VALIDATION_GROUP = "AMASS_PHC_VALIDATION_MOTIONS"
AMASS_PHC_FINAL_TEST_GROUP = "AMASS_PHC_FINAL_TEST_MOTIONS"
AMASS_PHC_TRAIN_30_GROUP = "AMASS_PHC_TRAIN_30_MOTIONS"
AMASS_PHC_TRAIN_50_GROUP = "AMASS_PHC_TRAIN_50_MOTIONS"
AMASS_PHC_TRAIN_100_GROUP = "AMASS_PHC_TRAIN_100_MOTIONS"

AMASS_PHC_QUALITY_CLEANED_GROUP = "AMASS_PHC_QUALITY_CLEANED_MOTIONS"
AMASS_PHC_QUALITY_TRAIN_5_GROUP = "AMASS_PHC_QUALITY_TRAIN_5_MOTIONS"
AMASS_PHC_QUALITY_TRAIN_30_GROUP = "AMASS_PHC_QUALITY_TRAIN_30_MOTIONS"
AMASS_PHC_QUALITY_TRAIN_50_GROUP = "AMASS_PHC_QUALITY_TRAIN_50_MOTIONS"
AMASS_PHC_QUALITY_TRAIN_100_GROUP = "AMASS_PHC_QUALITY_TRAIN_100_MOTIONS"
AMASS_PHC_QUALITY_VALIDATION_GROUP = "AMASS_PHC_QUALITY_VALIDATION_MOTIONS"
AMASS_PHC_QUALITY_FINAL_TEST_GROUP = "AMASS_PHC_QUALITY_FINAL_TEST_MOTIONS"

AMASS_PHC_HELDOUT_SUBSPLIT_SEED = 43

_TRAIN_GROUP_COUNTS = {
    AMASS_PHC_TRAIN_30_GROUP: 3_180,
    AMASS_PHC_TRAIN_50_GROUP: 5_300,
    AMASS_PHC_TRAIN_100_GROUP: 10_601,
}
_EXPECTED_UNIVERSE_COUNT = 11_776
_EXPECTED_TEST_COUNT = 1_175
_VALIDATION_COUNT = 294
_QUALITY_TRAIN_GROUP_COUNTS = {
    AMASS_PHC_QUALITY_TRAIN_5_GROUP: 517,
    AMASS_PHC_QUALITY_TRAIN_30_GROUP: 3_103,
    AMASS_PHC_QUALITY_TRAIN_50_GROUP: 5_173,
    AMASS_PHC_QUALITY_TRAIN_100_GROUP: 10_346,
}
_QUALITY_VALIDATION_COUNT = 283
_QUALITY_FINAL_TEST_COUNT = 850


def _heldout_rank_key(spec: AMASSMotionSpec) -> bytes:
    identity = f"{spec.path}#t={float(spec.start_time_s).hex()}:{float(spec.end_time_s).hex()}"
    payload = f"{AMASS_PHC_HELDOUT_SUBSPLIT_SEED}\0{identity}"
    return hashlib.sha256(payload.encode()).digest()


@lru_cache(maxsize=1)
def load_amass_phc_motion_groups() -> dict[str, tuple[AMASSMotionSpec, ...]]:
    """Load PHC-cleaned motion groups with their source-time bounds."""
    manifest_path = Path(__file__).with_name(AMASS_PHC_MANIFEST_FILENAME)
    with manifest_path.open(newline="") as file:
        rows = csv.DictReader(
            (line for line in file if not line.startswith("#")),
            delimiter="\t",
        )
        parsed_rows = [
            (
                AMASSMotionSpec(
                    path=row["path"],
                    start_time_s=float(row["start_time_s"]),
                    end_time_s=float(row["end_time_s"]),
                ),
                row["partition"],
                None if not row["train_rank"] else int(row["train_rank"]),
            )
            for row in rows
        ]

    universe = tuple(spec for spec, _partition, _rank in parsed_rows)
    test = tuple(spec for spec, partition, _rank in parsed_rows if partition == "test")
    heldout_by_rank = tuple(sorted(test, key=_heldout_rank_key))
    validation = heldout_by_rank[:_VALIDATION_COUNT]
    final_test = heldout_by_rank[_VALIDATION_COUNT:]
    train_by_rank = sorted(
        ((rank, spec) for spec, partition, rank in parsed_rows if partition == "train" and rank is not None),
        key=lambda item: item[0],
    )
    train = tuple(spec for _rank, spec in train_by_rank)

    if len(set(universe)) != len(universe):
        raise ValueError(f"Duplicate motion specifications in {manifest_path}.")
    if len(universe) != _EXPECTED_UNIVERSE_COUNT or len(test) != _EXPECTED_TEST_COUNT:
        raise ValueError(f"Unexpected PHC split counts in {manifest_path}.")
    if len(test) + len(train) != len(universe):
        raise ValueError(f"Invalid partition values in {manifest_path}.")
    if [rank for rank, _spec in train_by_rank] != list(range(len(train))):
        raise ValueError(f"Train ranks must be contiguous in {manifest_path}.")

    groups = {
        AMASS_PHC_CLEANED_GROUP: universe,
        AMASS_PHC_TEST_GROUP: test,
        AMASS_PHC_VALIDATION_GROUP: validation,
        AMASS_PHC_FINAL_TEST_GROUP: final_test,
        AMASS_PHC_TRAIN_100_GROUP: train,
    }
    for group_name, count in _TRAIN_GROUP_COUNTS.items():
        groups[group_name] = train[:count]
    return groups


@lru_cache(maxsize=1)
def load_amass_phc_quality_motion_groups() -> dict[str, tuple[AMASSMotionSpec, ...]]:
    """Load the frozen gmr_v6 quality-filtered PHC experiment groups."""
    manifest_path = Path(__file__).with_name(AMASS_PHC_QUALITY_MANIFEST_FILENAME)
    with manifest_path.open(newline="") as file:
        rows = csv.DictReader(
            (line for line in file if not line.startswith("#")),
            delimiter="\t",
        )
        parsed_rows = [
            (
                AMASSMotionSpec(
                    path=row["path"],
                    start_time_s=float(row["start_time_s"]),
                    end_time_s=float(row["end_time_s"]),
                ),
                row["partition"],
                int(row["rank"]),
            )
            for row in rows
        ]

    partitions = {}
    for partition in ("train", "validation", "final_test"):
        ranked = sorted(
            ((rank, spec) for spec, row_partition, rank in parsed_rows if row_partition == partition),
            key=lambda item: item[0],
        )
        ranks = [rank for rank, _spec in ranked]
        if any(rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError(f"{partition} ranks must be unique and nonnegative in {manifest_path}.")
        partitions[partition] = tuple(spec for _rank, spec in ranked)

    train = partitions["train"]
    validation = partitions["validation"]
    final_test = partitions["final_test"]
    universe = train + validation + final_test
    if len(universe) != len(set(universe)):
        raise ValueError(f"Duplicate motion specifications in {manifest_path}.")
    if len(parsed_rows) != len(universe):
        raise ValueError(f"Invalid partition values in {manifest_path}.")
    if (
        len(train) != _QUALITY_TRAIN_GROUP_COUNTS[AMASS_PHC_QUALITY_TRAIN_100_GROUP]
        or len(validation) != _QUALITY_VALIDATION_COUNT
        or len(final_test) != _QUALITY_FINAL_TEST_COUNT
    ):
        raise ValueError(f"Unexpected quality-filtered PHC split counts in {manifest_path}.")

    source_groups = load_amass_phc_motion_groups()
    source_partition_names = {
        "train": AMASS_PHC_TRAIN_100_GROUP,
        "validation": AMASS_PHC_VALIDATION_GROUP,
        "final_test": AMASS_PHC_FINAL_TEST_GROUP,
    }
    for partition, accepted in partitions.items():
        accepted_set = set(accepted)
        expected_order = tuple(
            spec for spec in source_groups[source_partition_names[partition]] if spec in accepted_set
        )
        if accepted != expected_order:
            raise ValueError(f"{partition} must preserve the source split and rank order in {manifest_path}.")

    groups = {
        AMASS_PHC_QUALITY_CLEANED_GROUP: universe,
        AMASS_PHC_QUALITY_VALIDATION_GROUP: validation,
        AMASS_PHC_QUALITY_FINAL_TEST_GROUP: final_test,
        AMASS_PHC_QUALITY_TRAIN_100_GROUP: train,
    }
    for group_name, count in _QUALITY_TRAIN_GROUP_COUNTS.items():
        groups[group_name] = train[:count]
    return groups


def load_amass_phc_quality_motion_ranks() -> dict[AMASSMotionSpec, int]:
    """Return original manifest ranks for the selected quality-filtered clips."""
    groups = load_amass_phc_quality_motion_groups()
    manifest_path = Path(__file__).with_name(AMASS_PHC_QUALITY_MANIFEST_FILENAME)
    with manifest_path.open(newline="") as file:
        rows = csv.DictReader((line for line in file if not line.startswith("#")), delimiter="\t")
        ranks = {
            AMASSMotionSpec(
                path=row["path"],
                start_time_s=float(row["start_time_s"]),
                end_time_s=float(row["end_time_s"]),
            ): int(row["rank"])
            for row in rows
        }
    if len(ranks) != len(groups[AMASS_PHC_QUALITY_CLEANED_GROUP]):
        raise ValueError(f"Duplicate motion specifications in {manifest_path}.")
    return ranks
