"""Generate reproducible MuscleMimic splits from PHC-cleaned AMASS YAML files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.10
TRAIN_RATIOS = (0.30, 0.50, 1.00)
EXCLUDED_MOTION_PATHS = frozenset(
    {
        "KIT/987/Experiment3a_03_poses",
        "BMLmovi/Subject_6_F_MoSh/Subject_6_F_17_poses",
    }
)
EXCLUDED_SOURCE_PREFIXES = ("TCD_handMocap/",)

_LOCAL_TO_PHC_PREFIXES = {
    "CMU/CMU/": "CMU/",
    "DFaust/DFaust_67/": "DFaust_67/",
    "EKUT/EKUT/": "EKUT/",
    "EyesJapanDataset/Eyes_Japan_Dataset/": "Eyes_Japan_Dataset/",
    "HDM05/MPI_HDM05/": "MPI_HDM05/",
    "HumanEva/HumanEva/": "HumanEva/",
    "MoSh/MPI_mosh/": "MPI_mosh/",
    "PosePrior/MPI_Limits/": "MPI_Limits/",
    "SFU/SFU/": "SFU/",
    "SSM/SSM_synced/": "SSM_synced/",
}


@dataclass(frozen=True)
class MotionRow:
    """One source motion and its PHC-cleaned time interval."""

    path: str
    start_time_s: float
    end_time_s: float
    source_split: str

    @property
    def identity(self) -> str:
        """Return a stable identity for deterministic random ranking."""
        return f"{self.path}#t={float(self.start_time_s).hex()}:{float(self.end_time_s).hex()}"


def _canonical_path(path: str) -> str:
    path = path.removesuffix(".npz").removesuffix(".motion").lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", path)).strip("_")


def _phc_relative_path(local_path: str) -> str:
    for local_prefix, phc_prefix in _LOCAL_TO_PHC_PREFIXES.items():
        if local_path.startswith(local_prefix):
            return phc_prefix + local_path[len(local_prefix) :]
    return local_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_amass(amass_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in sorted(amass_root.rglob("*.npz")):
        if path.name.endswith("shape.npz"):
            continue
        relative_path = path.relative_to(amass_root).as_posix().removesuffix(".npz")
        key = _canonical_path(_phc_relative_path(relative_path))
        if key in index:
            raise ValueError(f"AMASS paths {index[key]!r} and {relative_path!r} have the same canonical identity.")
        index[key] = relative_path
    return index


def _load_phc_yaml(path: Path, source_split: str, amass_index: dict[str, str]) -> list[MotionRow]:
    with path.open() as file:
        payload: dict[str, Any] = yaml.safe_load(file)

    rows = []
    for motion in payload["motions"]:
        sub_motions = motion.get("sub_motions", [])
        if len(sub_motions) != 1:
            raise ValueError(f"Expected one PHC interval for {motion['file']!r}, got {len(sub_motions)}.")
        timing = sub_motions[0]["timings"]
        key = _canonical_path(motion["file"])
        if key not in amass_index:
            raise ValueError(f"PHC motion {motion['file']!r} does not match a file under {path}.")
        rows.append(
            MotionRow(
                path=amass_index[key],
                start_time_s=float(timing["start"]),
                end_time_s=float(timing["end"]),
                source_split=source_split,
            )
        )
    return rows


def _random_rank(row: MotionRow, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{row.identity}".encode()).digest()


def _assign_splits(
    rows: list[MotionRow],
    seed: int,
    test_ratio: float,
) -> dict[str, tuple[str, int | None]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}.")

    ranked = sorted(rows, key=lambda row: (_random_rank(row, seed), row.identity))
    num_test = math.ceil(len(ranked) * test_ratio)
    assignments = {row.path: ("test", None) for row in ranked[:num_test]}
    for train_rank, row in enumerate(ranked[num_test:]):
        assignments[row.path] = ("train", train_rank)
    return assignments


def _exclude_motions(
    rows: list[MotionRow],
    assignments: dict[str, tuple[str, int | None]],
) -> tuple[list[MotionRow], dict[str, tuple[str, int | None]]]:
    """Remove excluded clips while retaining every other split and rank order."""
    retained = [
        row
        for row in rows
        if row.path not in EXCLUDED_MOTION_PATHS and not row.path.startswith(EXCLUDED_SOURCE_PREFIXES)
    ]
    train_paths = sorted(
        (row.path for row in retained if assignments[row.path][0] == "train"),
        key=lambda path: assignments[path][1],
    )
    train_ranks = {path: rank for rank, path in enumerate(train_paths)}
    retained_assignments = {
        row.path: ("train", train_ranks[row.path]) if assignments[row.path][0] == "train" else ("test", None)
        for row in retained
    }
    return retained, retained_assignments


def _write_manifest(
    output_path: Path,
    rows: list[MotionRow],
    assignments: dict[str, tuple[str, int | None]],
    source_paths: dict[str, Path],
    seed: int,
    test_ratio: float,
) -> None:
    num_test = sum(partition == "test" for partition, _rank in assignments.values())
    num_train = len(rows) - num_test
    train_counts = {ratio: int(num_train * ratio) for ratio in TRAIN_RATIOS}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        file.write("# PHC-cleaned AMASS motion clips mapped to the MuscleMimic AMASS layout.\n")
        file.write(f"# split_seed={seed}\n")
        file.write(f"# test_ratio={test_ratio}\n")
        file.write(f"# universe_count={len(rows)}\n")
        file.write(f"# test_count={num_test}\n")
        file.write(f"# train_30_count={train_counts[0.30]}\n")
        file.write(f"# train_50_count={train_counts[0.50]}\n")
        file.write(f"# train_100_count={train_counts[1.00]}\n")
        for source_split, source_path in source_paths.items():
            file.write(f"# source_{source_split}_sha256={_sha256(source_path)}\n")

        fieldnames = (
            "path",
            "start_time_s",
            "end_time_s",
            "partition",
            "train_rank",
            "source_split",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.path):
            partition, train_rank = assignments[row.path]
            writer.writerow(
                {
                    "path": row.path,
                    "start_time_s": repr(row.start_time_s),
                    "end_time_s": repr(row.end_time_s),
                    "partition": partition,
                    "train_rank": "" if train_rank is None else train_rank,
                    "source_split": row.source_split,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-root", type=Path, required=True)
    parser.add_argument("--train-yaml", type=Path, required=True)
    parser.add_argument("--validation-yaml", type=Path, required=True)
    parser.add_argument("--test-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = {
        "train": args.train_yaml,
        "validation": args.validation_yaml,
        "test": args.test_yaml,
    }
    amass_index = _index_amass(args.amass_root)
    rows = [
        row
        for source_split, source_path in source_paths.items()
        for row in _load_phc_yaml(source_path, source_split, amass_index)
    ]
    if len({row.path for row in rows}) != len(rows):
        raise ValueError("PHC YAML files contain duplicate AMASS paths.")

    assignments = _assign_splits(rows, seed=args.seed, test_ratio=args.test_ratio)
    rows, assignments = _exclude_motions(rows, assignments)
    _write_manifest(
        output_path=args.output,
        rows=rows,
        assignments=assignments,
        source_paths=source_paths,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )
    print(f"Wrote {len(rows)} PHC-cleaned AMASS clips to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
