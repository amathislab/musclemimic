"""Freeze quality-filtered PHC train, validation, and final-test manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from loco_mujoco.smpl.gmr_cache_identity import gmr_motion_relative_path
from loco_mujoco.smpl.motion_manifest import (
    AMASS_PHC_FINAL_TEST_GROUP,
    AMASS_PHC_MANIFEST_FILENAME,
    AMASS_PHC_TRAIN_100_GROUP,
    AMASS_PHC_VALIDATION_GROUP,
    load_amass_phc_motion_groups,
)
from loco_mujoco.smpl.motion_spec import AMASSMotionSpec

TRAIN_FRACTIONS = (0.05, 0.30, 0.50, 1.00)
SOURCE_GROUPS = {
    "train": AMASS_PHC_TRAIN_100_GROUP,
    "validation": AMASS_PHC_VALIDATION_GROUP,
    "final_test": AMASS_PHC_FINAL_TEST_GROUP,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_train_counts(num_train_motions: int) -> dict[str, int]:
    """Return floor-rounded counts for nested quality-filtered train sets."""
    if num_train_motions < 1:
        raise ValueError("num_train_motions must be positive")
    return {str(fraction): math.floor(num_train_motions * fraction) for fraction in TRAIN_FRACTIONS}


def _classify_motion_cache(cache_path: Path) -> str:
    """Classify one motion using its trajectory, sidecar, and rejection marker."""
    analysis_path = cache_path.with_name(f"{cache_path.stem}_analysis.npz")
    rejection_path = cache_path.with_name(f"{cache_path.stem}_quality_rejected.json")
    cache_exists = cache_path.exists()
    analysis_exists = analysis_path.exists()
    rejection_exists = rejection_path.exists()
    if rejection_exists and (cache_exists or analysis_exists):
        return "conflict"
    if rejection_exists:
        return "rejected"
    if cache_exists and analysis_exists:
        return "accepted"
    if cache_exists or analysis_exists:
        return "incomplete"
    return "missing"


def motion_cache_path(
    cache_root: Path,
    model: str,
    cache_namespace: str,
    motion: AMASSMotionSpec,
) -> Path:
    """Return a GMR cache path without importing the MuJoCo pipeline."""
    return cache_root / gmr_motion_relative_path(model, motion, cache_namespace)


def _rejection_kind(rejection_path: Path) -> str:
    try:
        reason = str(json.loads(rejection_path.read_text()).get("reason", ""))
    except (OSError, ValueError, TypeError):
        return "invalid"
    if "floor contact quality" in reason:
        return "floor"
    if "tracking quality" in reason:
        return "tracking"
    if "continuity quality" in reason:
        return "continuity"
    if "first-frame initialization" in reason:
        return "initialization"
    return "other"


def _validate_accepted_analysis(
    analysis_path: Path,
    *,
    clearance_threshold_m: float,
    max_floating_frame_fraction: float,
) -> tuple[float, float]:
    """Validate one accepted motion against the requested source-contact gate."""
    with np.load(analysis_path, allow_pickle=False) as analysis:
        required = {"floating_clearance_threshold_m", "floating_frame_fraction"}
        missing = required.difference(analysis.files)
        if missing:
            raise ValueError(f"missing analysis fields: {sorted(missing)}")
        stored_threshold_m = float(analysis["floating_clearance_threshold_m"])
        stored_floating_fraction = float(analysis["floating_frame_fraction"])
        source_contact_clearances_m = None
        legacy_global_clearances_m = None
        if "source_contact_frame_clearance_m" in analysis.files:
            source_contact_clearances_m = np.asarray(
                analysis["source_contact_frame_clearance_m"],
                dtype=float,
            )
        elif "floor_surface_clearance_m" in analysis.files:
            legacy_global_clearances_m = np.asarray(
                analysis["floor_surface_clearance_m"],
                dtype=float,
            )
    if not np.isfinite(stored_floating_fraction) or not 0.0 <= stored_floating_fraction <= 1.0:
        raise ValueError(f"invalid stored floating fraction {stored_floating_fraction}")
    if stored_threshold_m > clearance_threshold_m + 1e-12:
        raise ValueError(
            f"analysis threshold {stored_threshold_m} m is looser than requested {clearance_threshold_m} m"
        )
    if source_contact_clearances_m is not None:
        if source_contact_clearances_m.ndim != 1 or source_contact_clearances_m.size == 0:
            raise ValueError("source_contact_frame_clearance_m must be a nonempty vector")
        if np.any(np.isinf(source_contact_clearances_m)):
            raise ValueError("source_contact_frame_clearance_m must not contain infinite values")
        source_contact = np.isfinite(source_contact_clearances_m)
        if np.any(source_contact):
            floating_fraction = float(np.mean(source_contact_clearances_m[source_contact] > clearance_threshold_m))
        else:
            if stored_floating_fraction != 0.0:
                raise ValueError("analysis without source-contact frames must store zero floating fraction")
            floating_fraction = 0.0
    elif legacy_global_clearances_m is None:
        # A motion accepted at a stricter clearance threshold remains accepted
        # at a looser threshold. Early sidecars store this scalar evidence but
        # not the per-frame clearance vector.
        floating_fraction = stored_floating_fraction
    else:
        if (
            legacy_global_clearances_m.ndim != 1
            or legacy_global_clearances_m.size == 0
            or not np.all(np.isfinite(legacy_global_clearances_m))
        ):
            raise ValueError("floor_surface_clearance_m must be a nonempty finite vector")
        floating_fraction = float(np.mean(legacy_global_clearances_m > clearance_threshold_m))
    if floating_fraction > max_floating_frame_fraction + 1e-12:
        raise ValueError(f"recomputed floating fraction {floating_fraction} exceeds {max_floating_frame_fraction}")
    return stored_threshold_m, floating_fraction


def _validate_rejection_marker(
    rejection_path: Path,
    motion: AMASSMotionSpec,
    *,
    cache_namespace: str,
    clearance_threshold_m: float,
    max_floating_frame_fraction: float,
) -> str:
    """Validate one rejection marker and return its rejection kind."""
    payload = json.loads(rejection_path.read_text())
    if payload.get("cache_namespace") != cache_namespace:
        raise ValueError(f"unexpected cache namespace {payload.get('cache_namespace')!r}")
    if payload.get("cache_key") != motion.cache_key:
        raise ValueError(f"unexpected cache key {payload.get('cache_key')!r}")
    kind = _rejection_kind(rejection_path)
    if kind == "floor":
        metrics = payload.get("metrics", {})
        marker_threshold_m = float(metrics["floating_clearance_threshold_m"])
        marker_max_fraction = float(metrics["max_floating_frame_fraction"])
        marker_fraction = float(metrics["floating_frame_fraction"])
        if not np.isclose(marker_threshold_m, clearance_threshold_m, rtol=0.0, atol=1e-12):
            raise ValueError(f"floor rejection threshold is {marker_threshold_m} m")
        if not np.isclose(marker_max_fraction, max_floating_frame_fraction, rtol=0.0, atol=1e-12):
            raise ValueError(f"floor rejection fraction limit is {marker_max_fraction}")
        if marker_fraction <= max_floating_frame_fraction:
            raise ValueError(f"floor rejection fraction does not exceed its limit: {marker_fraction}")
    return kind


def _write_manifest(
    output_path: Path,
    accepted_by_partition: dict[str, list[AMASSMotionSpec]],
    *,
    source_manifest_path: Path,
    cache_namespace: str,
    clearance_threshold_m: float,
    max_floating_frame_fraction: float,
) -> None:
    train_counts = _nested_train_counts(len(accepted_by_partition["train"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        file.write("# Quality-filtered PHC-cleaned AMASS motion clips.\n")
        file.write(f"# source_manifest_sha256={_sha256(source_manifest_path)}\n")
        file.write(f"# cache_namespace={cache_namespace}\n")
        file.write(f"# floating_clearance_threshold_m={clearance_threshold_m}\n")
        file.write(f"# max_floating_frame_fraction={max_floating_frame_fraction}\n")
        for fraction, count in train_counts.items():
            file.write(f"# train_{fraction}_count={count}\n")
        file.write(f"# validation_count={len(accepted_by_partition['validation'])}\n")
        file.write(f"# final_test_count={len(accepted_by_partition['final_test'])}\n")
        fieldnames = ("path", "start_time_s", "end_time_s", "partition", "rank")
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for partition in ("train", "validation", "final_test"):
            for rank, motion in enumerate(accepted_by_partition[partition]):
                writer.writerow(
                    {
                        "path": motion.path,
                        "start_time_s": repr(motion.start_time_s),
                        "end_time_s": repr(motion.end_time_s),
                        "partition": partition,
                        "rank": rank,
                    }
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-namespace", required=True)
    parser.add_argument("--model", default="MyoFullBody")
    parser.add_argument("--clearance-threshold-m", type=float, required=True)
    parser.add_argument("--max-floating-frame-fraction", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.clearance_threshold_m < 0.0:
        raise SystemExit("--clearance-threshold-m must be nonnegative")
    if not 0.0 <= args.max_floating_frame_fraction <= 1.0:
        raise SystemExit("--max-floating-frame-fraction must be in [0, 1]")

    source_groups = load_amass_phc_motion_groups()
    accepted_by_partition: dict[str, list[AMASSMotionSpec]] = {}
    report_groups = {}
    nonterminal = []
    for partition, source_group in SOURCE_GROUPS.items():
        accepted = []
        states: Counter[str] = Counter()
        rejection_kinds: Counter[str] = Counter()
        accepted_analysis_thresholds: Counter[str] = Counter()
        maximum_recomputed_floating_fraction = 0.0
        for motion in source_groups[source_group]:
            cache_path = motion_cache_path(
                args.cache_root,
                args.model,
                args.cache_namespace,
                motion,
            )
            state = _classify_motion_cache(cache_path)
            if state == "accepted":
                analysis_path = cache_path.with_name(f"{cache_path.stem}_analysis.npz")
                try:
                    stored_threshold_m, floating_fraction = _validate_accepted_analysis(
                        analysis_path,
                        clearance_threshold_m=args.clearance_threshold_m,
                        max_floating_frame_fraction=args.max_floating_frame_fraction,
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    states["invalid_accepted"] += 1
                    nonterminal.append(
                        {"cache_key": motion.cache_key, "state": "invalid_accepted", "error": str(error)}
                    )
                    continue
                states[state] += 1
                accepted.append(motion)
                accepted_analysis_thresholds[str(stored_threshold_m)] += 1
                maximum_recomputed_floating_fraction = max(
                    maximum_recomputed_floating_fraction,
                    floating_fraction,
                )
            elif state == "rejected":
                rejection_path = cache_path.with_name(f"{cache_path.stem}_quality_rejected.json")
                try:
                    rejection_kind = _validate_rejection_marker(
                        rejection_path,
                        motion,
                        cache_namespace=args.cache_namespace,
                        clearance_threshold_m=args.clearance_threshold_m,
                        max_floating_frame_fraction=args.max_floating_frame_fraction,
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    states["invalid_rejected"] += 1
                    nonterminal.append(
                        {"cache_key": motion.cache_key, "state": "invalid_rejected", "error": str(error)}
                    )
                    continue
                states[state] += 1
                rejection_kinds[rejection_kind] += 1
            else:
                states[state] += 1
                nonterminal.append({"cache_key": motion.cache_key, "state": state})
        accepted_by_partition[partition] = accepted
        report_groups[partition] = {
            "source_group": source_group,
            "expected": len(source_groups[source_group]),
            "accepted": len(accepted),
            "states": dict(sorted(states.items())),
            "rejection_kinds": dict(sorted(rejection_kinds.items())),
            "accepted_analysis_thresholds_m": dict(sorted(accepted_analysis_thresholds.items())),
            "maximum_recomputed_floating_fraction": maximum_recomputed_floating_fraction,
        }

    report = {
        "schema_version": 1,
        "cache_namespace": args.cache_namespace,
        "clearance_threshold_m": args.clearance_threshold_m,
        "max_floating_frame_fraction": args.max_floating_frame_fraction,
        "source_manifest": AMASS_PHC_MANIFEST_FILENAME,
        "source_manifest_sha256": _sha256(
            Path(__file__).parents[1] / "loco_mujoco" / "smpl" / AMASS_PHC_MANIFEST_FILENAME
        ),
        "groups": report_groups,
        "nested_train_counts": _nested_train_counts(len(accepted_by_partition["train"])),
        "nonterminal_count": len(nonterminal),
        "nonterminal": nonterminal,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if nonterminal:
        print(f"Refusing to freeze split with {len(nonterminal)} missing, incomplete, or conflicting motions.")
        return 1

    source_manifest_path = Path(__file__).parents[1] / "loco_mujoco" / "smpl" / AMASS_PHC_MANIFEST_FILENAME
    _write_manifest(
        args.output,
        accepted_by_partition,
        source_manifest_path=source_manifest_path,
        cache_namespace=args.cache_namespace,
        clearance_threshold_m=args.clearance_threshold_m,
        max_floating_frame_fraction=args.max_floating_frame_fraction,
    )
    print(f"Wrote {sum(map(len, accepted_by_partition.values()))} accepted motions to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
