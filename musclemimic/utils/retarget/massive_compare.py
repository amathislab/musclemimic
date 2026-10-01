"""Dataset-wide retargeting comparison across an arbitrary set of methods.

Reports the core benchmark metric suite -- joint-limit violations, ground penetration,
floating, tendon jump, RMSE and speed -- for every motion in a dataset group, per method.

Originally hardcoded to compare exactly two methods (`gmr` against `mimic`, the mocap-body
pipeline). It is now driven by `--methods`, so adding OmniRetarget is a flag rather than an
edit; method names resolve through `msk_metrics.METHOD_DIRS`.

The columns here are only the core benchmark's. The standalone TERRA repository's
`scripts/terra/evaluate_subset.py` reports the
same numbers plus the source-referenced ones (coupler residual, foot slip, opening
transient, consistent tracking error), which need the SMPL body model and OmniRetarget on
the path and so cannot be a dependency of this module.
"""

import argparse
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from musclemimic.utils.retarget.msk_metrics import (
    METHOD_DIRS,
    _get_base_and_gmr_dirs,
    compute_msk_metrics,
    method_cache_path,
)

# ================== THRESHOLDS ==================
TH_JOINT = 0.5     # %
TH_PEN_PCT = 1      # % frames
TH_PEN_MAX = 0.005    # meters (5 mm)
TH_FLOAT = 0.025   # meters (2.5 cm), the radius of the foot collision capsule is already 2cm
TH_TEND  = 0.05   # %

# Per-method metric columns, in report order. The `{m}_` prefix is filled per method.
METRIC_COLS = [
    "joint_pct",
    "pen_pct",
    "pen_max_depth_m",
    "float_height",
    "tendon_max_jump_%",
    "pos_rmse",
    "sec_per_frame",
]

# Columns whose mean is taken over non-zero entries only: a zero means "no such event in
# this motion", and averaging those in reports the event rate a second time instead of the
# severity when it happens.
NONZERO_MEAN = ["pen_max_depth_m", "tendon_max_jump_%"]


def _record_fail(summary, method: str, category: str, motion_rel: str):
    summary.setdefault(method, {}).setdefault(category, set()).add(motion_rel)


def _update_fail_summary_from_result(summary, method: str, motion_rel: str, r: Dict):
    # Only count a motion for a category if it crosses the threshold
    checks = [
        (f"{method}_joint_pct", TH_JOINT, "joint violation"),
        (f"{method}_pen_max_depth_m", TH_PEN_MAX, "penetration max depth"),
        (f"{method}_float_height", TH_FLOAT, "floating"),
        (f"{method}_tendon_max_jump_%", TH_TEND, "tendon jump"),
    ]
    for key, threshold, category in checks:
        if r.get(key) is not None and r[key] > threshold:
            _record_fail(summary, method.upper(), category, motion_rel)


def new_fail_summary():
    return {}


def merge_fail_summaries(dst, src):
    for method, cats in src.items():
        for cat, motions in cats.items():
            dst[method][cat].update(motions)


def print_fail_summary(summary):
    print("\n" + "=" * 60)
    print("Failure summary (count = #motions)")
    print("=" * 60)
    for method in sorted(summary.keys()):
        print(f"\n{method}:")
        for cat in sorted(summary[method].keys()):
            print(f"  {cat}: {len(summary[method][cat])}")
    print("=" * 60 + "\n")


def collect_dataset_files(dataset_name: str) -> List[Path]:
    """
    Resolve dataset motion list (relative paths) to absolute KIT npz paths.
    """
    motion_list = get_motion_dataset(dataset_name)
    base, _ = _get_base_and_gmr_dirs()

    files: List[Path] = []
    for rel in motion_list:
        p = base / rel
        if p.suffix != ".npz":
            p = p.with_suffix(".npz")
        files.append(p)

    return files


def violates(r: Dict, method: str) -> bool:
    """Whether this motion crosses any quality threshold under `method`."""
    return (
        (r.get(f"{method}_joint_pct") is not None and r[f"{method}_joint_pct"] > TH_JOINT)
        or (r.get(f"{method}_pen_max_depth_m") is not None and r[f"{method}_pen_max_depth_m"] > TH_PEN_MAX)
        or (r.get(f"{method}_float_height") is not None and r[f"{method}_float_height"] > TH_FLOAT)
        or (r.get(f"{method}_tendon_max_jump_%") is not None and r[f"{method}_tendon_max_jump_%"] > TH_TEND)
    )


def worker_process(args: Tuple[int, List[Path], List[str]]):
    worker_id, batch, methods = args
    print(f"Worker {worker_id}: {len(batch)} motions")
    local_fail_summary = {}

    base, _ = _get_base_and_gmr_dirs()

    success = 0
    missing = 0
    all_records: List[Dict] = []
    bad: Dict[str, List[Dict]] = {m: [] for m in methods}
    errors: List[str] = []
    missing_files: List[str] = []

    for i, f in enumerate(batch):
        rel = f.relative_to(base)

        absent = [m for m in methods if not method_cache_path(m, rel).exists()]
        if absent:
            print(f"Worker {worker_id}: Missing {absent} cache for {rel}, skipping")
            missing += 1
            missing_files.append(f"{rel} (missing: {','.join(absent)})")
            continue

        try:
            print(f"Worker {worker_id}: [{i + 1}/{len(batch)}] {rel}")

            metrics = compute_msk_metrics(rel, methods=methods)

            result = {"file": str(rel)}
            for m in methods:
                for c in METRIC_COLS:
                    result[f"{m}_{c}"] = metrics[m].get(c)

            success += 1
            all_records.append(result)

            motion_rel = str(rel)
            for m in methods:
                _update_fail_summary_from_result(local_fail_summary, m, motion_rel, result)
                if violates(result, m):
                    bad[m].append(result)

        except Exception as e:
            errors.append(f"{rel}: {e!r}")

    return success, missing, all_records, bad, errors, missing_files, local_fail_summary


def distribute(lst: List[Path], num_workers: int) -> List[List[Path]]:
    batches = [[] for _ in range(num_workers)]
    for i, item in enumerate(lst):
        batches[i % num_workers].append(item)
    return batches


def get_motion_dataset(dataset_name: str) -> List[str]:
    """Load the specified motion dataset."""
    try:
        from loco_mujoco import smpl

        if hasattr(smpl.const, dataset_name):
            return list(getattr(smpl.const, dataset_name))
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in loco_mujoco.smpl.const")
    except ImportError as e:
        raise ImportError(f"Failed to import dataset {dataset_name}: {e}") from e


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--methods",
        default="gmr,mimic",
        help=f"comma-separated retargeting methods to compare; known: {sorted(METHOD_DIRS)}",
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "AMASS_BIMANUAL_MARGINAL_MOTIONS",
            "KIT_KINESIS_TRAINING_MOTIONS",
            "AMASS_BIMANUAL_TRAIN_MOTIONS",
            "AMASS_BIMANUAL_TEST_MOTIONS",
            "KIT_KINESIS_TRANSITION_TRAINING_MOTIONS",
            "AMASS_TRANSITION_MOTIONS",
            "ACCAD_TRAINING_MOTIONS"
        ],
        default="KIT_KINESIS_TRAINING_MOTIONS",
        help="Dataset constant name from loco_mujoco.smpl.const",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - set(METHOD_DIRS)
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}; known: {sorted(METHOD_DIRS)}")

    motion_files = collect_dataset_files(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Methods: {methods}")
    print(f"Found {len(motion_files)} motions")

    if not motion_files:
        print("No files found.")
        return 0

    num_workers = min(args.workers or mp.cpu_count(), 100, len(motion_files))
    print(f"Using {num_workers} workers")

    batches = distribute(motion_files, num_workers)
    worker_args = [(i, b, methods) for i, b in enumerate(batches)]

    start = time.time()

    with mp.get_context("spawn").Pool(num_workers) as pool:
        results = pool.map(worker_process, worker_args)

    elapsed = time.time() - start

    total_success = sum(r[0] for r in results)
    total_missing = sum(r[1] for r in results)
    all_records   = [item for r in results for item in r[2]]
    failed        = {m: [item for r in results for item in r[3][m]] for m in methods}
    all_errors    = [msg  for r in results for msg in r[4]]
    missing_files = [p for r in results for p in r[5]]

    print("\n=== RESULTS (PRE-SAVE) ===")
    print(f"Total discovered: {len(motion_files)}")
    print(f"Processed OK:     {total_success}")
    print(f"Missing caches:   {total_missing}")
    print(f"Errors:           {len(all_errors)}")
    print(f"Time: {elapsed:.2f}s")

    if missing_files:
        print("\n=== MISSING (NOT COMPARED) ===")
        # Print one path per line.
        for p in missing_files:
            print(p)

    # Abort output generation if any worker failed.
    if all_errors:
        # Print the first error only.
        print("\nERROR DETECTED — not writing CSV outputs.")
        print(all_errors[0])
        return 1

    # ================== WRITE OUTPUTS ==================
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = args.dataset.lower()

    all_cols = ["file"] + [f"{m}_{c}" for m in methods for c in METRIC_COLS]
    df_all = pd.DataFrame(all_records, columns=all_cols)

    all_path = args.out_dir / f"all_motions_{dataset_tag}.csv"
    df_all.to_csv(all_path, index=False, float_format="%.5f")
    print(f"Saved {all_path.name}")

    for m in methods:
        cols = ["file"] + [f"{m}_{c}" for c in METRIC_COLS]
        path = args.out_dir / f"failed_motions_{m}_{dataset_tag}.csv"
        pd.DataFrame(failed[m], columns=cols).to_csv(path, index=False, float_format="%.5f")
        print(f"Saved {path.name}")

    missing_counts = df_all.isna().sum().sort_values(ascending=False)
    print("\n=== MISSING COUNTS PER COLUMN ===")
    print(missing_counts[missing_counts > 0])

    mean_vals = df_all.mean(numeric_only=True)
    for m in methods:
        for c in NONZERO_MEAN:
            col = f"{m}_{c}"
            if col in df_all.columns:
                s = pd.to_numeric(df_all[col], errors="coerce")
                nz = s.abs() > 0.0
                mean_vals[col] = s[nz].mean() if nz.any() else 0.0

    print("\n=== SUMMARY (MEAN OVER ALL MOTIONS) ===")
    for m in methods:
        tendon_motion_pct = 100.0 * (df_all[f"{m}_tendon_max_jump_%"] > TH_TEND).mean()
        parts = [f"{c}={mean_vals.get(f'{m}_{c}', float('nan')):.5f}" for c in METRIC_COLS]
        print(f"{m.upper():6s} " + "  ".join(parts) + f"  tendon_motion_pct={tendon_motion_pct:.2f}%")

    merged_fail_summary = defaultdict(lambda: defaultdict(set))
    for fs in (r[6] for r in results):
        merge_fail_summaries(merged_fail_summary, fs)

    print_fail_summary(merged_fail_summary)

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
