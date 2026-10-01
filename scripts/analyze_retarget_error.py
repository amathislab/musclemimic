"""
Analyze retargeting position error from saved analysis files.

Analysis files are created alongside trajectory files during retargeting:
- Trajectory: MyoFullBody_gmr/KIT/3/walking_slow01_poses.npz
- Analysis:   MyoFullBody_gmr/KIT/3/walking_slow01_poses_analysis.npz

Usage:
    python scripts/analyze_retarget_error.py --motion KIT/3/walking_slow01_poses
    python scripts/analyze_retarget_error.py --dataset KIT_KINESIS_TRAINING_MOTIONS --output results.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def get_analysis_path(motion_name: str, method: str = "gmr") -> Path:
    from loco_mujoco.smpl.retargeting import (
        get_converted_amass_dataset_path,
    )

    base_dir = Path(get_converted_amass_dataset_path())

    if method == "gmr":
        return base_dir / "MyoFullBody" / "gmr" / f"{motion_name}_analysis.npz"
    else:
        return base_dir / "MyoFullBody" / f"{motion_name}_analysis.npz"


def load_analysis(motion_name: str, method: str = "gmr") -> Optional[Dict]:
    """Load analysis data from saved npz file."""
    path = get_analysis_path(motion_name, method)
    if not path.exists():
        return None

    try:
        data = np.load(path, allow_pickle=True)
        return {
            "pos_error": data["pos_error"],           # (T, num_sites) array
            "retarget_fps": float(data["retarget_fps"]),
            "site_names": list(data["site_names"]) if "site_names" in data else [],
        }
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None


def compute_error_stats(analysis: Dict) -> Dict:
    """Compute summary statistics from pos_error array."""
    pos_error = analysis["pos_error"]
    return {
        "mean_error_m": float(np.mean(pos_error)),
        "RMSE_error_m": float(np.sqrt(np.mean(pos_error**2))),
        "max_error_m": float(np.max(pos_error)),
        "std_error_m": float(np.std(pos_error)),
        "num_frames": pos_error.shape[0],
        "num_sites": pos_error.shape[1] if len(pos_error.shape) > 1 else 1,
        "retarget_fps": analysis["retarget_fps"],
    }


def analyze_single_motion(motion_name: str, method: str = "gmr") -> Optional[Dict]:
    """Analyze a single motion's retarget quality from saved analysis file."""
    print(f"Analyzing: {motion_name}")

    analysis = load_analysis(motion_name, method)
    if analysis is None:
        analysis_path = get_analysis_path(motion_name, method)
        print(f"  Analysis file not found: {analysis_path}")
        print(f"  Run retargeting first to generate analysis files.")
        return None

    stats = compute_error_stats(analysis)
    stats["motion"] = motion_name
    stats["method"] = method

    print(f"  Mean error: {stats['mean_error_m']:.4f}m")
    print(f"  RMSE error: {stats['RMSE_error_m']:.4f}m")
    print(f"  Max error:  {stats['max_error_m']:.4f}m")
    print(f"  Std error:  {stats['std_error_m']:.4f}m")
    print(f"  Frames:     {stats['num_frames']}")
    print(f"  Sites:      {stats['num_sites']}")
    print(f"  FPS:        {stats['retarget_fps']:.2f}")

    return stats


def get_motion_dataset(dataset_name: str) -> List[str]:
    """Load the specified motion dataset."""
    try:
        from loco_mujoco import smpl

        if hasattr(smpl.const, dataset_name):
            return list(getattr(smpl.const, dataset_name))
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in loco_mujoco.smpl.const")
    except ImportError as e:
        raise ImportError(f"Failed to import dataset {dataset_name}: {e}")


def analyze_dataset(dataset_name: str, method: str = "gmr") -> List[Dict]:
    """Analyze all motions in a dataset."""
    motions = get_motion_dataset(dataset_name)
    print(f"Dataset: {dataset_name} ({len(motions)} motions)")

    results = []
    missing = 0
    for motion in motions:
        analysis = load_analysis(motion, method)
        if analysis is not None:
            stats = compute_error_stats(analysis)
            stats["motion"] = motion
            stats["method"] = method
            results.append(stats)
        else:
            missing += 1

    if missing > 0:
        print(f"  Missing analysis files: {missing}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze retargeting position error from saved analysis files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--motion",
        help="Single motion name (e.g., KIT/3/walking_slow01_poses)"
    )
    group.add_argument(
        "--dataset",
        choices=[
            "KIT_KINESIS_TRAINING_MOTIONS",
            "AMASS_BIMANUAL_MARGINAL_MOTIONS",
            "AMASS_TRANSITION_MOTIONS",
        ],
        help="Dataset name from loco_mujoco.smpl.const"
    )

    parser.add_argument(
        "--method",
        choices=["gmr", "smpl"],
        default="gmr",
        help="Retargeting method to analyze"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (requires pandas)"
    )

    args = parser.parse_args()

    if args.motion:
        result = analyze_single_motion(args.motion, args.method)
        if result is None:
            return 1
    else:
        results = analyze_dataset(args.dataset, args.method)

        if not results:
            print("No analysis data available")
            return 1

        print(f"\n=== Dataset Summary ({len(results)} motions) ===")

        # Compute aggregate stats
        all_means = [r["mean_error_m"] for r in results]
        all_rmses = [r["RMSE_error_m"] for r in results]
        all_maxes = [r["max_error_m"] for r in results]
        all_fps = [r["retarget_fps"] for r in results]

        print(f"Mean position error: {np.mean(all_means):.4f}m (std: {np.std(all_means):.4f}m)")
        print(f"Max position error:  {np.max(all_maxes):.4f}m")
        print(f"Mean retarget FPS:   {np.mean(all_fps):.2f}")
        print(f"Mean RMSE error:     {np.mean(all_rmses):.4f}m (std: {np.std(all_rmses):.4f}m)")  # <-- added
        print(f"Max RMSE error:      {np.max(all_rmses):.4f}m") 

        if args.output:
            if not HAS_PANDAS:
                print("Error: pandas required for CSV output")
                return 1

            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False, float_format="%.5f")
            print(f"\nSaved detailed results to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
