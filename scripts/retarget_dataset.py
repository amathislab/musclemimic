"""
Parallel AMASS retargeting for muscle-based environments.
Supports MyoBimanualArm and MyoFullBody models with configurable datasets.
"""
# ruff: noqa: E402

from __future__ import annotations

import logging
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

import argparse
import csv
import multiprocessing as mp
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loco_mujoco.smpl.motion_spec import AMASSMotionSpec


BIMANUAL_TRAINING_AND_TESTING = (
    "AMASS_BIMANUAL_TRAIN_MOTIONS",
    "AMASS_BIMANUAL_TEST_MOTIONS",
)


def retarget_single_motion(
    args: tuple[AMASSMotionSpec, str, dict | None, bool],
) -> tuple[bool, str, str | None]:
    """
    Process a single motion in an isolated process.

    The worker pool recycles processes after a bounded number of motions so
    accumulated native-library memory is periodically returned to the OS.

    Args:
        args: Tuple of (motion_spec, model_type, retargeting_config, clear_cache)

    Returns:
        Tuple of (success, motion name, error message)
    """
    motion_spec, model_type, retargeting_config, clear_cache = args
    motion_name = motion_spec.path

    try:
        from loco_mujoco.task_factories import (
            AMASSDatasetConf,
            ImitationFactory,
        )
    except ImportError as e:
        print(f"[{motion_name}] Failed to import: {e}")
        return False, motion_name, f"{type(e).__name__}: {e}"

    try:
        # Create dataset configuration
        dataset_conf = AMASSDatasetConf([motion_spec])
        dataset_conf.clear_cache = clear_cache
        # This command builds the reusable per-motion retarget cache. Creating
        # an additional materialized runtime cache for every individual motion
        # wastes substantial disk space and is not used by dataset training.
        dataset_conf.write_materialized_cache = False

        # Apply retargeting configuration if provided
        if retargeting_config:
            dataset_conf.retargeting_method = retargeting_config["method"]
            if retargeting_config["method"] == "gmr":
                dataset_conf.gmr_config = retargeting_config["gmr_params"]

        # Configure environment creation based on model type
        env_params = {
            "amass_dataset_conf": dataset_conf,
            "headless": True,
        }

        # Add model-specific parameters
        if model_type == "MyoBimanualArm":
            env_params["goal_type"] = "GoalBimanualTrajMimic"

        # Create CPU-only environment - triggers automatic retargeting
        env = ImitationFactory.make(model_type, **env_params)
        del env

        print(f"[OK] {motion_name}")
        return True, motion_name, None

    except Exception as e:
        print(f"[FAIL] {motion_name}: {e}")
        return False, motion_name, f"{type(e).__name__}: {e}"


def write_failure_report(
    results: Sequence[tuple[bool, str, str | None]],
    output_path: Path,
    *,
    model: str,
    dataset_groups: Sequence[str],
    retargeting_method: str,
) -> int:
    """Write one CSV row per failed motion and return the failure count."""
    failed = [(motion, error or "Unknown error") for success, motion, error in results if not success]
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("motion", "error", "model", "dataset_groups", "retargeting_method"),
        )
        writer.writeheader()
        for motion, error in failed:
            writer.writerow(
                {
                    "motion": motion,
                    "error": error,
                    "model": model,
                    "dataset_groups": " ".join(dataset_groups),
                    "retargeting_method": retargeting_method,
                }
            )
    return len(failed)


def get_motion_dataset(dataset_group) -> list[AMASSMotionSpec]:
    """Load and combine motion datasets from one or more group names."""
    from loco_mujoco.task_factories.dataset_confs import (
        expand_amass_dataset_group_spec,
        get_amass_motion_spec_groups,
    )

    groups = get_amass_motion_spec_groups()
    group_names = expand_amass_dataset_group_spec(dataset_group)
    motion_specs = []
    for name in group_names:
        if name not in groups:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(groups.keys())}")
        motion_specs.extend(groups[name])
    return list({motion.cache_key: motion for motion in motion_specs}.values())


def get_gmr_motion_cache_path(
    motion_spec: AMASSMotionSpec,
    model_type: str,
    cache_root: str | Path | None = None,
    *,
    src_human: str = "smplh",
    gmr_config: dict | None = None,
) -> Path:
    """Return the versioned GMR cache path for one source motion selection."""
    from loco_mujoco.smpl import retargeting

    resolved_config = dict(gmr_config or {})
    resolved_config.setdefault("src_human", src_human)

    return retargeting.get_gmr_motion_cache_path(
        model_type,
        motion_spec,
        cache_root,
        gmr_config=resolved_config,
    )


def get_gmr_motion_rejection_path(
    motion_spec: AMASSMotionSpec,
    model_type: str,
    cache_root: str | Path | None = None,
    *,
    src_human: str = "smplh",
    gmr_config: dict | None = None,
) -> Path:
    """Return the persistent GMR quality-rejection path for one motion."""
    from loco_mujoco.smpl.retargeting import gmr_quality_rejection_path

    cache_path = get_gmr_motion_cache_path(
        motion_spec,
        model_type,
        cache_root,
        src_human=src_human,
        gmr_config=gmr_config,
    )
    return gmr_quality_rejection_path(cache_path)


def parse_arguments(argv: Sequence[str] | None = None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel AMASS retargeting for muscle-based environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Bimanual train and test groups: "
            f"{' '.join(BIMANUAL_TRAINING_AND_TESTING)}. "
            "Run bimanual groups separately from full-body groups because they use different models."
        ),
    )

    parser.add_argument(
        "--model",
        choices=["MyoBimanualArm", "MyoFullBody"],
        default="MyoBimanualArm",
        help="Model type to retarget for",
    )

    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["AMASS_BIMANUAL_MARGINAL_MOTIONS"],
        help=(
            "One or more dataset group names. Both training and testing can be "
            "processed together by passing the two corresponding group names"
        ),
    )

    parser.add_argument(
        "--workers", type=int, default=None, help="Number of worker processes (default: min(cpu_count, 8, num_motions))"
    )

    # Retargeting method configuration
    parser.add_argument(
        "--retargeting-method",
        choices=["smpl", "gmr"],
        default="smpl",
        help="Retargeting method: 'smpl' (optimization-based, default) or 'gmr' (IK-based)",
    )

    # GMR-specific configuration (only used when --retargeting-method=gmr)
    gmr_group = parser.add_argument_group("GMR Configuration (only applies when --retargeting-method=gmr)")
    gmr_group.add_argument("--gmr-src-human", default="smplh", help="Source human model for GMR")
    gmr_group.add_argument("--gmr-target-fps", type=int, default=30, help="Target FPS for GMR retargeting")
    gmr_group.add_argument("--gmr-solver", default="daqp", help="IK solver for GMR (e.g., daqp, qpswift)")
    gmr_group.add_argument("--gmr-damping", type=float, default=0.5, help="Damping factor for GMR solver")
    gmr_group.add_argument(
        "--gmr-offset-to-ground",
        action="store_true",
        default=False,
        help="Enable GMR's per-frame source-joint grounding, which changes vertical motion",
    )
    gmr_group.add_argument(
        "--gmr-no-offset-to-ground",
        dest="gmr_offset_to_ground",
        action="store_false",
        help="Disable GMR's per-frame source-joint grounding (default)",
    )
    gmr_group.add_argument(
        "--gmr-use-velocity-limit", action="store_true", default=False, help="Use velocity limits in GMR"
    )
    gmr_group.add_argument("--gmr-verbose", action="store_true", default=False, help="Enable verbose GMR output")
    gmr_group.add_argument(
        "--gmr-use-fitted-shape",
        dest="gmr_use_fitted_shape",
        action="store_true",
        default=None,
        help="Fit source anatomy to the target robot before retargeting",
    )
    gmr_group.add_argument(
        "--gmr-no-use-fitted-shape",
        dest="gmr_use_fitted_shape",
        action="store_false",
        help="Disable source anatomy fitting for an explicit manual baseline",
    )

    # Cache control
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Clear existing cache and force re-retargeting even if cached files exist",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Process only motions missing from the active versioned GMR cache",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Number of deterministic dataset shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard selected by this process")
    parser.add_argument(
        "--failed-motions-csv",
        type=Path,
        default=Path("retargeting_failed_motions.csv"),
        help="CSV report written after the run, with one row per failed motion",
    )

    args = parser.parse_args(argv)
    args.gmr_use_fitted_shape = bool(args.gmr_use_fitted_shape)
    return args


def build_gmr_params(args: argparse.Namespace) -> dict:
    """Return explicit GMR parameters for one source model."""
    return {
        "src_human": args.gmr_src_human,
        "target_fps": args.gmr_target_fps,
        "solver": args.gmr_solver,
        "damping": args.gmr_damping,
        "offset_to_ground": args.gmr_offset_to_ground,
        "use_velocity_limit": args.gmr_use_velocity_limit,
        "verbose": args.gmr_verbose,
        "use_fitted_shape": args.gmr_use_fitted_shape,
    }


def main() -> int:
    """Main parallel retargeting function."""
    args = parse_arguments()

    print(f"Parallel AMASS Retargeting for {args.model} (CPU-only)")
    print("=" * 60)

    # Load dataset
    try:
        all_motions = get_motion_dataset(args.dataset)
    except (ImportError, ValueError, TypeError) as e:
        print(f"Error loading dataset: {e}")
        return 1

    if not all_motions:
        print("Error: no motions found for the specified dataset groups.")
        return 1
    if args.num_shards < 1:
        print("Error: --num-shards must be at least 1")
        return 1
    if not 0 <= args.shard_index < args.num_shards:
        print("Error: --shard-index must be in [0, --num-shards)")
        return 1

    total_unique_motions = len(all_motions)
    all_motions = all_motions[args.shard_index :: args.num_shards]
    print(f"Dataset shard: {args.shard_index}/{args.num_shards} ({len(all_motions)}/{total_unique_motions} motions)")

    gmr_params = None
    if args.retargeting_method == "gmr":
        gmr_params = build_gmr_params(args)

    if args.skip_existing:
        if args.retargeting_method != "gmr":
            print("Error: --skip-existing currently requires --retargeting-method=gmr")
            return 1
        existing = [
            motion
            for motion in all_motions
            if get_gmr_motion_cache_path(
                motion,
                args.model,
                src_human=args.gmr_src_human,
                gmr_config=gmr_params,
            ).exists()
        ]
        rejected = [
            motion
            for motion in all_motions
            if get_gmr_motion_rejection_path(
                motion,
                args.model,
                src_human=args.gmr_src_human,
                gmr_config=gmr_params,
            ).exists()
        ]
        terminal_keys = {motion.cache_key for motion in (*existing, *rejected)}
        all_motions = [motion for motion in all_motions if motion.cache_key not in terminal_keys]
        print(f"Existing versioned GMR caches: {len(existing)}")
        print(f"Persistent quality rejections: {len(rejected)}")
        print(f"Missing versioned GMR caches: {len(all_motions)}")
        if not all_motions:
            print("All requested motions already have versioned GMR caches")
            return 0

    print(f"Model: {args.model}")
    print(f"Dataset: {', '.join(args.dataset)} ({len(all_motions)} motions)")

    # Build retargeting configuration
    retargeting_config = None
    if args.retargeting_method:
        method_display = args.retargeting_method.upper()
        print(f"Retargeting Method: {method_display}")

        retargeting_config = {"method": args.retargeting_method}

        # Add GMR-specific configuration if GMR is selected
        if args.retargeting_method == "gmr":
            retargeting_config["gmr_params"] = gmr_params
            print("GMR Configuration:")
            print(f"  Source Human: {args.gmr_src_human}")
            print(f"  Target FPS: {args.gmr_target_fps}")
            print(f"  Solver: {args.gmr_solver}")
            print(f"  Damping: {args.gmr_damping}")
            print(f"  Offset to Ground: {args.gmr_offset_to_ground}")
            print(f"  Use Velocity Limit: {args.gmr_use_velocity_limit}")
            print(f"  Use Fitted Shape: {args.gmr_use_fitted_shape}")

    if args.clear_cache:
        print("Cache clearing ENABLED: Will overwrite existing cached files")
    else:
        print("Retargeting system will handle caching automatically")
        print("  (GMR and SMPL caches are separate)")

    # Determine worker configuration
    if args.workers:
        max_workers = min(args.workers, len(all_motions))
    else:
        max_workers = min(mp.cpu_count(), 8, len(all_motions))
    print(f"Using {max_workers} CPU worker processes (maxtasksperchild=10 for memory control)")

    # Prepare single-motion args (each motion runs in isolated process)
    worker_args = [(motion_spec, args.model, retargeting_config, args.clear_cache) for motion_spec in all_motions]

    # Execute parallel processing with process recycling
    start_time = time.time()
    print(f"\nProcessing {len(all_motions)} motions in isolated worker processes...")

    # Use spawn method for clean worker processes
    original_method = mp.get_start_method(allow_none=True)
    try:
        mp.set_start_method("spawn", force=True)

        # maxtasksperchild=10: each worker processes 10 motions then dies
        # This guarantees RSS is returned to OS after each motion
        with mp.Pool(processes=max_workers, maxtasksperchild=10) as pool:
            results = list(pool.imap_unordered(retarget_single_motion, worker_args))

    finally:
        if original_method:
            mp.set_start_method(original_method, force=True)

    elapsed_time = time.time() - start_time

    # Aggregate results
    total_successful = sum(1 for success, _, _ in results if success)
    total_failed = write_failure_report(
        results,
        args.failed_motions_csv,
        model=args.model,
        dataset_groups=args.dataset,
        retargeting_method=args.retargeting_method,
    )
    failed_motions = [name for success, name, _ in results if not success]

    # Report results
    print("\nProcessing Results:")
    print(f"  Total processed: {total_successful + total_failed}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    print(f"  Failure report: {args.failed_motions_csv.expanduser().resolve()}")
    print(f"  Processing time: {elapsed_time:.1f} seconds")

    if len(all_motions) > 0:
        avg_time = elapsed_time / len(all_motions)
        print(f"  Average time per motion: {avg_time:.2f} seconds")

    if max_workers > 1:
        estimated_sequential_time = elapsed_time * max_workers
        speedup_factor = estimated_sequential_time / elapsed_time
        print(f"  Estimated speedup: {speedup_factor:.1f}x over sequential processing")

    if failed_motions:
        print(f"\nFailed motions ({len(failed_motions)}):")
        for motion in failed_motions[:20]:  # Show first 20
            print(f"  {motion}")
        if len(failed_motions) > 20:
            print(f"  ... and {len(failed_motions) - 20} more")

    success_rate = (total_successful / len(all_motions)) * 100 if len(all_motions) > 0 else 0
    print("\nFinal Status:")
    print(f"  Success rate: {success_rate:.1f}% ({total_successful}/{len(all_motions)})")

    if total_failed == 0:
        print("All motions processed successfully")
        return 0
    else:
        print(f"Warning: {total_failed} motions failed to process")
        return 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
