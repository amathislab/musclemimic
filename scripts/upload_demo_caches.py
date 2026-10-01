#!/usr/bin/env python3
"""
Upload pre-retargeted demo motion caches to HuggingFace.

Usage:
    python scripts/upload_demo_caches.py --repo YOUR_HF_USERNAME/musclemimic-demos

This uploads a few small pre-retargeted motions so users can test without AMASS.
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload_demo_caches(repo_id: str, cache_base: Path):
    """
    Upload demo motion caches to HuggingFace.

    Args:
        repo_id: HuggingFace repo (e.g., "username/musclemimic-demos")
        cache_base: Base path to cache directory (e.g., /users/shared/loco-mujoco-caches/AMASS)
    """
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"[OK] Repository ready: {repo_id}")
    except Exception as e:
        print(f"[WARN] Could not create repo: {e}")
        return

    # Demo motions to upload (small, useful for testing)
    demo_files = {
        "MyoBimanualArm": [
            "KIT/3/tennis_forehand_right04_poses.npz",
            "BMLhandball/S10_Expert/Trial_upper_left_057_poses.npz",
            "BioMotionLab_NTroje/rub109/0021_catching_and_throwing_poses.npz",
        ],
        "MyoFullBody": [
            "KIT/12/LeftTurn03_poses.npz",
        ],
    }

    uploaded_count = 0
    for env_name, motions in demo_files.items():
        for motion_path in motions:
            local_file = cache_base / env_name / motion_path

            if not local_file.exists():
                print(f"[WARN] Skipping {motion_path} (not found)")
                continue

            # Upload to HF with same structure: {env_name}/{motion_path}
            hf_path = f"{env_name}/{motion_path}"

            try:
                api.upload_file(
                    path_or_fileobj=str(local_file),
                    path_in_repo=hf_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                print(f"[OK] Uploaded: {hf_path}")
                uploaded_count += 1
            except Exception as e:
                print(f"[WARN] Failed to upload {motion_path}: {e}")

    print(f"\n[OK] Uploaded {uploaded_count} demo motion caches")
    print(f"Repository: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload demo caches to HuggingFace")
    parser.add_argument(
        "--repo",
        type=str,
        default="musclemimic/demo-motions",
        help="HuggingFace repo ID (default: musclemimic/demo-motions)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Path to AMASS cache directory",
    )

    args = parser.parse_args()

    if args.cache_dir is None:
        print("Error: --cache-dir is required")
        return

    if not args.cache_dir.exists():
        print(f"[WARN] Cache directory not found: {args.cache_dir}")
        return

    print(f"Uploading demo caches from: {args.cache_dir}")
    print(f"To HuggingFace repo: {args.repo}")
    print()

    upload_demo_caches(args.repo, args.cache_dir)


if __name__ == "__main__":
    main()
