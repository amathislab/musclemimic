"""
Upload checkpoint to HuggingFace.

Usage:
    python scripts/upload_checkpoint.py /path/to/checkpoint_30000 --repo your-org/model-name
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DOWNLOAD_STATS_CONFIG = "config.json"


def _read_checkpoint_metadata(checkpoint_path: Path) -> dict:
    metadata_path = checkpoint_path / "metadata" / "metadata"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open() as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _build_download_stats_config(checkpoint_path: Path) -> bytes:
    """Build the root config.json used by Hugging Face download counters."""
    metadata = _read_checkpoint_metadata(checkpoint_path)
    config = {
        "model_type": "musclemimic-policy",
        "checkpoint_format": "orbax",
        "checkpoint_name": checkpoint_path.name,
    }
    for key in ("algo_version", "schema_version", "backend", "env_name", "global_timestep"):
        if key in metadata:
            config[key] = metadata[key]
    return json.dumps(config, indent=2, sort_keys=True).encode("utf-8")


def _upload_download_stats_config(api: HfApi, checkpoint_path: Path, repo_id: str) -> None:
    """Upload config.json so Hub model download stats count policy downloads."""
    api.upload_file(
        path_or_fileobj=_build_download_stats_config(checkpoint_path),
        path_in_repo=DOWNLOAD_STATS_CONFIG,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add download stats config",
    )


def upload_checkpoint(checkpoint_path: Path, repo_id: str):
    """
    Upload an Orbax checkpoint directory to HuggingFace.

    Args:
        checkpoint_path: Path to checkpoint directory (e.g., checkpoint_30000)
        repo_id: HuggingFace repo ID (e.g., "your-org/model-name")
    """
    api = HfApi()

    # Validate checkpoint structure
    required = ["train_state", "metadata", "config", "_CHECKPOINT_METADATA"]
    missing = [r for r in required if not (checkpoint_path / r).exists()]
    if missing:
        raise ValueError(f"Invalid checkpoint - missing: {missing}")

    # Create repo if needed
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repository ready: {repo_id}")
    except Exception as e:
        print(f"Warning: {e}")

    # Upload
    print(f"Uploading {checkpoint_path} to {repo_id}...")
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        repo_type="model",
    )
    _upload_download_stats_config(api, checkpoint_path, repo_id)
    print(f"Done: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint directory")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repo ID")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    upload_checkpoint(args.checkpoint, args.repo)


if __name__ == "__main__":
    main()
