from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

# Fields excluded from config hash (they don't affect training identity)
_HASH_EXCLUDE_FIELDS = frozenset({
    "resume_from",
    "reset_logging_timestep",
    "checkpoint_dir",
    "checkpoint_root",
    "run_id",
    "auto_resume",
})


def _is_checkpoint_complete(checkpoint_path: Path) -> bool:
    """Check if checkpoint has completed writing (Orbax finalized)."""
    from orbax.checkpoint import utils as ocp_utils

    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        return False

    # Check for Orbax checkpoint metadata marker file
    if not (checkpoint_path / "_CHECKPOINT_METADATA").exists():
        return False

    return ocp_utils.is_checkpoint_finalized(checkpoint_path)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> str | None:
    """Find latest complete checkpoint_* in directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        Path to latest complete checkpoint directory, or None if none found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return None

    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("checkpoint_"):
            try:
                step = int(name.split("_")[1])
                ckpt_path = checkpoint_dir / name
                # Only include checkpoints with completed metadata
                if _is_checkpoint_complete(ckpt_path):
                    checkpoints.append((step, ckpt_path))
            except (IndexError, ValueError):
                continue

    if not checkpoints:
        return None

    return str(max(checkpoints, key=lambda x: x[0])[1])


def config_hash(config: Any, exclude: frozenset[str] | None = None) -> str:
    """Compute stable hash of experiment config (excludes volatile fields).

    Args:
        config: OmegaConf config or dict to hash.
        exclude: Additional fields to exclude from hash.

    Returns:
        12-character hex hash string.
    """
    import hashlib

    exclude_set = _HASH_EXCLUDE_FIELDS | (exclude or frozenset())

    # Always use OmegaConf.to_container for deep conversion to native Python types
    if OmegaConf.is_config(config):
        cfg_dict = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, dict):
        cfg_dict = config
    else:
        cfg_dict = dict(config)

    def _remove_excluded(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in exclude_set}

    cfg_dict = _remove_excluded(cfg_dict)

    # Stable JSON serialization
    cfg_str = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256(cfg_str.encode()).hexdigest()[:12]


def _get_git_sha() -> str | None:
    """Get current git SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def write_manifest(
    checkpoint_dir: str | Path,
    config: Any,
    config_hash_value: str,
) -> None:
    """Write manifest.json in checkpoint directory on first save.

    Args:
        checkpoint_dir: Checkpoint directory path.
        config: Experiment config to save.
        config_hash_value: Pre-computed config hash.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"

    # Only write once
    if manifest_path.exists():
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_hash": config_hash_value,
        "git_sha": _get_git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config": OmegaConf.to_container(config, resolve=True),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def validate_checkpoint_compatibility(
    checkpoint_dir: str | Path,
    current_hash: str,
) -> bool:
    """Validate checkpoint config hash matches current config.

    Args:
        checkpoint_dir: Checkpoint directory containing manifest.json.
        current_hash: Current config hash to compare.

    Returns:
        True if compatible (or no manifest), False with warning if mismatch.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"

    if not manifest_path.exists():
        return True

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        saved_hash = manifest.get("config_hash")
        if saved_hash and saved_hash != current_hash:
            print(f"WARNING: Config hash mismatch!")
            print(f"  Checkpoint: {saved_hash}")
            print(f"  Current:    {current_hash}")
            print("  Continuing anyway (override with explicit resume_from if needed)")
            return False
    except Exception as e:
        print(f"Warning: Could not read manifest: {e}")

    return True


def resolve_checkpoint_dir(
    configured_ckpt_dir: str,
    launch_dir: str,
    result_dir: str,
    experiment_id: str,
    auto_resume: bool,
    checkpoint_root: str | None = None,
) -> str:
    """Resolve the checkpoint directory path based on auto_resume setting.

    Args:
        configured_ckpt_dir: Default checkpoint directory name (e.g., "checkpoints").
        launch_dir: Directory where script was launched from.
        result_dir: Hydra per-run output directory.
        experiment_id: Experiment identifier (config hash or run_id).
        auto_resume: Whether auto-resume is enabled.
        checkpoint_root: Optional explicit checkpoint root path.

    Returns:
        Resolved absolute checkpoint directory path.
    """
    import os

    # Determine base directory
    if checkpoint_root:
        base = checkpoint_root if os.path.isabs(checkpoint_root) else os.path.join(launch_dir, checkpoint_root)
    elif auto_resume:
        base = configured_ckpt_dir if os.path.isabs(configured_ckpt_dir) else os.path.join(launch_dir, configured_ckpt_dir)
    else:
        base = configured_ckpt_dir if os.path.isabs(configured_ckpt_dir) else os.path.join(result_dir, configured_ckpt_dir)

    # Append experiment_id for auto_resume, otherwise use unique suffix
    if auto_resume:
        return os.path.join(base, experiment_id)
    else:
        return base  # Caller adds unique suffix


def _download_from_huggingface(repo_id: str, revision: str = None) -> str:
    """Download checkpoint from HuggingFace and return local path.

    Args:
        repo_id: HuggingFace repo ID

    Returns:
        Local path to the downloaded checkpoint directory.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
    )
    return local_dir


def _canonicalize_resume_path(path_like: str, revision: str = None) -> str:
    r"""Return a concrete checkpoint path to resume from.

    Accepts multiple user-friendly inputs and normalizes to an Orbax checkpoint
    directory path (.../checkpoint_<step>).

    Supported formats:
        - Local path: /path/to/checkpoint_30000
        - Local parent dir: /path/to/checkpoints/ (picks latest)
        - HuggingFace: hf://username/repo-name

    Raises:
        ValueError: If path does not exist, is a file, or contains no valid checkpoints.
    """
    # Handle HuggingFace URLs
    if path_like.startswith("hf://"):
        repo_id = path_like[5:]  # Remove "hf://" prefix
        print(f"Downloading checkpoint from HuggingFace: {repo_id}")
        local_path = _download_from_huggingface(repo_id, revision=revision)
        print(f"Downloaded to: {local_path}")
        # Recursively canonicalize the downloaded path
        return _canonicalize_resume_path(local_path)

    p = Path(path_like)
    if not p.exists():
        raise ValueError(f"Checkpoint path does not exist: {path_like}")

    if p.is_file():
        raise ValueError(f"Checkpoint path is a file, not a directory: {path_like}")

    name = p.name
    if re.match(r"^checkpoint_\d+$", name):
        return str(p)

    # Directory that may contain multiple checkpoint_* subdirs
    subdirs = [d for d in p.iterdir() if d.is_dir() and re.match(r"^checkpoint_\d+$", d.name)]
    # Filter to only complete checkpoints (have metadata)
    complete_subdirs = [d for d in subdirs if _is_checkpoint_complete(d)]
    if complete_subdirs:
        # Pick latest complete checkpoint by numeric step
        step = max(int(d.name.split("_")[-1]) for d in complete_subdirs)
        return str(p / f"checkpoint_{step}")

    # Check if it's an Orbax checkpoint directory (has train_state subdir)
    if (p / "train_state").is_dir():
        # Read step from metadata/metadata and create expected checkpoint_<step> structure
        import json
        metadata_file = p / "metadata" / "metadata"
        if metadata_file.is_symlink():
            metadata_file = metadata_file.resolve()
        with open(metadata_file) as f:
            step = json.load(f).get("update_number", 0)
        # Create symlink with expected name in parent directory
        symlink_path = p.parent / f"checkpoint_{step}"
        if not symlink_path.exists():
            symlink_path.symlink_to(p)
        return str(symlink_path)

    # Not a recognizable checkpoint location
    raise ValueError(f"No valid checkpoints found in: {path_like}")


def resume_or_fresh(
    env: Any,
    agent_conf: Any,
    algorithm_cls: Any,
    config: Any,
    mh: Any,
    logging_callback,
    logging_interval: int = 1,
    val_env: Any = None,
    apply_resume_resets: bool = True,
):
    """Return a train function that resumes from checkpoint or starts fresh.

    If `experiment.resume_from` is set, validates the path and raises ValueError
    if invalid. If not set, starts fresh training.

    Raises:
        ValueError: If resume_from is set but path is invalid or contains no checkpoints.
    """
    resume_from = getattr(config.experiment, "resume_from", None)
    revision = getattr(config.experiment, "revision", None)

    # Fresh training
    if not resume_from:
        return algorithm_cls.build_train_fn(
            env,
            agent_conf,
            mh=mh,
            online_logging_callback=logging_callback,
            logging_interval=getattr(
                config.experiment, "online_logging_interval", logging_interval
            ),
            val_env=val_env,
        )

    # Normalize resume path to a specific checkpoint (raises ValueError if invalid)
    canonical = _canonicalize_resume_path(resume_from, revision=revision)

    # Build resume train fn that loads on first call
    return algorithm_cls.build_resume_train_fn_from_path(
        env,
        agent_conf,
        canonical,
        mh=mh,
        online_logging_callback=logging_callback,
        logging_interval=getattr(
            config.experiment, "online_logging_interval", logging_interval
        ),
        val_env=val_env,
        apply_resume_resets=apply_resume_resets,
    )
