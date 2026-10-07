from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import hf_hub_download

import loco_mujoco
from loco_mujoco.smpl.gmr_cache_identity import GMR_CACHE_NAMESPACE, gmr_motion_relative_path
from loco_mujoco.smpl.motion_spec import AMASSMotionSpec
from musclemimic.utils.logging import setup_logger

logger = setup_logger(__name__, identifier="[GMRCache]")

_BIMANUAL_ENV_NAME = "MyoBimanualArm"
_GMR_DATASET_REPOS = {
    "MyoFullBody": "amathislab/musclemimic-retargeted",
    _BIMANUAL_ENV_NAME: "amathislab/musclemimic-bimanual-retargeted",
}


def _load_dataset_group_helpers():
    from loco_mujoco.task_factories.dataset_confs import (
        expand_amass_dataset_group_spec,
        get_amass_dataset_groups,
    )

    return expand_amass_dataset_group_spec, get_amass_dataset_groups


def normalize_gmr_env_name(env_name: str) -> str:
    normalized = env_name.removeprefix("Mjx")
    if normalized not in _GMR_DATASET_REPOS:
        supported = ", ".join(sorted(_GMR_DATASET_REPOS))
        raise ValueError(f"Unsupported GMR cache environment '{env_name}'. Supported values: {supported}")
    return normalized


def get_gmr_dataset_repo_id(env_name: str) -> str:
    return _GMR_DATASET_REPOS[normalize_gmr_env_name(env_name)]


def resolve_gmr_cache_root(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    path = os.environ.get("CONVERTED_AMASS_PATH") or os.environ.get("MUSCLEMIMIC_CONVERTED_AMASS_PATH")
    if path:
        return Path(path).expanduser()

    path_config = loco_mujoco.load_path_config()
    path = path_config.get("CONVERTED_AMASS_PATH") or path_config.get("MUSCLEMIMIC_CONVERTED_AMASS_PATH")
    if path:
        return Path(path).expanduser()

    return loco_mujoco.get_musclemimic_home() / "caches" / "AMASS"


def resolve_gmr_dataset_names(dataset_group: str | Sequence[str]) -> list[str]:
    expand_amass_dataset_group_spec, get_amass_dataset_groups = _load_dataset_group_helpers()
    group_names = expand_amass_dataset_group_spec(dataset_group)
    available_groups = get_amass_dataset_groups()

    missing = [group for group in group_names if group not in available_groups]
    if missing:
        available = ", ".join(sorted(available_groups))
        raise ValueError(f"Unknown dataset group(s): {missing}. Available groups: {available}")

    dataset_names: list[str] = []
    seen: set[str] = set()
    for group in group_names:
        for dataset_name in available_groups[group]:
            if dataset_name not in seen:
                seen.add(dataset_name)
                dataset_names.append(dataset_name)
    return dataset_names


def resolve_gmr_motion_specs(dataset_group: str | Sequence[str]) -> list[AMASSMotionSpec]:
    """Expand download groups while preserving clip intervals and group order."""
    from loco_mujoco.task_factories.dataset_confs import (
        expand_amass_dataset_group_spec,
        get_amass_motion_spec_groups,
    )

    groups = get_amass_motion_spec_groups()
    selected = {}
    for name in expand_amass_dataset_group_spec(dataset_group):
        if name not in groups:
            raise ValueError(f"Unknown dataset group: {name}")
        for motion in groups[name]:
            selected.setdefault(motion.cache_key, motion)
    return list(selected.values())


def _remote_gmr_cache_path(env_name: str, dataset_name: str | AMASSMotionSpec) -> str:
    normalized_env_name = normalize_gmr_env_name(env_name)
    return gmr_motion_relative_path(normalized_env_name, dataset_name).as_posix()


def _local_gmr_cache_path(
    env_name: str, dataset_name: str | AMASSMotionSpec, cache_root: str | Path | None = None
) -> Path:
    root = resolve_gmr_cache_root(cache_root)
    normalized_env_name = normalize_gmr_env_name(env_name)
    return root / gmr_motion_relative_path(normalized_env_name, dataset_name)


def download_gmr_cache(
    dataset_name: str | AMASSMotionSpec,
    env_name: str = "MyoFullBody",
    repo_id: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    normalized_env_name = normalize_gmr_env_name(env_name)
    repo_id = repo_id or get_gmr_dataset_repo_id(normalized_env_name)
    cache_root = resolve_gmr_cache_root(cache_dir)
    local_path = _local_gmr_cache_path(normalized_env_name, dataset_name, cache_root=cache_root)

    if local_path.exists() and not force_download:
        logger.info("Using existing GMR cache: %s", local_path)
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_path = _remote_gmr_cache_path(normalized_env_name, dataset_name)
    logger.info("Downloading GMR cache %s -> %s", remote_path, local_path)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        repo_type="dataset",
        force_download=force_download,
    )
    shutil.copy2(downloaded, local_path)
    return local_path


def try_download_gmr_cache(
    dataset_name: str | AMASSMotionSpec,
    env_name: str = "MyoFullBody",
    repo_id: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    logger_override=None,
) -> Path | None:
    active_logger = logger_override or logger
    try:
        path = download_gmr_cache(
            dataset_name=dataset_name,
            env_name=env_name,
            repo_id=repo_id,
            cache_dir=cache_dir,
            force_download=force_download,
        )
        active_logger.info("Downloaded GMR cache: %s", path)
        return path
    except Exception as exc:
        active_logger.warning("Could not download GMR cache for %s: %s", dataset_name, exc)
        return None


def download_gmr_dataset_group(
    dataset_group: str | Sequence[str],
    env_name: str = "MyoFullBody",
    repo_id: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> list[Path]:
    dataset_names = resolve_gmr_motion_specs(dataset_group)
    normalized_env_name = normalize_gmr_env_name(env_name)
    repo_id = repo_id or get_gmr_dataset_repo_id(normalized_env_name)
    cache_root = resolve_gmr_cache_root(cache_dir)

    logger.info(
        "Downloading %s GMR caches for %s from %s",
        len(dataset_names),
        normalized_env_name,
        repo_id,
    )
    logger.info("Saving GMR caches under %s", cache_root)

    downloaded_paths: list[Path] = []
    failures: list[str] = []
    for index, dataset_name in enumerate(dataset_names, start=1):
        try:
            path = download_gmr_cache(
                dataset_name=dataset_name,
                env_name=normalized_env_name,
                repo_id=repo_id,
                cache_dir=cache_root,
                force_download=force_download,
            )
            logger.info("  [%s/%s] %s", index, len(dataset_names), dataset_name)
            downloaded_paths.append(path)
        except Exception as exc:
            failures.append(dataset_name.cache_key)
            logger.warning("  [%s/%s] failed: %s (%s)", index, len(dataset_names), dataset_name, exc)

    if failures:
        failed_preview = ", ".join(failures[:5])
        suffix = " ..." if len(failures) > 5 else ""
        raise RuntimeError(
            f"Failed to download {len(failures)} GMR cache(s) for {normalized_env_name}: {failed_preview}{suffix}"
        )

    logger.info("Finished downloading %s GMR caches to %s", len(downloaded_paths), cache_root)
    return downloaded_paths


def download_gmr_dataset_group_cli() -> None:
    parser = argparse.ArgumentParser(description="Download pre-retargeted GMR caches from Hugging Face.")
    parser.add_argument(
        "--dataset-group",
        required=True,
        help="AMASS dataset group name or a + joined combination, e.g. KIT_KINESIS_TRAINING_MOTIONS",
    )
    parser.add_argument(
        "--env-name",
        default="MyoFullBody",
        help="Target environment. Defaults to MyoFullBody. For the arm model use MyoBimanualArm.",
    )
    parser.add_argument("--repo-id", default=None, help="Override the Hugging Face dataset repo id.")
    parser.add_argument("--cache-dir", default=None, help="Override the local converted AMASS cache root.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    args = parser.parse_args()

    download_gmr_dataset_group(
        dataset_group=args.dataset_group,
        env_name=args.env_name,
        repo_id=args.repo_id,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
    )
