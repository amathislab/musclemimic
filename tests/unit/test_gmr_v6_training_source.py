from pathlib import Path
from typing import Any

import pytest
import yaml

from loco_mujoco.task_factories.dataset_confs import (
    GMR_TRAINING_CACHE_NAMESPACE,
    resolve_training_gmr_cache_namespace,
)
from musclemimic.utils.gmr_cache import GMR_CACHE_NAMESPACE

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _gmr_dataset_configs(value: Any):
    if isinstance(value, dict):
        if value.get("retargeting_method") == "gmr":
            yield value
        for child in value.values():
            yield from _gmr_dataset_configs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _gmr_dataset_configs(child)


def test_canonical_gmr_training_namespace_matches_cache_loader() -> None:
    assert GMR_TRAINING_CACHE_NAMESPACE == "gmr_v6"
    assert GMR_TRAINING_CACHE_NAMESPACE == GMR_CACHE_NAMESPACE


def test_training_loader_rejects_alternate_gmr_namespace() -> None:
    assert resolve_training_gmr_cache_namespace("gmr", None) == "gmr_v6"
    assert resolve_training_gmr_cache_namespace("gmr", "gmr_v6") == "gmr_v6"
    with pytest.raises(ValueError, match="must use the canonical gmr_v6"):
        resolve_training_gmr_cache_namespace("gmr", "experimental_gmr")


def test_all_explicit_gmr_training_configs_use_gmr_v6() -> None:
    config_paths = sorted((REPOSITORY_ROOT / "fullbody").glob("*.yaml"))
    config_paths += sorted((REPOSITORY_ROOT / "bimanual").glob("*.yaml"))
    checked = []
    for config_path in config_paths:
        document = yaml.safe_load(config_path.read_text())
        for dataset_config in _gmr_dataset_configs(document):
            checked.append(config_path)
            assert dataset_config.get("gmr_cache_namespace") == "gmr_v6", config_path

    assert checked
