"""Smoke tests for the fullbody and bimanual experiment entrypoints."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_MODULES = ("fullbody.experiment", "bimanual.experiment")
EXPERIMENT_ENTRYPOINTS = (
    REPOSITORY_ROOT / "fullbody" / "experiment.py",
    REPOSITORY_ROOT / "bimanual" / "experiment.py",
)


@pytest.mark.parametrize("module_name", EXPERIMENT_MODULES)
def test_experiment_module_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert callable(module.experiment)


@pytest.mark.parametrize("entrypoint", EXPERIMENT_ENTRYPOINTS)
def test_experiment_cfg_job_smoke(entrypoint: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(entrypoint), "--cfg", "job"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "experiment:" in result.stdout
