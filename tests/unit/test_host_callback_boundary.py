"""Architecture gates for host-owned logging and checkpointing."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_algorithm_tree_has_no_io_callback():
    algorithm_root = REPOSITORY_ROOT / "musclemimic" / "algorithms"
    offenders = []
    for path in algorithm_root.rglob("*.py"):
        if "io_callback" in path.read_text():
            offenders.append(path.relative_to(REPOSITORY_ROOT))

    assert offenders == []


def test_checkpoint_and_logging_routes_are_host_functions():
    runner_root = REPOSITORY_ROOT / "musclemimic" / "runner"
    sources = {
        path.name: path.read_text()
        for path in (
            runner_root / "training_runtime.py",
            runner_root / "checkpoint_saving.py",
            runner_root / "ppo_logging.py",
            runner_root / "flashsac_runtime.py",
            runner_root / "flashsac_checkpoint_saving.py",
            runner_root / "flashsac_logging.py",
        )
    }

    assert all("io_callback" not in source for source in sources.values())
    assert "jax.debug.callback" not in sources["training_runtime.py"]
    assert "jax.debug.callback" not in sources["flashsac_runtime.py"]
    assert "saver.save(runtime_state" in sources["training_runtime.py"]
    assert "saver.save(runtime_state.agent" in sources["flashsac_runtime.py"]
    assert "logging_callback(log)" in sources["training_runtime.py"]
    assert "logging_callback(log)" in sources["flashsac_runtime.py"]
