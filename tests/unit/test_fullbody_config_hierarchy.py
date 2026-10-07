"""Composition tests for the maintained full-body config hierarchy."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).parents[2] / "fullbody"


def _compose(config_name: str, *overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=list(overrides))


def test_base_config_uses_tracking_observations() -> None:
    env = _compose("conf_fullbody").experiment.env_params

    assert env.use_egocentric_root_observations is True
    assert env.goal_type == "GoalFullBodyTracking"
    assert list(env.goal_params.lookahead_steps) == [1, 20, 40, 60, 80]
    assert env.goal_params.include_support_intent is False


def test_ppo_four_gpu_parent_shards_trajectories_across_devices() -> None:
    experiment = _compose("conf_fullbody_gmr_resnet_4gpu").experiment

    assert experiment.distributed.num_devices == 4
    assert experiment.trajectory.sharding == "static"


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("conf_fullbody*.yaml")), ids=lambda path: path.stem)
def test_maintained_configs_compose(config_path: Path) -> None:
    assert _compose(config_path.stem).experiment.env_params.env_name
