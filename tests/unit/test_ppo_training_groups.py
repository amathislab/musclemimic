"""PPO motion-family sampler configuration tests."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from musclemimic.algorithms.ppo.train import _install_adaptive_sampling_groups


def _environment(group_ids=(0, 1, 0)):
    return SimpleNamespace(
        th=SimpleNamespace(n_trajectories=3),
        trajectory_group_ids=group_ids,
        trajectory_group_names=("stairs", "ramps"),
    )


def test_motion_group_balancing_can_be_disabled_for_legacy_resume():
    config = OmegaConf.create({"adaptive_sampling": {"balance_motion_groups": False}})

    _install_adaptive_sampling_groups(config, _environment())

    assert "group_ids" not in config.adaptive_sampling
    assert "group_names" not in config.adaptive_sampling


def test_motion_group_balancing_installs_valid_environment_groups():
    config = OmegaConf.create({"adaptive_sampling": {"balance_motion_groups": True}})

    _install_adaptive_sampling_groups(config, _environment())

    assert list(config.adaptive_sampling.group_ids) == [0, 1, 0]
    assert list(config.adaptive_sampling.group_names) == ["stairs", "ramps"]


def test_motion_group_balancing_rejects_incomplete_group_ids():
    config = OmegaConf.create({"adaptive_sampling": {"balance_motion_groups": True}})

    with pytest.raises(ValueError, match="one group id per trajectory"):
        _install_adaptive_sampling_groups(config, _environment((0, 1)))
