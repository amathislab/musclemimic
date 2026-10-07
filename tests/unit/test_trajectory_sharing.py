"""Tests for validation trajectory sharing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from loco_mujoco.trajectory import TrajectoryCacheType
from musclemimic.runner.trajectory_sharing import (
    attach_shared_validation_trajectory,
    should_share_validation_trajectory,
)


def test_validation_without_dataset_override_shares_training_trajectory():
    config = OmegaConf.create({"experiment": {"validation": {"active": True}}})

    assert should_share_validation_trajectory(config) is True


def test_inactive_validation_does_not_share_training_trajectory():
    config = OmegaConf.create({"experiment": {"validation": {"active": False}}})

    assert should_share_validation_trajectory(config) is False


def test_validation_can_disable_trajectory_sharing():
    config = OmegaConf.create(
        {
            "experiment": {
                "validation": {
                    "active": True,
                    "share_trajectory_handler": False,
                }
            }
        }
    )

    assert should_share_validation_trajectory(config) is False


def test_explicit_amass_override_uses_separate_trajectory_even_when_motions_match():
    config = OmegaConf.create(
        {
            "experiment": {
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["motion_a", "motion_b"],
                        }
                    }
                },
                "validation": {
                    "active": True,
                    "amass_dataset_conf": {
                        "rel_dataset_path": ["motion_b", "motion_a", "motion_a"],
                    },
                },
            }
        }
    )

    assert should_share_validation_trajectory(config) is False


def test_explicit_amass_override_does_not_share_identical_sampled_subset():
    dataset_config = {
        "rel_dataset_path": ["motion_a", "motion_b", "motion_c"],
        "max_motions": 1,
    }
    config = OmegaConf.create(
        {
            "experiment": {
                "task_factory": {"params": {"amass_dataset_conf": dataset_config}},
                "validation": {
                    "active": True,
                    "amass_dataset_conf": dataset_config,
                },
            }
        }
    )

    assert should_share_validation_trajectory(config) is False


def test_different_retargeting_method_uses_separate_validation_trajectory():
    config = OmegaConf.create(
        {
            "experiment": {
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["motion_a"],
                            "retargeting_method": "smpl",
                        }
                    }
                },
                "validation": {
                    "active": True,
                    "amass_dataset_conf": {
                        "rel_dataset_path": ["motion_a"],
                        "retargeting_method": "gmr",
                    },
                },
            }
        }
    )

    assert should_share_validation_trajectory(config) is False


def test_different_max_motions_use_separate_trajectories():
    config = OmegaConf.create(
        {
            "experiment": {
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["motion_a", "motion_b", "motion_c"],
                            "max_motions": 1,
                        }
                    }
                },
                "validation": {
                    "active": True,
                    "amass_dataset_conf": {
                        "rel_dataset_path": ["motion_a", "motion_b", "motion_c"],
                        "max_motions": 2,
                    },
                },
            }
        }
    )

    assert should_share_validation_trajectory(config) is False


def test_different_max_motion_seeds_use_separate_trajectories():
    config = OmegaConf.create(
        {
            "experiment": {
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["motion_a", "motion_b", "motion_c"],
                            "max_motions": 1,
                            "max_motions_seed": 0,
                        }
                    }
                },
                "validation": {
                    "active": True,
                    "amass_dataset_conf": {
                        "rel_dataset_path": ["motion_a", "motion_b", "motion_c"],
                        "max_motions": 1,
                        "max_motions_seed": 1,
                    },
                },
            }
        }
    )

    assert should_share_validation_trajectory(config) is False


def test_different_amass_motion_sets_use_separate_trajectories():
    config = OmegaConf.create(
        {
            "experiment": {
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["motion_a", "motion_b"],
                        }
                    }
                },
                "validation": {
                    "active": True,
                    "amass_dataset_conf": {
                        "rel_dataset_path": ["motion_c"],
                    },
                },
            }
        }
    )

    assert should_share_validation_trajectory(config) is False


def test_non_amass_validation_dataset_uses_separate_trajectory():
    config = OmegaConf.create(
        {
            "experiment": {
                "task_factory": {
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["motion_a"],
                        }
                    }
                },
                "validation": {
                    "active": True,
                    "c3d_dataset_conf": {"rel_dataset_path": ["recording_a"]},
                },
            }
        }
    )

    assert should_share_validation_trajectory(config) is False


class FakeValidationEnv:
    def __init__(self, trajectory=None):
        self.th = None if trajectory is None else SimpleNamespace(traj=trajectory)
        self.load_calls = []

    def load_trajectory(self, **kwargs):
        self.load_calls.append(kwargs)
        self.th = SimpleNamespace(traj=kwargs["traj"])


def test_attach_uses_explicit_training_cache_layout():
    trajectory = object()
    training_env = SimpleNamespace(
        th=SimpleNamespace(
            traj=trajectory,
            cache_type=TrajectoryCacheType.SPARSE,
            cache_site_names=["left_foot", "right_foot"],
            motion_names=("motion-a",),
        )
    )
    validation_env = FakeValidationEnv()

    attach_shared_validation_trajectory(training_env, validation_env)

    assert validation_env.load_calls == [
        {
            "traj": trajectory,
            "warn": False,
            "cache_type": TrajectoryCacheType.SPARSE,
            "site_names": ["left_foot", "right_foot"],
            "materialized": True,
            "motion_names": ("motion-a",),
        }
    ]
    assert validation_env.th.traj is training_env.th.traj


def test_attach_replaces_existing_validation_trajectory_through_public_api():
    training_trajectory = object()
    validation_env = FakeValidationEnv(trajectory=object())
    training_env = SimpleNamespace(
        th=SimpleNamespace(
            traj=training_trajectory,
            cache_type=TrajectoryCacheType.FULL,
            cache_site_names=None,
            motion_names=None,
        )
    )

    attach_shared_validation_trajectory(training_env, validation_env)

    assert len(validation_env.load_calls) == 1
    assert validation_env.th.traj is training_trajectory


def test_attach_is_noop_when_trajectory_is_already_shared():
    trajectory = object()
    training_env = SimpleNamespace(
        th=SimpleNamespace(
            traj=trajectory,
            cache_type=TrajectoryCacheType.NONE,
            cache_site_names=None,
            motion_names=None,
        )
    )
    validation_env = FakeValidationEnv(trajectory=trajectory)

    attach_shared_validation_trajectory(training_env, validation_env)

    assert validation_env.load_calls == []


def test_attach_requires_training_trajectory_handler():
    with pytest.raises(ValueError, match="trajectory handler"):
        attach_shared_validation_trajectory(
            SimpleNamespace(th=None),
            FakeValidationEnv(),
        )
