"""Integration checks for full-body tracking goal registration and visualization."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from loco_mujoco.core.observations.base import Observation
from musclemimic.core.goals.bimanual_tracking import (
    GoalBimanualTracking,
    GoalBimanualTrackingVisual,
    bimanual_visual_goal_type,
)
from musclemimic.core.goals.fullbody_tracking import (
    GoalFullBodyTracking,
    GoalFullBodyTrackingVisual,
    fullbody_visual_goal_type,
)
from musclemimic.runner.validation_video_recorder import ValidationVideoRecorder


def test_fullbody_tracking_goals_are_registered() -> None:
    assert Observation.registered[GoalFullBodyTracking.__name__] is GoalFullBodyTracking
    assert Observation.registered[GoalFullBodyTrackingVisual.__name__] is GoalFullBodyTrackingVisual
    assert Observation.registered[GoalBimanualTracking.__name__] is GoalBimanualTracking
    assert Observation.registered[GoalBimanualTrackingVisual.__name__] is GoalBimanualTrackingVisual


def test_fullbody_visual_goal_preserves_tracking_encoding() -> None:
    assert fullbody_visual_goal_type("GoalFullBodyTracking") == "GoalFullBodyTrackingVisual"
    assert fullbody_visual_goal_type("GoalFullBodyTrackingVisual") == "GoalFullBodyTrackingVisual"
    assert fullbody_visual_goal_type("GoalTrajMimic") == "GoalTrajMimicv2"
    assert fullbody_visual_goal_type("GoalTrajMimicv2") == "GoalTrajMimicv2"


def test_fullbody_visual_goal_rejects_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="Unsupported full-body goal type"):
        fullbody_visual_goal_type("UnknownGoal")


def test_bimanual_visual_goal_preserves_tracking_encoding() -> None:
    assert bimanual_visual_goal_type("GoalBimanualTracking") == "GoalBimanualTrackingVisual"
    assert bimanual_visual_goal_type("GoalBimanualTrackingVisual") == "GoalBimanualTrackingVisual"
    assert bimanual_visual_goal_type("GoalBimanualTrajMimic") == "GoalBimanualTrajMimicv2"
    assert bimanual_visual_goal_type("GoalBimanualTrajMimicv2") == "GoalBimanualTrajMimicv2"


def test_validation_video_uses_tracking_visual_goal(tmp_path) -> None:
    config = OmegaConf.create(
        {
            "experiment": {
                "env_params": {
                    "env_name": "MjxMyoFullBody",
                    "goal_type": "GoalFullBodyTracking",
                    "goal_params": {"sites_for_mimic": ["pelvis_mimic"]},
                    "timestep": 0.002,
                    "n_substeps": 5,
                },
                "validation": {},
            }
        }
    )
    agent_conf = SimpleNamespace(config=config)
    recorder = ValidationVideoRecorder(str(tmp_path))

    env_params = recorder._build_env_params(agent_conf, tag="validation")

    assert env_params["env_name"] == "MyoFullBody"
    assert env_params["goal_type"] == "GoalFullBodyTrackingVisual"


def test_validation_video_uses_bimanual_tracking_visual_goal(tmp_path) -> None:
    config = OmegaConf.create(
        {
            "experiment": {
                "env_params": {
                    "env_name": "MjxMyoBimanualArm",
                    "goal_type": "GoalBimanualTracking",
                    "goal_params": {"sites_for_mimic": ["upper_body_mimic"]},
                    "timestep": 0.002,
                    "n_substeps": 5,
                },
                "validation": {},
            }
        }
    )
    agent_conf = SimpleNamespace(config=config)
    recorder = ValidationVideoRecorder(str(tmp_path))

    env_params = recorder._build_env_params(agent_conf, tag="validation")

    assert env_params["env_name"] == "MyoBimanualArm"
    assert env_params["goal_type"] == "GoalBimanualTrackingVisual"
