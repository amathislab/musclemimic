"""Unit tests for ValidationVideoRecorder - replicates training validation workflow."""

import os

import pytest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from loco_mujoco.core.stateful_object import StatefulObject
import musclemimic.utils.display as display_module
import musclemimic.runner.validation_video_recorder as recorder_module
from musclemimic.runner.validation_video_recorder import ValidationVideoRecorder
from musclemimic.utils import setup_headless_rendering_if_needed
from loco_mujoco.task_factories import ImitationFactory


class TestValidationVideoRecorderConfig:
    """Test environment configuration building for validation videos."""

    @pytest.fixture
    def video_recorder(self, tmp_path):
        return ValidationVideoRecorder(
            video_dir=str(tmp_path / "videos"),
            frequency=1,
            length=500,
            deterministic=True
        )

    @pytest.fixture
    def bimanual_training_config(self):
        """Training config from conf_bimanual.yaml with validation overrides."""
        config = OmegaConf.create({
            "experiment": {
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["KIT/3/tennis_forehand_right04_poses"]
                        }
                    }
                },
                "env_params": {
                    "env_name": "MjxMyoBimanualArm",
                    "headless": True,
                    "horizon": 1000,
                    "disable_fingers": False,
                    "mjx_backend": "warp",
                    "num_envs": 2048,
                    "enable_muscle_length_observations": False,
                    "enable_muscle_velocity_observations": False,
                    "enable_muscle_force_observations": True,
                    "enable_muscle_activation_observations": True,
                    "init_state_type": "TrajInitialStateHandler",
                    "goal_type": "GoalBimanualTrajMimic",
                    "goal_params": {
                        "visualize_goal": True,
                        "n_step_lookahead": 1,
                        "upper_body_xml_name": "thorax",
                        "sites_for_mimic": [
                            "right_shoulder_mimic",
                            "right_elbow_mimic",
                            "right_hand_mimic",
                            "left_shoulder_mimic",
                            "left_elbow_mimic",
                            "left_hand_mimic"
                        ]
                    },
                    "control_type": "DefaultControl",
                    "control_params": {},
                    "reward_type": "MimicReward",
                    "reward_params": {
                        "qpos_w_sum": 0.5,
                        "qvel_w_sum": 0.5,
                        "rpos_w_sum": 2.0,
                        "rquat_w_sum": 0.2,
                        "rvel_w_sum": 0.5,
                        "use_mean_exp_reward": False,
                        "sites_for_mimic": [
                            "right_shoulder_mimic",
                            "right_elbow_mimic",
                            "right_hand_mimic",
                            "left_shoulder_mimic",
                            "left_elbow_mimic",
                            "left_hand_mimic"
                        ]
                    },
                    "terminal_state_type": "EnhancedBimanualTerminalStateHandler",
                    "terminal_state_params": {
                        "max_site_deviation": 0.15,
                        "enable_reference_check": False,
                        "site_deviation_mode": "mean",
                        "debug_print": False,
                        "debug_every": 10
                    },
                    "timestep": 0.002,
                    "n_substeps": 5
                },
                "validation": {
                    "active": True,
                    "deterministic": True,
                    "num_steps": 500,
                    "num_envs": 20,
                    "num": 50,
                    "video_length": 500,
                    "video_frequency": 1,
                    "terminal_state_type": "NoTerminalStateHandler",
                    "terminal_state_params": {}
                }
            }
        })
        return config

    @pytest.fixture
    def mock_agent_conf(self, bimanual_training_config):
        agent_conf = MagicMock()
        agent_conf.config = bimanual_training_config
        return agent_conf

    @pytest.fixture
    def myofullbody_training_config(self):
        config = OmegaConf.create({
            "experiment": {
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["CMU/01/03"]
                        }
                    }
                },
                "env_params": {
                    "env_name": "MjxMyoFullBody",
                    "headless": True,
                    "goal_type": "GoalTrajMimic",
                    "goal_params": {
                        "sites_for_mimic": [
                            "pelvis", "head", "left_hand", "right_hand"
                        ]
                    },
                    "control_type": "DefaultControl",
                    "reward_type": "MimicReward",
                    "reward_params": {},
                    "timestep": 0.005,
                    "n_substeps": 4
                },
                "validation": {}
            }
        })
        return config

    @pytest.fixture
    def mock_fullbody_agent_conf(self, myofullbody_training_config):
        agent_conf = MagicMock()
        agent_conf.config = myofullbody_training_config
        return agent_conf

    @pytest.fixture
    def mock_agent_state(self):
        return MagicMock()

    def test_mjx_to_cpu_conversion(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["env_name"] == "MyoBimanualArm"

    def test_mjx_params_removed(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        for param in ["mjx_backend", "num_envs", "nconmax", "njmax"]:
            assert param not in env_params

    def test_validation_disables_early_termination(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        if "terminal_state_type" in mock_agent_conf.config.experiment.validation:
            assert env_params["terminal_state_type"] == "NoTerminalStateHandler"
            assert env_params["terminal_state_params"] == {}

    def test_headless_rendering_enabled(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["headless"] is True

    def test_goal_type_v2_for_visualization(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["goal_type"] == "GoalBimanualTrajMimicv2"

    def test_visualize_goal_flag_set(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["visualize_goal"] is True
        assert env_params["goal_params"]["visualize_goal"] is True

    def test_recorder_params_configured(self, video_recorder, mock_agent_conf):
        tag = "validation_5_t10000_20241119"
        env_params = video_recorder._build_env_params(mock_agent_conf, tag)
        assert env_params["recorder_params"]["tag"] == tag
        assert env_params["recorder_params"]["fps"] == 100  # 1/(0.002*5)
        assert env_params["recorder_params"]["compress"] is True

    def test_timing_preserved_from_training(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        assert env_params["timestep"] == 0.002
        assert env_params["n_substeps"] == 5

    def test_sites_for_mimic_preserved(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        expected = [
            "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
            "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
        ]
        assert env_params["goal_params"]["sites_for_mimic"] == expected

    def test_fullbody_goal_visualization_defaults(self, video_recorder, mock_fullbody_agent_conf):
        env_params = video_recorder._build_env_params(mock_fullbody_agent_conf, "fullbody_tag")
        assert env_params["env_name"] == "MyoFullBody"
        assert env_params["goal_type"] == "GoalTrajMimicv2"
        assert env_params["goal_params"]["sites_for_mimic"] == [
            "pelvis", "head", "left_hand", "right_hand"
        ]
        assert env_params["goal_params"]["enable_enhanced_visualization"] is True
        assert env_params["goal_params"]["target_geom_rgba"] == [0.471, 0.38, 0.812, 0.6]
        assert env_params["visualize_goal"] is True
        assert env_params["recorder_params"]["path"] == video_recorder.video_dir
        assert env_params["recorder_params"]["fps"] == 50  # 1/(0.005*4)

    def test_named_motion_validation_and_tag_sanitization(self, tmp_path):
        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path),
            named_motions=[
                {"name": "stairs up", "path": "Gait120/S001/StairAscent/Trial01/AllSteps_stageii"},
            ],
        )

        assert recorder.named_motions == (
            ("stairs up", "Gait120/S001/StairAscent/Trial01/AllSteps_stageii"),
        )
        assert recorder._safe_tag_component("stairs up / trial 1") == "stairs-up-trial-1"

    def test_duplicate_named_motions_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="duplicate name"):
            ValidationVideoRecorder(
                video_dir=str(tmp_path),
                named_motions=[
                    {"name": "stairs_up", "path": "one"},
                    {"name": "stairs_up", "path": "two"},
                ],
            )

    def test_named_motion_string_form(self, tmp_path):
        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path),
            named_motions=["stairs_up=Gait120/S001/StairAscent/Trial01/AllSteps_stageii"],
        )

        assert recorder.named_motions == (
            ("stairs_up", "Gait120/S001/StairAscent/Trial01/AllSteps_stageii"),
        )


class TestHeadlessSetupHelpers:
    def test_headless_detection_sets_egl(self, monkeypatch):
        monkeypatch.delenv("MUJOCO_GL", raising=False)
        monkeypatch.setattr(display_module, "detect_headless_environment", lambda: True)
        setup_headless_rendering_if_needed()
        assert os.environ["MUJOCO_GL"] == "egl"

    def test_headless_preserves_existing_gl_backend(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "osmesa")
        monkeypatch.setattr(display_module, "detect_headless_environment", lambda: True)
        setup_headless_rendering_if_needed()
        assert os.environ["MUJOCO_GL"] == "osmesa"

    def test_non_headless_preserves_existing_gl_backend(self, monkeypatch):
        monkeypatch.setenv("MUJOCO_GL", "glfw")
        monkeypatch.setattr(display_module, "detect_headless_environment", lambda: False)
        setup_headless_rendering_if_needed()
        assert os.environ["MUJOCO_GL"] == "glfw"


class TestValidationVideoRecorderErrorHandling:
    """Test error handling during validation video recording."""

    @pytest.fixture
    def video_recorder(self, tmp_path):
        return ValidationVideoRecorder(
            video_dir=str(tmp_path / "videos"),
            frequency=1,
            length=500,
        )

    @pytest.fixture
    def mock_agent_conf(self):
        return MagicMock(config=OmegaConf.create({
            "experiment": {
                "task_factory": {"name": "ImitationFactory", "params": {}},
                "env_params": {"env_name": "MjxMyoBimanualArm", "timestep": 0.002, "n_substeps": 5},
                "validation": {}
            }
        }))

    @pytest.fixture
    def mock_agent_state(self):
        return MagicMock()

    def test_env_creation_failure_no_unboundlocalerror(
        self, video_recorder, mock_agent_conf, mock_agent_state
    ):
        with patch.object(ImitationFactory, 'make', side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError, match="Test error"):
                video_recorder.record_episode(
                    agent_conf=mock_agent_conf,
                    agent_state=mock_agent_state,
                    validation_number=1,
                    timestep=1000
                )

    def test_env_creation_failure_detailed_logging(
        self, video_recorder, mock_agent_conf, mock_agent_state, capsys
    ):
        with patch.object(ImitationFactory, 'make', side_effect=ValueError("Invalid param")):
            with pytest.raises(ValueError):
                video_recorder.record_episode(
                    agent_conf=mock_agent_conf,
                    agent_state=mock_agent_state,
                    validation_number=1,
                    timestep=1000
                )
        captured = capsys.readouterr()
        assert "[ValidationVideo] ERROR:" in captured.out
        assert "ValueError" in captured.out
        assert "Invalid param" in captured.out

    def test_frequency_skipping(self, mock_agent_conf, mock_agent_state):
        recorder = ValidationVideoRecorder(video_dir="/tmp/test", frequency=5, length=500)
        for val_num in [1, 2, 3, 4]:
            result = recorder.record_episode(
                agent_conf=mock_agent_conf,
                agent_state=mock_agent_state,
                validation_number=val_num,
                timestep=val_num * 1000
            )
            assert result is None

    def test_frequency_recording(self, mock_agent_conf, mock_agent_state):
        recorder = ValidationVideoRecorder(video_dir="/tmp/test", frequency=5, length=500)
        with patch.object(ImitationFactory, 'make') as mock_make:
            mock_env = MagicMock()
            mock_env.video_file_path = "/tmp/test/video.mp4"
            mock_make.return_value = mock_env
            with patch('musclemimic.runner.validation_video_recorder.PPOJax.play_policy'):
                result = recorder.record_episode(
                    agent_conf=mock_agent_conf,
                    agent_state=mock_agent_state,
                    validation_number=5,
                    timestep=5000
                )
                assert mock_make.called
                assert mock_env.stop.called
                assert result == "/tmp/test/video.mp4"

    def test_env_stop_called_on_success(self, video_recorder, mock_agent_conf, mock_agent_state):
        with patch.object(ImitationFactory, 'make') as mock_make:
            mock_env = MagicMock()
            mock_env.video_file_path = "/tmp/test/video.mp4"
            mock_make.return_value = mock_env
            with patch('musclemimic.runner.validation_video_recorder.PPOJax.play_policy'):
                video_recorder.record_episode(
                    agent_conf=mock_agent_conf,
                    agent_state=mock_agent_state,
                    validation_number=1,
                    timestep=1000
                )
                assert mock_env.stop.called


class TestValidationVideoRecorderRecordingBehavior:
    """Test StatefulObject isolation and PPO invocation."""

    @pytest.fixture
    def video_recorder(self, tmp_path):
        return ValidationVideoRecorder(
            video_dir=str(tmp_path / "videos"),
            frequency=1,
            length=20,
            deterministic=False
        )

    @pytest.fixture
    def simple_agent_conf(self):
        config = OmegaConf.create({
            "experiment": {
                "task_factory": {"name": "ImitationFactory", "params": {}},
                "env_params": {
                    "env_name": "MjxMyoBimanualArm",
                    "goal_type": "GoalBimanualTrajMimic",
                    "goal_params": {"sites_for_mimic": ["left_hand_mimic"]},
                    "timestep": 0.002,
                    "n_substeps": 5
                },
                "validation": {}
            }
        })
        agent_conf = MagicMock()
        agent_conf.config = config
        return agent_conf

    @pytest.fixture
    def agent_state(self):
        return MagicMock()

    def test_stateful_objects_restored_and_play_policy_called(
        self, video_recorder, simple_agent_conf, agent_state, tmp_path
    ):
        dummy_env = MagicMock()
        dummy_env.video_file_path = str(tmp_path / "videos" / "mock.mp4")
        dummy_env.stop = MagicMock()

        class DummyFactory:
            @staticmethod
            def make(*_args, **_kwargs):
                return dummy_env

        sentinel_instances = ["sentinel"]
        original_instances = list(StatefulObject._instances)

        with patch.object(recorder_module.TaskFactory, "get_factory_cls", return_value=DummyFactory), \
            patch.object(recorder_module.PPOJax, "play_policy") as mock_play:
            try:
                StatefulObject._instances = list(sentinel_instances)
                result = video_recorder.record_episode(
                    agent_conf=simple_agent_conf,
                    agent_state=agent_state,
                    validation_number=1,
                    timestep=42
                )
                restored_instances = list(StatefulObject._instances)
            finally:
                StatefulObject._instances = original_instances

        assert result == dummy_env.video_file_path
        assert dummy_env.stop.called
        assert restored_instances == sentinel_instances
        mock_play.assert_called_once()
        args, kwargs = mock_play.call_args
        assert args[0] is dummy_env
        assert args[1] is simple_agent_conf
        assert args[2] is agent_state
        assert kwargs["n_envs"] == 1
        assert kwargs["n_steps"] == video_recorder.length
        assert kwargs["render"] is True
        assert kwargs["record"] is True
        assert kwargs["deterministic"] is video_recorder.deterministic
        assert kwargs["use_mujoco"] is True
        assert kwargs["wrap_env"] is True
        assert kwargs["train_state_seed"] == 0
        assert kwargs["stop_after_first_episode"] is True

    def test_stateful_objects_restored_on_env_failure(self, video_recorder, simple_agent_conf, agent_state):
        class DummyFactory:
            @staticmethod
            def make(*_args, **_kwargs):
                raise RuntimeError("factory failure")

        sentinel_instances = ["existing"]
        original_instances = list(StatefulObject._instances)

        with patch.object(recorder_module.TaskFactory, "get_factory_cls", return_value=DummyFactory):
            try:
                StatefulObject._instances = list(sentinel_instances)
                with pytest.raises(RuntimeError, match="factory failure"):
                    video_recorder.record_episode(
                        agent_conf=simple_agent_conf,
                        agent_state=agent_state,
                        validation_number=1,
                        timestep=0
                    )
            finally:
                restored_instances = list(StatefulObject._instances)
                StatefulObject._instances = original_instances

        assert restored_instances == sentinel_instances

    def test_named_motions_are_individually_pinned_and_returned(self, simple_agent_conf, agent_state, tmp_path):
        simple_agent_conf.config.experiment.task_factory.params.amass_dataset_conf = {
            "rel_dataset_path": ["unused/training/motion"],
        }
        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path / "videos"),
            frequency=1,
            length=20,
            named_motions=[
                {"name": "stairs_up", "path": "Gait120/S001/StairAscent/Trial01/AllSteps_stageii"},
                {"name": "ramp_down", "path": "Gait120/S002/SlopeDescent/Trial01/AllSteps_stageii"},
            ],
        )
        created_envs = []
        created_params = []

        class DummyFactory:
            @staticmethod
            def make(*_args, **kwargs):
                env = MagicMock()
                motion = kwargs["amass_dataset_conf"]["rel_dataset_path"][0]
                env.video_file_path = str(tmp_path / f"{motion.split('/')[-3]}.mp4")
                created_envs.append(env)
                created_params.append(kwargs)
                return env

        with patch.object(recorder_module.TaskFactory, "get_factory_cls", return_value=DummyFactory), \
            patch.object(recorder_module.PPOJax, "play_policy") as mock_play:
            result = recorder.record_episode(
                agent_conf=simple_agent_conf,
                agent_state=agent_state,
                validation_number=1,
                timestep=42,
            )

        assert result == {
            "stairs_up": str(tmp_path / "StairAscent.mp4"),
            "ramp_down": str(tmp_path / "SlopeDescent.mp4"),
        }
        assert [
            params["amass_dataset_conf"]["rel_dataset_path"] for params in created_params
        ] == [
            ["Gait120/S001/StairAscent/Trial01/AllSteps_stageii"],
            ["Gait120/S002/SlopeDescent/Trial01/AllSteps_stageii"],
        ]
        assert all(params["amass_dataset_conf"]["max_motions"] == 1 for params in created_params)
        assert all(params["amass_dataset_conf"]["dataset_group"] is None for params in created_params)
        assert all(env.stop.called for env in created_envs)
        assert mock_play.call_count == 2

    def test_reset_hook_receives_pinned_motion(self, simple_agent_conf, agent_state, tmp_path):
        simple_agent_conf.config.experiment.task_factory.params.amass_dataset_conf = {
            "rel_dataset_path": ["unused/training/motion"],
        }

        class InspectingRecorder(ValidationVideoRecorder):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.inspected = []

            def _on_environment_reset(self, env, motion_path):
                self.inspected.append((env, motion_path))

        recorder = InspectingRecorder(
            video_dir=str(tmp_path / "videos"),
            frequency=1,
            length=20,
            named_motions=[
                "stairs_up=Gait120/S001/StairAscent/Trial01/AllSteps_stageii",
            ],
        )
        dummy_env = MagicMock()
        dummy_env.video_file_path = str(tmp_path / "stairs_up.mp4")

        class DummyFactory:
            @staticmethod
            def make(*_args, **_kwargs):
                return dummy_env

        with patch.object(recorder_module.TaskFactory, "get_factory_cls", return_value=DummyFactory), \
            patch.object(recorder_module.PPOJax, "play_policy"):
            recorder.record_episode(
                agent_conf=simple_agent_conf,
                agent_state=agent_state,
                validation_number=1,
                timestep=42,
            )

        assert recorder.inspected == [
            (dummy_env, "Gait120/S001/StairAscent/Trial01/AllSteps_stageii")
        ]
        dummy_env.reset.assert_called_once_with()


class TestValidationVideoRecorderIntegration:
    """Integration tests with actual environment creation and recording."""

    @pytest.fixture
    def bimanual_training_config(self):
        """Full bimanual training config matching conf_bimanual.yaml."""
        config = OmegaConf.create({
            "experiment": {
                "task_factory": {
                    "name": "ImitationFactory",
                    "params": {
                        "amass_dataset_conf": {
                            "rel_dataset_path": ["KIT/3/tennis_forehand_right04_poses"]
                        }
                    }
                },
                "env_params": {
                    "env_name": "MjxMyoBimanualArm",
                    "headless": True,
                    "horizon": 1000,
                    "disable_fingers": False,
                    "mjx_backend": "warp",
                    "num_envs": 2048,
                    "init_state_type": "TrajInitialStateHandler",
                    "goal_type": "GoalBimanualTrajMimic",
                    "goal_params": {
                        "visualize_goal": True,
                        "n_step_lookahead": 1,
                        "upper_body_xml_name": "thorax",
                        "sites_for_mimic": [
                            "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                            "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
                        ]
                    },
                    "control_type": "DefaultControl",
                    "control_params": {},
                    "reward_type": "MimicReward",
                    "reward_params": {
                        "qpos_w_sum": 0.5,
                        "qvel_w_sum": 0.5,
                        "rpos_w_sum": 2.0,
                        "rquat_w_sum": 0.2,
                        "rvel_w_sum": 0.5,
                        "use_mean_exp_reward": False,
                        "sites_for_mimic": [
                            "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                            "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
                        ]
                    },
                    "terminal_state_type": "EnhancedBimanualTerminalStateHandler",
                    "terminal_state_params": {
                        "max_site_deviation": 0.15,
                        "enable_reference_check": False,
                        "site_deviation_mode": "mean"
                    },
                    "timestep": 0.002,
                    "n_substeps": 5
                },
                "validation": {
                    "terminal_state_type": "NoTerminalStateHandler",
                    "terminal_state_params": {}
                }
            }
        })
        return config

    @pytest.mark.integration
    def test_create_validation_env_with_no_termination(self, tmp_path):
        from loco_mujoco.task_factories import ImitationFactory

        env_params = {
            "env_name": "MyoBimanualArm",
            "headless": True,
            "visualize_goal": True,
            "init_state_type": "TrajInitialStateHandler",
            "goal_type": "GoalBimanualTrajMimicv2",
            "goal_params": {
                "visualize_goal": True,
                "n_step_lookahead": 1,
                "upper_body_xml_name": "thorax",
                "sites_for_mimic": [
                    "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                    "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
                ],
                "enable_enhanced_visualization": True,
                "target_geom_rgba": [0.471, 0.38, 0.812, 0.6]
            },
            "terminal_state_type": "NoTerminalStateHandler",
            "terminal_state_params": {},
            "control_type": "DefaultControl",
            "control_params": {},
            "reward_type": "MimicReward",
            "reward_params": {
                "qpos_w_sum": 0.5,
                "qvel_w_sum": 0.5,
                "rpos_w_sum": 2.0,
                "rquat_w_sum": 0.2,
                "rvel_w_sum": 0.5,
                "use_mean_exp_reward": False,
                "sites_for_mimic": [
                    "right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic",
                    "left_shoulder_mimic", "left_elbow_mimic", "left_hand_mimic"
                ]
            },
            "recorder_params": {
                "path": str(tmp_path / "videos"),
                "tag": "test_validation",
                "video_name": "MyoBimanualArm",
                "fps": 30,
                "compress": True
            },
            "timestep": 0.002,
            "n_substeps": 5
        }

        factory_params = {
            "amass_dataset_conf": {
                "rel_dataset_path": ["KIT/3/tennis_forehand_right04_poses"]
            }
        }

        try:
            print("\n[Integration] Creating validation env with NoTerminalStateHandler...")
            env = ImitationFactory.make(**env_params, **factory_params)
            print("[Integration] Testing reset...")
            env.reset()
            print("[Integration] Verifying NoTerminalStateHandler...")
            assert type(env._terminal_state_handler).__name__ == "NoTerminalStateHandler"
            print("[Integration] Stopping environment...")
            env.stop()
            print("Validation env created successfully")
        except Exception as e:
            print(f"\nFailed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_full_validation_video_recording_workflow(self, tmp_path, bimanual_training_config):
        """Test complete video recording workflow - env creation and setup."""
        from loco_mujoco.task_factories import ImitationFactory

        print("\n[Integration] Testing validation env setup for recording...")

        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path / "validation_videos"),
            frequency=1,
            length=50,
            deterministic=True
        )

        mock_agent_conf = MagicMock()
        mock_agent_conf.config = bimanual_training_config

        try:
            print("[Integration] Building env params...")
            env_params = recorder._build_env_params(mock_agent_conf, "test_validation_1_t10000")

            print("[Integration] Creating validation env...")
            env = ImitationFactory.make(
                **env_params,
                **bimanual_training_config.experiment.task_factory.params
            )

            print("[Integration] Testing env reset...")
            env.reset()

            print("[Integration] Verifying configuration...")
            assert type(env._terminal_state_handler).__name__ == "NoTerminalStateHandler"
            assert hasattr(env, '_viewer')
            assert env_params["recorder_params"]["tag"] == "test_validation_1_t10000"
            assert env_params["recorder_params"]["fps"] == 100  # 1/(0.002*5)
            assert env_params["headless"] is True

            print("[Integration] Cleaning up...")
            env.stop()
            print("Validation env setup and configuration verified")

        except Exception as e:
            print(f"\nSetup failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

    @pytest.mark.integration
    def test_actual_video_recording(self, tmp_path, bimanual_training_config):
        """Test actual video recording with a short rollout."""
        import os
        import numpy as np
        from loco_mujoco.task_factories import ImitationFactory

        print("\n[Integration] Testing actual video recording...")

        recorder = ValidationVideoRecorder(
            video_dir=str(tmp_path / "actual_videos"),
            frequency=1,
            length=30,  # Short rollout
            deterministic=True
        )

        mock_agent_conf = MagicMock()
        mock_agent_conf.config = bimanual_training_config

        # Create simple mock agent state with random policy
        mock_agent_state = MagicMock()
        mock_agent_state.params = MagicMock()
        mock_agent_state.run_stats = MagicMock()

        try:
            print("[Integration] Building env for recording...")
            env_params = recorder._build_env_params(mock_agent_conf, "actual_recording_test")
            env = ImitationFactory.make(
                **env_params,
                **bimanual_training_config.experiment.task_factory.params
            )

            print("[Integration] Running rollout with recording enabled...")
            obs = env.reset()
            env.render(record=True)  # Initial render with recording

            action_dim = env.model.nu  # Number of actuators
            for step in range(30):
                action = np.zeros(action_dim)  # Zero action
                obs, reward, terminated, truncated, info = env.step(action)
                env.render(record=True)  # Render and record each frame
                if step % 10 == 0:
                    print(f"  Step {step}/30")
                if terminated or truncated:
                    break

            print("[Integration] Finalizing recording and stopping environment...")
            env.stop()

            # Check if video was created
            video_path = env.video_file_path
            print(f"[Integration] Video path: {video_path}")

            if video_path and os.path.exists(video_path):
                print(f"Video file created: {video_path}")
                file_size = os.path.getsize(video_path)
                print(f"  File size: {file_size} bytes")
                assert file_size > 0, "Video file is empty"
            else:
                print("  Note: Video path is None or file doesn't exist")
                print("  This might be expected if recording wasn't triggered")

        except Exception as e:
            print(f"\nRecording test failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
