"""Unit tests for ValidationVideoRecorder - replicates training validation workflow."""

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

import musclemimic.runner.validation_video_recorder as recorder_module
import musclemimic.utils.display as display_module
from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.task_factories import ImitationFactory
from musclemimic.runner.logging import ExperimentHooks
from musclemimic.runner.validation_video_checkpoint import ValidationPolicyCheckpoint
from musclemimic.runner.validation_video_recorder import (
    AsyncValidationVideoRecorder,
    FailedValidationVideo,
    ValidationVideoRecorder,
)
from musclemimic.utils import setup_headless_rendering_if_needed

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _checkpoint_path(tmp_path, update: int = 1) -> str:
    return str(tmp_path / "checkpoints" / f"checkpoint_{update}")


class TestValidationVideoRecorderHooks:
    """Test hook-level validation video enable/disable behavior."""

    @pytest.mark.parametrize("video_frequency", [0, -1])
    def test_video_frequency_non_positive_disables_recorder(self, tmp_path, video_frequency):
        config = OmegaConf.create(
            {
                "experiment": {
                    "validation": {
                        "active": True,
                        "video_frequency": video_frequency,
                        "video_length": 500,
                    }
                }
            }
        )

        recorder = ExperimentHooks().build_video_recorder(str(tmp_path), config)

        assert recorder is None

    def test_positive_video_frequency_builds_async_recorder_by_default(self, tmp_path):
        config = OmegaConf.create(
            {
                "experiment": {
                    "save_checkpoints": True,
                    "checkpoints_on_validation": True,
                    "validation": {
                        "active": True,
                        "video_frequency": 1,
                        "video_length": 500,
                    },
                }
            }
        )

        recorder = ExperimentHooks().build_video_recorder(str(tmp_path), config)

        assert isinstance(recorder, AsyncValidationVideoRecorder)
        assert recorder.length == 500
        assert recorder.shutdown_timeout_s is None

    @pytest.mark.parametrize(
        ("config_path", "async_checkpointing"),
        [
            ("fullbody/conf_fullbody.yaml", False),
            ("bimanual/conf_bimanual.yaml", True),
        ],
    )
    def test_base_experiment_configs_support_checkpoint_backed_video(
        self,
        tmp_path,
        config_path,
        async_checkpointing,
    ):
        config = OmegaConf.load(REPOSITORY_ROOT / config_path)

        recorder = ExperimentHooks().build_video_recorder(str(tmp_path), config)

        assert isinstance(recorder, AsyncValidationVideoRecorder)
        assert config.experiment.async_checkpointing is async_checkpointing

    def test_video_recorder_requires_saved_checkpoints(
        self,
        tmp_path,
    ):
        config = OmegaConf.create(
            {
                "experiment": {
                    "save_checkpoints": False,
                    "checkpoints_on_validation": True,
                    "validation": {
                        "active": True,
                        "video_frequency": 1,
                        "video_length": 500,
                    },
                }
            }
        )

        with pytest.raises(ValueError, match="requires save_checkpoints"):
            ExperimentHooks().build_video_recorder(str(tmp_path), config)

    def test_video_recorder_does_not_require_every_validation_to_checkpoint(self, tmp_path):
        config = OmegaConf.create(
            {
                "experiment": {
                    "save_checkpoints": True,
                    "checkpoints_on_validation": False,
                    "validation": {
                        "active": True,
                        "video_frequency": 10,
                        "video_length": 500,
                    },
                }
            }
        )

        recorder = ExperimentHooks().build_video_recorder(str(tmp_path), config)

        assert isinstance(recorder, AsyncValidationVideoRecorder)


class TestValidationVideoRecorderConfig:
    """Test environment configuration building for validation videos."""

    @pytest.fixture
    def video_recorder(self, tmp_path):
        return ValidationVideoRecorder(video_dir=str(tmp_path / "videos"), length=500, deterministic=True)

    @pytest.fixture
    def bimanual_training_config(self):
        """Training config from conf_bimanual.yaml with validation overrides."""
        config = OmegaConf.create(
            {
                "experiment": {
                    "task_factory": {
                        "name": "ImitationFactory",
                        "params": {"amass_dataset_conf": {"rel_dataset_path": ["KIT/3/tennis_forehand_right04_poses"]}},
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
                                "left_hand_mimic",
                            ],
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
                                "left_hand_mimic",
                            ],
                        },
                        "terminal_state_type": "EnhancedBimanualTerminalStateHandler",
                        "terminal_state_params": {
                            "max_site_deviation": 0.15,
                            "enable_reference_check": False,
                            "site_deviation_mode": "mean",
                            "debug_print": False,
                            "debug_every": 10,
                        },
                        "timestep": 0.002,
                        "n_substeps": 5,
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
                        "terminal_state_params": {},
                    },
                }
            }
        )
        return config

    @pytest.fixture
    def mock_agent_conf(self, bimanual_training_config):
        agent_conf = MagicMock()
        agent_conf.config = bimanual_training_config
        return agent_conf

    @pytest.fixture
    def myofullbody_training_config(self):
        config = OmegaConf.create(
            {
                "experiment": {
                    "task_factory": {
                        "name": "ImitationFactory",
                        "params": {"amass_dataset_conf": {"rel_dataset_path": ["CMU/01/03"]}},
                    },
                    "env_params": {
                        "env_name": "MjxMyoFullBody",
                        "headless": True,
                        "goal_type": "GoalTrajMimic",
                        "goal_params": {"sites_for_mimic": ["pelvis", "head", "left_hand", "right_hand"]},
                        "control_type": "DefaultControl",
                        "reward_type": "MimicReward",
                        "reward_params": {},
                        "timestep": 0.005,
                        "n_substeps": 4,
                    },
                    "validation": {},
                }
            }
        )
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
        mock_agent_conf.config.experiment.env_params.naconmax = 128
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        for param in ["mjx_backend", "num_envs", "nconmax", "naconmax", "njmax"]:
            assert param not in env_params

    def test_validation_disables_early_termination(self, video_recorder, mock_agent_conf):
        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")
        if "terminal_state_type" in mock_agent_conf.config.experiment.validation:
            assert env_params["terminal_state_type"] == "NoTerminalStateHandler"
            assert env_params["terminal_state_params"] == {}

    def test_validation_reward_termination_overrides_training_values(self, video_recorder, mock_agent_conf):
        mock_agent_conf.config.experiment.env_params.reward_termination_type = "LowRewardTerminationHandler"
        mock_agent_conf.config.experiment.env_params.reward_termination_params = {"threshold": 0.968, "patience": 100}
        mock_agent_conf.config.experiment.validation.reward_termination_type = "NoRewardTerminationHandler"
        mock_agent_conf.config.experiment.validation.reward_termination_params = {}

        env_params = video_recorder._build_env_params(mock_agent_conf, "test_tag")

        assert env_params["reward_params"]["qpos_w_sum"] == 0.5
        assert env_params["reward_termination_type"] == "NoRewardTerminationHandler"
        assert env_params["reward_termination_params"] == {}

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
            "right_shoulder_mimic",
            "right_elbow_mimic",
            "right_hand_mimic",
            "left_shoulder_mimic",
            "left_elbow_mimic",
            "left_hand_mimic",
        ]
        assert env_params["goal_params"]["sites_for_mimic"] == expected

    def test_fullbody_goal_visualization_defaults(self, video_recorder, mock_fullbody_agent_conf):
        env_params = video_recorder._build_env_params(mock_fullbody_agent_conf, "fullbody_tag")
        assert env_params["env_name"] == "MyoFullBody"
        assert env_params["goal_type"] == "GoalTrajMimicv2"
        assert env_params["goal_params"]["sites_for_mimic"] == ["pelvis", "head", "left_hand", "right_hand"]
        assert env_params["goal_params"]["enable_enhanced_visualization"] is True
        assert env_params["goal_params"]["target_geom_rgba"] == [0.471, 0.38, 0.812, 0.6]
        assert env_params["visualize_goal"] is True
        assert env_params["recorder_params"]["path"] == video_recorder.video_dir
        assert env_params["recorder_params"]["fps"] == 50  # 1/(0.005*4)

    def test_task_params_cap_video_motions(self, video_recorder, mock_agent_conf):
        task_params = video_recorder._build_task_params(
            mock_agent_conf,
            validation_number=4,
        )

        assert task_params["amass_dataset_conf"]["max_motions"] == 3
        assert task_params["amass_dataset_conf"]["max_motions_seed"] > 0
        assert task_params["amass_dataset_conf"]["write_materialized_cache"] is False

    def test_task_params_change_video_subset_by_validation_number(
        self,
        video_recorder,
        mock_agent_conf,
    ):
        mock_agent_conf.config.experiment.validation.eval_seed = 19

        first = video_recorder._build_task_params(
            mock_agent_conf,
            validation_number=2,
        )
        repeated = video_recorder._build_task_params(
            mock_agent_conf,
            validation_number=2,
        )
        later = video_recorder._build_task_params(
            mock_agent_conf,
            validation_number=3,
        )

        assert first["amass_dataset_conf"]["max_motions_seed"] == repeated["amass_dataset_conf"]["max_motions_seed"]
        assert first["amass_dataset_conf"]["max_motions_seed"] != later["amass_dataset_conf"]["max_motions_seed"]

    def test_task_params_use_validation_amass_conf_for_video(self, video_recorder, mock_agent_conf):
        mock_agent_conf.config.experiment.validation.amass_dataset_conf = {
            "dataset_group": "KIT_KINESIS_TESTING_MOTIONS",
            "trajectory_cache_type": "sparse",
        }
        mock_agent_conf.config.experiment.validation.video_max_motions = 2

        task_params = video_recorder._build_task_params(mock_agent_conf)

        assert task_params["amass_dataset_conf"] == {
            "dataset_group": "KIT_KINESIS_TESTING_MOTIONS",
            "trajectory_cache_type": "sparse",
            "max_motions": 2,
            "max_motions_seed": task_params["amass_dataset_conf"]["max_motions_seed"],
            "write_materialized_cache": False,
        }

    @pytest.mark.parametrize(
        "dataset_key",
        ["c3d_dataset_conf", "lafan1_dataset_conf", "custom_dataset_conf"],
    )
    def test_task_params_replace_training_dataset_for_video(
        self,
        video_recorder,
        mock_agent_conf,
        dataset_key,
    ):
        validation_dataset = {"rel_dataset_path": ["validation_motion"]}
        mock_agent_conf.config.experiment.validation[dataset_key] = validation_dataset

        task_params = video_recorder._build_task_params(mock_agent_conf)

        assert "amass_dataset_conf" not in task_params
        assert task_params[dataset_key] == validation_dataset


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

    def test_validation_worker_overrides_interactive_gl_backend(self):
        env = {"MUJOCO_GL": "glfw"}

        worker_env = recorder_module.configure_validation_video_worker_env(env)

        assert worker_env["MUJOCO_GL"] == "egl"


class TestValidationVideoRecorderErrorHandling:
    """Test error handling when building the recording environment."""

    @pytest.fixture
    def video_recorder(self, tmp_path):
        return ValidationVideoRecorder(
            video_dir=str(tmp_path / "videos"),
            length=500,
        )

    @pytest.fixture
    def mock_agent_conf(self):
        return MagicMock(
            config=OmegaConf.create(
                {
                    "experiment": {
                        "task_factory": {"name": "ImitationFactory", "params": {}},
                        "env_params": {"env_name": "MjxMyoBimanualArm", "timestep": 0.002, "n_substeps": 5},
                        "validation": {},
                    }
                }
            )
        )

    def test_env_creation_failure_propagates(self, video_recorder, mock_agent_conf):
        with patch.object(ImitationFactory, "make", side_effect=RuntimeError("Test error")):
            with pytest.raises(RuntimeError, match="Test error"):
                video_recorder.make_recording_env(mock_agent_conf, "test_tag")

    def test_env_creation_failure_detailed_logging(self, video_recorder, mock_agent_conf, caplog):
        caplog.set_level(logging.ERROR, logger=recorder_module.__name__)
        with patch.object(ImitationFactory, "make", side_effect=ValueError("Invalid param")):
            with pytest.raises(ValueError):
                video_recorder.make_recording_env(mock_agent_conf, "test_tag")
        assert "Failed to create validation recording environment" in caplog.text
        assert "ValueError" in caplog.text
        assert "Invalid param" in caplog.text


class TestValidationVideoRecorderRecordingBehavior:
    """Test rollout invocation and the async worker lifecycle."""

    @pytest.fixture
    def video_recorder(self, tmp_path):
        return ValidationVideoRecorder(video_dir=str(tmp_path / "videos"), length=20, deterministic=False)

    @pytest.fixture
    def simple_agent_conf(self):
        config = OmegaConf.create(
            {
                "experiment": {
                    "task_factory": {"name": "ImitationFactory", "params": {}},
                    "env_params": {
                        "env_name": "MjxMyoBimanualArm",
                        "goal_type": "GoalBimanualTrajMimic",
                        "goal_params": {"sites_for_mimic": ["left_hand_mimic"]},
                        "timestep": 0.002,
                        "n_substeps": 5,
                    },
                    "validation": {},
                }
            }
        )
        agent_conf = MagicMock()
        agent_conf.config = config
        return agent_conf

    @pytest.fixture
    def agent_state(self):
        return MagicMock()

    def test_run_recording_rollout_invokes_play_policy(self, video_recorder, simple_agent_conf, agent_state):
        dummy_env = MagicMock()
        env_params = {"recorder_params": {"fps": 100}}
        algorithm_cls = MagicMock()

        video_recorder.run_recording_rollout(
            algorithm_cls,
            dummy_env,
            simple_agent_conf,
            agent_state,
            env_params,
        )

        mock_play = algorithm_cls.play_policy
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

    def test_async_recorder_schedules_cpu_worker(self, tmp_path):
        recorder = AsyncValidationVideoRecorder(
            video_dir=str(tmp_path),
            frequency=1,
            length=50,
            deterministic=True,
        )
        with patch.object(recorder_module.os, "posix_spawn", return_value=1234) as posix_spawn:
            result = recorder.record_episode(
                _checkpoint_path(tmp_path, 7),
                validation_number=1,
                timestep=42,
            )

        assert result is None
        assert posix_spawn.call_count == 1
        executable, cmd, env = posix_spawn.call_args.args
        file_actions = posix_spawn.call_args.kwargs["file_actions"]
        payload_path = cmd[-1]
        assert executable == recorder_module.sys.executable
        assert cmd[:3] == [recorder_module.sys.executable, "-m", "musclemimic.runner.validation_video_worker"]
        assert env["JAX_PLATFORMS"] == "cpu"
        assert env["JAX_PLATFORM_NAME"] == "cpu"
        assert env["MUJOCO_GL"] == "egl"
        assert file_actions[0][0] == recorder_module.os.POSIX_SPAWN_OPEN
        assert file_actions[1][0] == recorder_module.os.POSIX_SPAWN_DUP2

        with open(payload_path) as f:
            payload = json.load(f)
        assert set(payload) == {
            "checkpoint_path",
            "checkpoint_wait_timeout_s",
            "video_dir",
            "length",
            "deterministic",
            "validation_number",
            "timestep",
            "tag",
            "result_path",
        }
        assert payload["length"] == 50
        assert payload["timestep"] == 42
        assert payload["checkpoint_path"] == str((tmp_path / "checkpoints" / "checkpoint_7").resolve())

    def test_async_recorder_collects_finished_video(self, tmp_path):
        recorder = AsyncValidationVideoRecorder(video_dir=str(tmp_path), frequency=1, length=50)

        with patch.object(recorder_module.os, "posix_spawn", return_value=1234):
            recorder.record_episode(_checkpoint_path(tmp_path, 2), validation_number=2, timestep=84)

        job = recorder._jobs[0]
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"mp4")
        with open(job.result_path, "w") as f:
            json.dump({"video_path": str(video_path)}, f)
        job.log_path.write_text("worker log")

        with patch.object(recorder_module.os, "waitpid", return_value=(1234, 0)):
            completed = recorder.collect_finished()

        assert len(completed) == 1
        assert completed[0].video_path == str(video_path)
        assert completed[0].validation_number == 2
        assert completed[0].timestep == 84
        assert recorder._jobs == []
        assert not job.payload_path.exists()
        assert not job.result_path.exists()
        assert not job.log_path.exists()

    def test_async_recorder_reports_nonzero_worker_error_and_keeps_log(self, tmp_path):
        recorder = AsyncValidationVideoRecorder(video_dir=str(tmp_path), frequency=1, length=50)

        with patch.object(recorder_module.os, "posix_spawn", return_value=1234):
            recorder.record_episode(_checkpoint_path(tmp_path, 3), validation_number=3, timestep=126)

        job = recorder._jobs[0]
        with open(job.result_path, "w") as f:
            json.dump({"ok": False, "error": "RuntimeError: render failed"}, f)
        job.log_path.write_text("traceback")

        status = 1 << 8
        with patch.object(recorder_module.os, "waitpid", return_value=(1234, status)):
            completed = recorder.collect_finished()

        assert len(completed) == 1
        assert isinstance(completed[0], FailedValidationVideo)
        assert completed[0].error == "RuntimeError: render failed"
        assert completed[0].return_code == 1
        assert recorder._jobs == []
        assert not job.payload_path.exists()
        assert not job.result_path.exists()
        # The worker log is the only diagnostics for the failure; it must survive.
        assert job.log_path.exists()
        assert completed[0].log_path == str(job.log_path)

    def test_async_recorder_skips_while_previous_job_is_running(self, tmp_path):
        recorder = AsyncValidationVideoRecorder(video_dir=str(tmp_path), frequency=1, length=50)

        with (
            patch.object(recorder_module.os, "posix_spawn", return_value=1234) as posix_spawn,
            patch.object(recorder_module.os, "waitpid", return_value=(0, 0)),
        ):
            recorder.record_episode(_checkpoint_path(tmp_path, 1), validation_number=1, timestep=42)
            recorder.record_episode(_checkpoint_path(tmp_path, 2), validation_number=2, timestep=84)

        assert posix_spawn.call_count == 1
        assert len(recorder._jobs) == 1

    def test_async_recorder_respects_frequency(self, tmp_path):
        recorder = AsyncValidationVideoRecorder(video_dir=str(tmp_path), frequency=5, length=50)

        with patch.object(recorder_module.os, "posix_spawn", return_value=1234) as posix_spawn:
            for validation_number in (1, 2, 3, 4):
                recorder.record_episode(
                    _checkpoint_path(tmp_path, validation_number),
                    validation_number=validation_number,
                    timestep=42,
                )
            assert posix_spawn.call_count == 0
            recorder.record_episode(_checkpoint_path(tmp_path, 5), validation_number=5, timestep=42)
            assert posix_spawn.call_count == 1

    def test_wait_for_all_returns_results_when_job_finishes(self, tmp_path):
        recorder = AsyncValidationVideoRecorder(video_dir=str(tmp_path), frequency=1, length=50)

        with patch.object(recorder_module.os, "posix_spawn", return_value=1234):
            recorder.record_episode(_checkpoint_path(tmp_path), validation_number=1, timestep=42)

        job = recorder._jobs[0]
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"mp4")
        with open(job.result_path, "w") as f:
            json.dump({"video_path": str(video_path)}, f)

        # First poll: still running; second poll: exited cleanly.
        waitpid_results = iter([(0, 0), (1234, 0)])
        with patch.object(recorder_module.os, "waitpid", side_effect=lambda *_: next(waitpid_results)):
            results = recorder.wait_for_all(timeout_s=5.0, poll_interval_s=0.01)

        assert len(results) == 1
        assert results[0].video_path == str(video_path)
        assert recorder._jobs == []

    def test_wait_for_all_without_timeout_waits_until_job_finishes(self, tmp_path):
        recorder = AsyncValidationVideoRecorder(video_dir=str(tmp_path), frequency=1, length=3000)

        with patch.object(recorder_module.os, "posix_spawn", return_value=1234):
            recorder.record_episode(_checkpoint_path(tmp_path), validation_number=1, timestep=42)

        job = recorder._jobs[0]
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"mp4")
        with open(job.result_path, "w") as f:
            json.dump({"video_path": str(video_path)}, f)

        waitpid_results = iter([(0, 0), (1234, 0)])
        with patch.object(recorder_module.os, "waitpid", side_effect=lambda *_: next(waitpid_results)):
            results = recorder.wait_for_all(poll_interval_s=0.01)

        assert len(results) == 1
        assert results[0].video_path == str(video_path)
        assert recorder._jobs == []

    def test_wait_for_all_times_out_on_never_finishing_job(self, tmp_path):
        recorder = AsyncValidationVideoRecorder(video_dir=str(tmp_path), frequency=1, length=50)

        with patch.object(recorder_module.os, "posix_spawn", return_value=1234):
            recorder.record_episode(_checkpoint_path(tmp_path), validation_number=1, timestep=42)

        with patch.object(recorder_module.os, "waitpid", return_value=(0, 0)):
            results = recorder.wait_for_all(timeout_s=0.05, poll_interval_s=0.01)

        assert results == []
        assert len(recorder._jobs) == 1


class TestValidationVideoWorker:
    """Test the subprocess entrypoint end-to-end with a stubbed env."""

    def _make_payload(self, tmp_path):
        result_path = tmp_path / "job.result.json"
        payload = {
            "checkpoint_path": _checkpoint_path(tmp_path),
            "checkpoint_wait_timeout_s": 5.0,
            "video_dir": str(tmp_path),
            "length": 5,
            "deterministic": True,
            "validation_number": 7,
            "timestep": 700,
            "tag": "validation_7_t700_test",
            "result_path": str(result_path),
        }
        payload_path = tmp_path / "job.payload.json"
        with open(payload_path, "w") as f:
            json.dump(payload, f)
        return payload_path, result_path

    def test_worker_requires_cpu_jax_env(self, tmp_path, monkeypatch):
        from musclemimic.runner import validation_video_worker

        payload_path, _ = self._make_payload(tmp_path)
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)

        assert validation_video_worker.main([str(payload_path)]) == 2

    def test_worker_writes_success_result_and_restores_stateful_objects(self, tmp_path, monkeypatch):
        from musclemimic.runner import validation_video_worker

        payload_path, result_path = self._make_payload(tmp_path)
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"mp4")
        dummy_env = MagicMock()
        dummy_env.video_file_path = str(video_path)
        restored = ValidationPolicyCheckpoint(
            config={"experiment": {"algorithm": "PPOJax"}},
            policy_state={
                "params": {"w": np.asarray([1.0], dtype=np.float32)},
                "run_stats": {},
            },
        )
        algorithm_cls = MagicMock()

        original_instances = StatefulObject._instances.copy()
        with (
            patch.object(
                ValidationVideoRecorder,
                "make_recording_env",
                return_value=(dummy_env, {"recorder_params": {"fps": 10}}),
            ),
            patch.object(ValidationVideoRecorder, "run_recording_rollout"),
            patch(
                "musclemimic.runner.validation_video_checkpoint.load_validation_policy_checkpoint",
                return_value=restored,
            ),
            patch(
                "musclemimic.runner.engine.pick_algorithm",
                return_value=algorithm_cls,
            ),
        ):
            exit_code = validation_video_worker.main([str(payload_path)])

        assert exit_code == 0
        assert dummy_env.stop.called
        assert StatefulObject._instances == original_instances
        with open(result_path) as f:
            result = json.load(f)
        assert result["ok"] is True
        assert result["video_path"] == str(video_path)
        assert result["validation_number"] == 7
        assert result["timestep"] == 700

    def test_worker_writes_error_result_on_exception(self, tmp_path, monkeypatch):
        from musclemimic.runner import validation_video_worker

        payload_path, result_path = self._make_payload(tmp_path)
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")
        restored = ValidationPolicyCheckpoint(
            config={"experiment": {"algorithm": "PPOJax"}},
            policy_state={"params": {}, "run_stats": {}},
        )

        with (
            patch.object(ValidationVideoRecorder, "make_recording_env", side_effect=RuntimeError("env build failed")),
            patch(
                "musclemimic.runner.validation_video_checkpoint.load_validation_policy_checkpoint",
                return_value=restored,
            ),
        ):
            exit_code = validation_video_worker.main([str(payload_path)])

        assert exit_code == 1
        with open(result_path) as f:
            result = json.load(f)
        assert result["ok"] is False
        assert "env build failed" in result["error"]
        assert "RuntimeError" in result["traceback"]


class TestValidationVideoRecorderIntegration:
    """Integration tests with actual environment creation and recording."""

    @pytest.fixture
    def bimanual_training_config(self):
        """Full bimanual training config matching conf_bimanual.yaml."""
        config = OmegaConf.create(
            {
                "experiment": {
                    "task_factory": {
                        "name": "ImitationFactory",
                        "params": {"amass_dataset_conf": {"rel_dataset_path": ["KIT/3/tennis_forehand_right04_poses"]}},
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
                                "right_shoulder_mimic",
                                "right_elbow_mimic",
                                "right_hand_mimic",
                                "left_shoulder_mimic",
                                "left_elbow_mimic",
                                "left_hand_mimic",
                            ],
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
                                "left_hand_mimic",
                            ],
                        },
                        "terminal_state_type": "EnhancedBimanualTerminalStateHandler",
                        "terminal_state_params": {
                            "max_site_deviation": 0.15,
                            "enable_reference_check": False,
                            "site_deviation_mode": "mean",
                        },
                        "timestep": 0.002,
                        "n_substeps": 5,
                    },
                    "validation": {"terminal_state_type": "NoTerminalStateHandler", "terminal_state_params": {}},
                }
            }
        )
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
                    "right_shoulder_mimic",
                    "right_elbow_mimic",
                    "right_hand_mimic",
                    "left_shoulder_mimic",
                    "left_elbow_mimic",
                    "left_hand_mimic",
                ],
                "enable_enhanced_visualization": True,
                "target_geom_rgba": [0.471, 0.38, 0.812, 0.6],
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
                    "right_shoulder_mimic",
                    "right_elbow_mimic",
                    "right_hand_mimic",
                    "left_shoulder_mimic",
                    "left_elbow_mimic",
                    "left_hand_mimic",
                ],
            },
            "recorder_params": {
                "path": str(tmp_path / "videos"),
                "tag": "test_validation",
                "video_name": "MyoBimanualArm",
                "fps": 30,
                "compress": True,
            },
            "timestep": 0.002,
            "n_substeps": 5,
        }

        factory_params = {"amass_dataset_conf": {"rel_dataset_path": ["KIT/3/tennis_forehand_right04_poses"]}}

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

        recorder = ValidationVideoRecorder(video_dir=str(tmp_path / "validation_videos"), length=50, deterministic=True)

        mock_agent_conf = MagicMock()
        mock_agent_conf.config = bimanual_training_config

        try:
            print("[Integration] Building env params...")
            env_params = recorder._build_env_params(mock_agent_conf, "test_validation_1_t10000")

            print("[Integration] Creating validation env...")
            env = ImitationFactory.make(**env_params, **bimanual_training_config.experiment.task_factory.params)

            print("[Integration] Testing env reset...")
            env.reset()

            print("[Integration] Verifying configuration...")
            assert type(env._terminal_state_handler).__name__ == "NoTerminalStateHandler"
            assert hasattr(env, "_viewer")
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
            length=30,  # Short rollout
            deterministic=True,
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
            env = ImitationFactory.make(**env_params, **bimanual_training_config.experiment.task_factory.params)

            print("[Integration] Running rollout with recording enabled...")
            env.reset()
            env.render(record=True)  # Initial render with recording

            action_dim = env.model.nu  # Number of actuators
            for step in range(30):
                action = np.zeros(action_dim)  # Zero action
                _obs, _reward, terminated, truncated, _info = env.step(action)
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
