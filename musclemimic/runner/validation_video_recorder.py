"""Validation video recording utilities for training workflows."""

import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms.common.evaluation_rng import evaluation_video_seed
from musclemimic.core.goals.bimanual_tracking import bimanual_visual_goal_type
from musclemimic.core.goals.fullbody_tracking import fullbody_visual_goal_type
from musclemimic.utils.logging import setup_logger

from .trajectory_sharing import VALIDATION_DATASET_CONFIG_KEYS
from .validation_config import apply_validation_reward_termination, to_plain_dict
from .validation_video_checkpoint import DEFAULT_CHECKPOINT_WAIT_TIMEOUT_S
from .validation_video_env import configure_validation_video_worker_env

logger = setup_logger(__name__)


def validation_video_due(validation_number: int, frequency: int) -> bool:
    """Return whether a numbered validation should record a video."""
    return frequency > 0 and validation_number % frequency == 0


@dataclass
class CompletedValidationVideo:
    """A validation video completed by an async recorder process."""

    video_path: str
    timestep: int
    validation_number: int
    log_path: str | None = None


@dataclass
class FailedValidationVideo:
    """A validation video job that finished without producing a usable video."""

    error: str
    timestep: int
    validation_number: int
    return_code: int | None = None
    log_path: str | None = None


@dataclass
class _VideoJob:
    pid: int
    payload_path: Path
    result_path: Path
    log_path: Path
    timestep: int
    validation_number: int
    return_code: int | None = None


class ValidationVideoRecorder:
    """
    Build the standalone recording env and run policy playback in it.

    Used by the CPU-only worker process spawned by AsyncValidationVideoRecorder:
    it reconstructs a MuJoCo env with headless rendering and invokes the
    configured algorithm's policy playback to write a video.
    """

    def __init__(self, video_dir: str, length: int = 500, deterministic: bool = True):
        """
        Args:
            video_dir: Base directory where videos are written.
            length: Number of steps to record per episode.
            deterministic: Use deterministic policy for reproducibility.
        """
        self.video_dir = video_dir
        self.length = max(1, int(length))
        self.deterministic = deterministic

    def _build_env_params(self, agent_conf, tag: str) -> dict:
        """Clone training env params and add recording-specific settings."""
        # Copy training env params.
        env_params = dict(agent_conf.config.experiment.env_params)

        # Switch to the MuJoCo CPU env for recording.
        env_name = env_params.get("env_name", "")
        if isinstance(env_name, str) and env_name.startswith("Mjx"):
            env_params["env_name"] = env_name.replace("Mjx", "", 1)
        # Drop MJX-only parameters.
        for k in ("mjx_backend", "num_envs", "nconmax", "naconmax", "njmax"):
            if k in env_params:
                env_params.pop(k, None)

        validation_config = None

        # Apply validation terminal-state settings.
        if hasattr(agent_conf.config.experiment, "validation"):
            validation_config = agent_conf.config.experiment.validation
            env_params["terminal_state_type"] = validation_config.get("terminal_state_type", "NoTerminalStateHandler")
            env_params["terminal_state_params"] = dict(validation_config.get("terminal_state_params", {}))
            val_init_state_type = validation_config.get("init_state_type", None)
            if val_init_state_type is not None:
                env_params["init_state_type"] = val_init_state_type
                env_params["init_state_params"] = dict(validation_config.get("init_state_params", {}))
        else:
            env_params["terminal_state_type"] = "NoTerminalStateHandler"
            env_params["terminal_state_params"] = {}
        apply_validation_reward_termination(env_params, validation_config)

        # Configure headless recording.
        env_params["headless"] = True
        # Enable goal visualization during recording.
        env_params["visualize_goal"] = True
        # Match recording FPS to the control rate.
        timestep = env_params.get("timestep", 0.002)
        n_substeps = env_params.get("n_substeps", 5)
        control_dt = timestep * n_substeps
        fps = round(1.0 / control_dt)
        env_params["recorder_params"] = {
            "path": self.video_dir,
            "tag": tag,
            "video_name": f"{env_params.get('env_name', 'env')}",
            "fps": fps,
            "compress": True,
        }

        # Mirror visualization settings into goal_params.
        goal_params = dict(env_params.get("goal_params", {}))
        goal_params["visualize_goal"] = True
        # Enable enhanced goal visualization when supported.
        goal_params.setdefault("enable_enhanced_visualization", True)
        goal_params.setdefault("target_geom_rgba", [0.471, 0.38, 0.812, 0.6])
        env_params["goal_params"] = goal_params

        # Use visualization-specific goal classes.
        env_name = env_params.get("env_name", "")
        goal_type = env_params.get("goal_type")
        sites = goal_params.get("sites_for_mimic", [])
        if "Bimanual" in env_name and goal_type:
            env_params["goal_type"] = bimanual_visual_goal_type(goal_type)
            if sites:
                env_params["goal_params"]["sites_for_mimic"] = sites
        elif "MyoFullBody" in env_name and goal_type:
            env_params["goal_type"] = fullbody_visual_goal_type(goal_type)
            if sites:
                env_params["goal_params"]["sites_for_mimic"] = sites

        # Reuse training timing parameters.
        for k in ("timestep", "n_substeps"):
            if k in agent_conf.config.experiment.env_params:
                env_params[k] = agent_conf.config.experiment.env_params[k]

        # Start each validation rollout at trajectory step 0.
        if hasattr(agent_conf.config.experiment, "validation"):
            if agent_conf.config.experiment.validation.get("start_from_beginning", False):
                if "th_params" not in env_params:
                    env_params["th_params"] = {}
                env_params["th_params"]["start_from_random_step"] = False

        return env_params

    def _build_task_params(
        self,
        agent_conf,
        validation_number: int = 1,
    ) -> dict:
        """Build task factory params for a short validation recording."""
        raw_task_params = agent_conf.config.experiment.task_factory.params
        task_params = to_plain_dict(raw_task_params)

        validation_cfg = agent_conf.config.experiment.get("validation", {})
        dataset_overrides = {
            key: validation_cfg.get(key)
            for key in VALIDATION_DATASET_CONFIG_KEYS
            if validation_cfg.get(key) is not None
        }
        if dataset_overrides:
            for key in VALIDATION_DATASET_CONFIG_KEYS:
                task_params.pop(key, None)
            task_params.update(
                {key: to_plain_dict(dataset_config) for key, dataset_config in dataset_overrides.items()}
            )

        # Cap large AMASS datasets for short recordings.
        amass_conf = task_params.get("amass_dataset_conf")
        if isinstance(amass_conf, dict):
            amass_conf = dict(amass_conf)
            amass_conf["max_motions"] = max(1, int(validation_cfg.get("video_max_motions", 3)))
            amass_conf["max_motions_seed"] = evaluation_video_seed(
                int(validation_cfg.get("eval_seed", 0)),
                int(validation_number),
            )
            amass_conf["write_materialized_cache"] = False
            task_params["amass_dataset_conf"] = amass_conf
        return task_params

    def make_recording_env(
        self,
        agent_conf,
        tag: str,
        validation_number: int = 1,
    ):
        """Create the standalone MuJoCo environment used for recording."""
        try:
            factory = TaskFactory.get_factory_cls(agent_conf.config.experiment.task_factory.name)
            env_params = self._build_env_params(agent_conf, tag)
            task_params = self._build_task_params(
                agent_conf,
                validation_number=validation_number,
            )
            logger.info("Building eval env for validation recording (tag=%s)", tag)
            env = factory.make(**env_params, **task_params)
        except Exception as e:
            logger.exception("Failed to create validation recording environment: %s: %s", type(e).__name__, e)
            raise
        return env, env_params

    def run_recording_rollout(
        self,
        algorithm_cls,
        env,
        agent_conf,
        agent_state,
        env_params: dict,
    ) -> None:
        """Run policy playback in an already-created recording env."""
        fps = env_params["recorder_params"]["fps"]
        logger.info("Eval env ready; starting validation recording rollout for %s steps @ %s fps", self.length, fps)

        algorithm_cls.play_policy(
            env,
            agent_conf,
            agent_state,
            n_envs=1,
            n_steps=self.length,
            render=True,  # must be True for recording to emit frames
            record=True,
            deterministic=self.deterministic,
            use_mujoco=True,  # always record with the MuJoCo CPU env
            wrap_env=True,
            train_state_seed=0,
        )

    @staticmethod
    def build_recording_tag(validation_number: int, timestep: int | None = None) -> str:
        time_tag = time.strftime("%Y%m%d_%H%M%S")
        return f"validation_{validation_number}_t{timestep if timestep is not None else 0}_{time_tag}"


class AsyncValidationVideoRecorder:
    """
    Record validation videos in a separate CPU-only Python process.

    record_episode writes a lightweight checkpoint reference and spawns a
    worker (at most one at a time). The worker runs with JAX_PLATFORMS=cpu
    injected into its environment at spawn time, so it never touches the
    training GPUs, and writes a JSON result file that collect_finished and
    wait_for_all turn into Completed/FailedValidationVideo.
    """

    def __init__(
        self,
        video_dir: str,
        frequency: int = 10,
        length: int = 500,
        deterministic: bool = True,
        shutdown_timeout_s: float | None = None,
        checkpoint_wait_timeout_s: float = DEFAULT_CHECKPOINT_WAIT_TIMEOUT_S,
    ):
        self.video_dir = video_dir
        self.frequency = max(1, int(frequency))
        self.length = max(1, int(length))
        self.deterministic = deterministic
        self.shutdown_timeout_s = None if shutdown_timeout_s is None else float(shutdown_timeout_s)
        self.checkpoint_wait_timeout_s = float(checkpoint_wait_timeout_s)
        if self.checkpoint_wait_timeout_s <= 0.0:
            raise ValueError("checkpoint_wait_timeout_s must be positive")
        self.job_dir = Path(video_dir) / "_validation_video_jobs"
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_paths(
            *(
                path
                for pattern in ("*.pkl", "*.payload.json", "*.result.json", "*.log")
                for path in self.job_dir.glob(pattern)
            )
        )
        self._jobs: list[_VideoJob] = []
        self._jobs_lock = threading.Lock()

    @staticmethod
    def _cleanup_paths(*paths: Path) -> None:
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Failed to remove validation video job file %s: %s", path, exc)

    def _job_return_code(self, job: _VideoJob) -> int | None:
        """Reap the worker if it exited; must be called with _jobs_lock held."""
        if job.return_code is not None:
            return job.return_code
        try:
            waited_pid, status = os.waitpid(job.pid, os.WNOHANG)
        except ChildProcessError:
            job.return_code = 1
            return job.return_code
        if waited_pid == 0:
            return None
        job.return_code = os.waitstatus_to_exitcode(status)
        return job.return_code

    @staticmethod
    def _spawn_worker(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        file_actions = [
            (os.POSIX_SPAWN_OPEN, 1, str(log_path), flags, 0o666),
            (os.POSIX_SPAWN_DUP2, 1, 2),
        ]
        return os.posix_spawn(cmd[0], cmd, env, file_actions=file_actions)

    @staticmethod
    def _read_result(result_path: Path) -> tuple[dict | None, str | None]:
        try:
            with open(result_path) as f:
                return json.load(f), None
        except Exception as exc:
            return None, f"finished without a readable result: {exc}"

    def _failed_result(self, job: _VideoJob, error: str, return_code: int | None) -> FailedValidationVideo:
        failure = FailedValidationVideo(
            error=error,
            timestep=job.timestep,
            validation_number=job.validation_number,
            return_code=return_code,
            log_path=str(job.log_path),
        )
        logger.error(
            "Async validation recording failed (validation=%s, timestep=%s, returncode=%s, log=%s): %s",
            job.validation_number,
            job.timestep,
            return_code,
            job.log_path,
            error,
        )
        return failure

    def collect_finished(self) -> list[CompletedValidationVideo | FailedValidationVideo]:
        """Collect completed async recording jobs.

        Payload and result files are always removed; the worker log file is
        kept on failure since it is the only diagnostics for workers that
        crashed before writing a result.
        """
        completed: list[CompletedValidationVideo | FailedValidationVideo] = []
        active: list[_VideoJob] = []
        with self._jobs_lock:
            for job in self._jobs:
                return_code = self._job_return_code(job)
                if return_code is None:
                    active.append(job)
                    continue

                result, read_error = self._read_result(job.result_path)
                self._cleanup_paths(job.payload_path, job.result_path)

                if read_error is not None:
                    completed.append(self._failed_result(job, read_error, return_code))
                    continue
                if return_code != 0:
                    error = result.get("error") or result.get("traceback") or f"worker exited with {return_code}"
                    completed.append(self._failed_result(job, error, return_code))
                    continue

                video_path = result.get("video_path")
                if video_path and os.path.exists(video_path):
                    self._cleanup_paths(job.log_path)
                    completed.append(
                        CompletedValidationVideo(
                            video_path=video_path,
                            timestep=job.timestep,
                            validation_number=job.validation_number,
                        )
                    )
                else:
                    error = result.get("error") or f"result did not contain an existing video: {result}"
                    completed.append(self._failed_result(job, error, return_code))

            self._jobs = active
        return completed

    def wait_for_all(
        self,
        timeout_s: float | None = None,
        poll_interval_s: float = 0.5,
    ) -> list[CompletedValidationVideo | FailedValidationVideo]:
        """Wait for scheduled recordings and return every completed or failed job."""
        timeout = self.shutdown_timeout_s if timeout_s is None else float(timeout_s)
        deadline = None if timeout is None else time.monotonic() + timeout
        completed: list[CompletedValidationVideo | FailedValidationVideo] = []
        while True:
            completed.extend(self.collect_finished())
            with self._jobs_lock:
                done = not self._jobs
            if done:
                return completed
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning("Timed out waiting for async validation recording to finish")
                return completed
            time.sleep(float(poll_interval_s))

    def record_episode(
        self,
        checkpoint_path: str,
        *,
        validation_number: int,
        timestep: int,
    ) -> None:
        """Spawn a worker for an exact checkpoint unless one is still running."""
        if not validation_video_due(validation_number, self.frequency):
            return

        tag = ValidationVideoRecorder.build_recording_tag(validation_number, timestep)
        job_id = f"{tag}_{uuid.uuid4().hex[:8]}"
        payload_path = self.job_dir / f"{job_id}.payload.json"
        result_path = self.job_dir / f"{job_id}.result.json"
        log_path = self.job_dir / f"{job_id}.log"

        payload = {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_wait_timeout_s": self.checkpoint_wait_timeout_s,
            "video_dir": self.video_dir,
            "length": self.length,
            "deterministic": self.deterministic,
            "validation_number": int(validation_number),
            "timestep": int(timestep),
            "tag": tag,
            "result_path": str(result_path),
        }

        with self._jobs_lock:
            if any(self._job_return_code(job) is None for job in self._jobs):
                logger.warning("Skipping async validation recording because another recording is still running")
                return
            try:
                with open(payload_path, "w") as f:
                    json.dump(payload, f)
                worker_env = configure_validation_video_worker_env(os.environ.copy())
                cmd = [sys.executable, "-m", "musclemimic.runner.validation_video_worker", str(payload_path)]
                pid = self._spawn_worker(cmd, worker_env, log_path)
            except Exception:
                self._cleanup_paths(payload_path, result_path, log_path)
                raise
            self._jobs.append(
                _VideoJob(
                    pid=pid,
                    payload_path=payload_path,
                    result_path=result_path,
                    log_path=log_path,
                    timestep=int(timestep),
                    validation_number=int(validation_number),
                )
            )
        logger.info(
            "Scheduled async validation recording (validation=%s, timestep=%s, pid=%s, log=%s)",
            validation_number,
            timestep,
            pid,
            log_path,
        )
