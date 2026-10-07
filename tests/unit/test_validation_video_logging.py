import time
from threading import Event, Thread

from omegaconf import OmegaConf

from musclemimic.runner.experiment_logging import build_logging_callback
from musclemimic.runner.logging import ExperimentHooks, ExperimentLogger
from musclemimic.runner.validation_video_logging import (
    ValidationVideoLogMonitor,
    flush_validation_videos,
    log_video_results,
)
from musclemimic.runner.validation_video_recorder import CompletedValidationVideo, FailedValidationVideo


class _FinishedRecorder:
    def collect_finished(self):
        return [
            CompletedValidationVideo(
                video_path="/tmp/validation.mp4",
                timestep=100,
                validation_number=1,
            )
        ]


class _PendingRecorder:
    def __init__(self):
        self.waited = False
        self.collected = False
        self.timeout_s = None

    def wait_for_all(self, timeout_s=None):
        self.waited = True
        self.timeout_s = timeout_s
        return [
            CompletedValidationVideo(
                video_path="/tmp/final_validation.mp4",
                timestep=300,
                validation_number=3,
            )
        ]

    def collect_finished(self):
        self.collected = True
        return []


class _FailedRecorder:
    def collect_finished(self):
        return [
            FailedValidationVideo(
                error="RuntimeError: renderer failed",
                timestep=200,
                validation_number=2,
                return_code=1,
            )
        ]


class _EventuallyFinishedRecorder:
    def __init__(self):
        self.calls = 0

    def collect_finished(self):
        self.calls += 1
        if self.calls == 1:
            return [
                CompletedValidationVideo(
                    video_path="/tmp/async_validation.mp4",
                    timestep=300,
                    validation_number=3,
                )
            ]
        return []


class _SchedulingRecorder:
    def __init__(self):
        self.calls = []
        self.frequency = 1

    def record_episode(self, checkpoint_path, *, validation_number, timestep):
        self.calls.append((checkpoint_path, validation_number, timestep))


class _FakeWandb:
    def __init__(self):
        self.logs = []

    def log(self, data, *, step):
        self.logs.append((data, step))


class _CaptureHooks(ExperimentHooks):
    def __init__(self):
        self.video_logs = []

    def on_validation_video(self, use_wandb, wandb, video_path, timestep, *, log_step=None):
        self.video_logs.append((use_wandb, video_path, timestep, log_step))


class _BlockingHooks(_CaptureHooks):
    def __init__(self):
        super().__init__()
        self.video_started = Event()
        self.release_video = Event()

    def on_validation_video(self, use_wandb, wandb, video_path, timestep, *, log_step=None):
        self.video_started.set()
        self.release_video.wait(timeout=1.0)
        super().on_validation_video(use_wandb, wandb, video_path, timestep, log_step=log_step)


def test_experiment_logger_emits_scalars_and_videos_through_one_owner():
    hooks = _CaptureHooks()
    wandb = _FakeWandb()
    experiment_logger = ExperimentLogger(hooks, wandb)

    experiment_logger.log({"loss": 1.0}, step=100)
    experiment_logger.log_video("/tmp/validation.mp4", timestep=50)

    assert wandb.logs == [({"loss": 1.0}, 100)]
    assert hooks.video_logs == [(True, "/tmp/validation.mp4", 50, 100)]


def test_async_validation_video_uses_current_wandb_step():
    config = OmegaConf.create({"experiment": {"algorithm": "PPOJax"}})
    hooks = _CaptureHooks()
    experiment_logger = ExperimentLogger(hooks, None)
    recorder = _FinishedRecorder()
    callback = build_logging_callback(
        env=None,
        config=config,
        experiment_logger=experiment_logger,
        video_recorder=None,
    )

    # The training callback advances the W&B step watermark ...
    callback(
        {
            "mean_episode_return": 1.0,
            "mean_episode_length": 2.0,
            "learning_rate": 3.0,
            "max_timestep": 200,
            "jax_raw_timestep": 200,
        }
    )
    # ... which the video log (normally driven by the monitor thread) must respect.
    log_video_results(experiment_logger, recorder.collect_finished())

    assert hooks.video_logs == [(False, "/tmp/validation.mp4", 100, 200)]


def test_video_upload_does_not_hold_scalar_logging_lock():
    hooks = _BlockingHooks()
    experiment_logger = ExperimentLogger(hooks, None)
    video_thread = Thread(
        target=experiment_logger.log_video,
        args=("/tmp/validation.mp4",),
        kwargs={"timestep": 100},
    )
    scalar_logged = Event()

    video_thread.start()
    assert hooks.video_started.wait(timeout=1.0)
    scalar_thread = Thread(
        target=lambda: (experiment_logger.log({}, step=200), scalar_logged.set()),
    )
    scalar_thread.start()
    assert scalar_logged.wait(timeout=1.0)

    hooks.release_video.set()
    video_thread.join(timeout=1.0)
    scalar_thread.join(timeout=1.0)
    assert not video_thread.is_alive()
    assert not scalar_thread.is_alive()


def test_validation_video_uses_exact_checkpoint_update(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "algorithm": "PPOJax",
                "checkpoint_dir": str(tmp_path / "checkpoints"),
            }
        }
    )
    hooks = _CaptureHooks()
    experiment_logger = ExperimentLogger(hooks, None)
    recorder = _SchedulingRecorder()
    callback = build_logging_callback(
        env=None,
        config=config,
        experiment_logger=experiment_logger,
        video_recorder=recorder,
    )

    callback(
        {
            "mean_episode_return": 1.0,
            "mean_episode_length": 2.0,
            "learning_rate": 3e-4,
            "max_timestep": 200,
            "jax_raw_timestep": 200,
            "has_validation_update": True,
            "_checkpoint_update": 17,
        }
    )

    assert recorder.calls == [(str((tmp_path / "checkpoints" / "checkpoint_17").resolve()), 1, 200)]


def test_validation_video_uses_scheduled_number_after_resume(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "algorithm": "FlashSACJax",
                "checkpoint_dir": str(tmp_path / "checkpoints"),
            }
        }
    )
    recorder = _SchedulingRecorder()
    callback = build_logging_callback(
        env=None,
        config=config,
        experiment_logger=ExperimentLogger(_CaptureHooks(), None),
        video_recorder=recorder,
    )

    callback(
        {
            "max_timestep": 2_000,
            "jax_raw_timestep": 200,
            "has_validation_update": True,
            "_checkpoint_update": 17,
            "_validation_number": 9,
        }
    )

    assert recorder.calls == [(str((tmp_path / "checkpoints" / "checkpoint_17").resolve()), 9, 2_000)]


def test_flush_validation_videos_waits_for_pending_jobs():
    hooks = _CaptureHooks()
    experiment_logger = ExperimentLogger(hooks, None)
    recorder = _PendingRecorder()

    flush_validation_videos(recorder, experiment_logger, timeout_s=12.0)

    assert recorder.waited is True
    assert recorder.timeout_s == 12.0
    assert hooks.video_logs == [(False, "/tmp/final_validation.mp4", 300, 300)]


def test_flush_validation_videos_uses_last_wandb_step():
    hooks = _CaptureHooks()
    experiment_logger = ExperimentLogger(hooks, None)
    recorder = _PendingRecorder()
    experiment_logger.log({}, step=400)

    flush_validation_videos(recorder, experiment_logger)

    assert recorder.timeout_s is None
    assert hooks.video_logs == [(False, "/tmp/final_validation.mp4", 300, 400)]


def test_flush_validation_videos_can_skip_waiting_on_error_exit():
    hooks = _CaptureHooks()
    experiment_logger = ExperimentLogger(hooks, None)
    recorder = _PendingRecorder()

    flush_validation_videos(recorder, experiment_logger, wait=False)

    assert recorder.waited is False
    assert recorder.collected is True


def test_finished_video_failures_skip_video_callback():
    # Failures are logged by the recorder when collected; the logging layer
    # must skip them without invoking the video callback.
    hooks = _CaptureHooks()
    experiment_logger = ExperimentLogger(hooks, None)
    recorder = _FailedRecorder()

    log_video_results(experiment_logger, recorder.collect_finished())

    assert hooks.video_logs == []


def test_validation_video_monitor_logs_without_training_callback():
    hooks = _CaptureHooks()
    experiment_logger = ExperimentLogger(hooks, None)
    experiment_logger.log({}, step=500)
    recorder = _EventuallyFinishedRecorder()

    monitor = ValidationVideoLogMonitor(recorder, experiment_logger, poll_interval_s=0.01)
    monitor.start()
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not hooks.video_logs:
            time.sleep(0.01)
    finally:
        monitor.stop()

    assert hooks.video_logs == [(False, "/tmp/async_validation.mp4", 300, 500)]
