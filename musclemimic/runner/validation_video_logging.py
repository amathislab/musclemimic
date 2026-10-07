"""Host-side logging for async validation videos."""

from __future__ import annotations

import threading

from musclemimic.utils.logging import setup_logger

from .logging import ExperimentLogger
from .validation_video_recorder import (
    AsyncValidationVideoRecorder,
    CompletedValidationVideo,
    FailedValidationVideo,
)

logger = setup_logger(__name__)


class ValidationVideoLogMonitor:
    """Poll completed validation videos and log them outside JAX callbacks."""

    def __init__(
        self,
        recorder: AsyncValidationVideoRecorder,
        experiment_logger: ExperimentLogger,
        poll_interval_s: float = 2.0,
    ):
        self._recorder = recorder
        self._experiment_logger = experiment_logger
        self._poll_interval_s = float(poll_interval_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="validation-video-log-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._poll_interval_s + 1.0))

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval_s):
            log_video_results(
                self._experiment_logger,
                self._recorder.collect_finished(),
            )


def log_video_results(
    experiment_logger: ExperimentLogger,
    results: list[CompletedValidationVideo | FailedValidationVideo],
) -> None:
    """Log completed videos to W&B; failures were already logged by the recorder."""
    for result in results:
        if isinstance(result, FailedValidationVideo):
            continue
        logger.info(f"Validation video recorded: {result.video_path} (validation_step={result.timestep})")
        experiment_logger.log_video(
            result.video_path,
            timestep=result.timestep,
        )


def flush_validation_videos(
    recorder: AsyncValidationVideoRecorder | None,
    experiment_logger: ExperimentLogger,
    *,
    wait: bool = True,
    timeout_s: float | None = None,
) -> None:
    """Collect and log pending validation videos before shutdown."""
    if recorder is None:
        return

    results = recorder.wait_for_all(timeout_s=timeout_s) if wait else recorder.collect_finished()
    log_video_results(experiment_logger, results)
