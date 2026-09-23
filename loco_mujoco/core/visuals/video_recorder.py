import os
import subprocess
import datetime
from pathlib import Path

def _resolve_ffmpeg_binary():
    """Return the imageio-ffmpeg bundled binary, falling back to PATH.

    imageio-ffmpeg is a required dependency and its wheels bundle a static ffmpeg
    built with libx264. Set IMAGEIO_FFMPEG_EXE to override the binary; sdist
    installs (musl, uncommon architectures) ship no binary and fall back to PATH.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


class VideoRecorder(object):
    """
    Simple video record that creates a video from a stream of images.
    """

    def __init__(self, path="./eval_recordings", tag=None, video_name=None, fps=60, compress=True):
        """
        Constructor.

        Args:
            path: Path at which videos will be stored.
            tag: Name of the directory at path in which the video will be stored. If None, a timestamp will be created.
            video_name: Name of the video without extension. Default is "recording".
            fps: Frame rate of the video.
            compress: Whether to compress the video after recording.
        """

        if tag is None:
            date_time = datetime.datetime.now()
            tag = date_time.strftime("%d-%m-%Y_%H-%M-%S")

        self._path = Path(path)
        self._path = self._path / tag

        self._video_name = video_name if video_name else "recording"
        self._counter = 0

        self._fps = fps

        self._compress = compress
        self._video_writer = None
        self._video_writer_path = None

    def __call__(self, frame, debug_info=None):
        """
        Args:
            frame (np.ndarray): Frame to be added to the video (H, W, RGB)
            debug_info (dict): Optional debug info to overlay on frame.
                Keys: global_step, episode_step, done, terminated, truncated
        """
        try:
            import cv2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Video recording requires OpenCV (`cv2`), but it failed to import. "
                "If you're on a headless system, you may be missing system OpenGL libs "
                "(e.g., `libGL.so.1`). Install the appropriate system packages or "
                "disable recording."
            ) from e

        assert frame is not None

        # Overlay debug info if provided
        if debug_info is not None:
            frame = frame.copy()
            g = debug_info.get('global_step', '?')
            e = debug_info.get('episode_step', '?')
            d = debug_info.get('done', '?')
            pd = debug_info.get('prev_done', '?')
            ab = debug_info.get('absorbing', '?')
            term = debug_info.get('terminated', '?')
            trunc = debug_info.get('truncated', '?')
            traj = debug_info.get('traj_no', '?')
            substep = debug_info.get('subtraj_step', '?')
            traj_len = debug_info.get('traj_len', '?')

            # Line 1: step counters
            text1 = f"global:{g} | episode:{e} | traj:{traj} | step:{substep}/{traj_len}"
            cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

            # Line 2: termination flags (highlight in red if any triggered)
            text2 = f"done:{d} | prev_done:{pd} | absorbing:{ab} | terminated:{term} | truncated:{trunc}"
            # Use red color if done or terminated or truncated
            color = (0, 0, 255) if any(v == 1 for v in [d, pd, term, trunc] if isinstance(v, (int, float))) else (255, 255, 0)
            cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._create_video_writer(height, width)

        self._video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _create_video_writer(self, height, width):
        try:
            import cv2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Video recording requires OpenCV (`cv2`), but it failed to import. "
                "If you're on a headless system, you may be missing system OpenGL libs "
                "(e.g., `libGL.so.1`). Install the appropriate system packages or "
                "disable recording."
            ) from e

        name = self._video_name
        if self._counter > 0:
            name += f"-{self._counter}.mp4"
        else:
            name += ".mp4"

        self._path.mkdir(parents=True, exist_ok=True)

        path = self._path / name

        self._video_writer_path = str(path)
        self._video_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                             self._fps, (width, height))

    def stop(self):
        # Idempotent: allow stop() to be called multiple times and/or when no
        # frames were ever recorded (writer never created).
        if self._video_writer is None:
            return self._video_writer_path

        try:
            import cv2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Video recording requires OpenCV (`cv2`), but it failed to import. "
                "If you're on a headless system, you may be missing system OpenGL libs "
                "(e.g., `libGL.so.1`). Install the appropriate system packages or "
                "disable recording."
            ) from e

        cv2.destroyAllWindows()
        self._video_writer.release()

        # compress video
        if self._compress and self._video_writer_path is not None:
            try:
                tmp_file = str(self._path / "tmp_") + self._video_name + ".mp4"
                subprocess.run(
                    [
                        _resolve_ffmpeg_binary(),
                        "-i", self._video_writer_path,  # Input video
                        "-c:v", "libx264",  # H.264 codec
                        "-profile:v", "baseline",  # Set to Baseline profile (can change to main if needed)
                        "-preset", "fast",  # Encoding preset
                        "-crf", "23",  # Quality setting
                        "-an",  # Remove audio
                        "-r", str(self._fps),  # Frame rate
                        "-y",  # Overwrite existing file
                        tmp_file  # Output file
                    ],
                    stdout=subprocess.DEVNULL,  # Suppress standard output
                    check=True  # Raise an error if ffmpeg fails
                )
                os.replace(tmp_file, self._video_writer_path)
                print("Successfully compressed recorded video and saved at: ", self._video_writer_path)

            except subprocess.CalledProcessError as e:
                print(f"Video compression failed: {e}")

        self._video_writer = None

        self._counter += 1

        return self._video_writer_path
