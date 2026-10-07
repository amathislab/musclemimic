import sys
import types

import numpy as np

from loco_mujoco.core.visuals.video_recorder import VideoRecorder


def test_stop_is_idempotent(tmp_path, monkeypatch):
    release_calls = {"n": 0}

    class DummyWriter:
        def write(self, _frame):
            return None

        def release(self):
            release_calls["n"] += 1
            return None

    cv2 = types.ModuleType("cv2")
    cv2.COLOR_RGB2BGR = 0
    cv2.cvtColor = lambda frame, _code: frame
    cv2.VideoWriter_fourcc = lambda *_args: 0
    cv2.VideoWriter = lambda *_args, **_kwargs: DummyWriter()

    def fail_destroy_all_windows():
        raise AssertionError("VideoRecorder.stop() must not touch GUI windows")

    cv2.destroyAllWindows = fail_destroy_all_windows
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    recorder = VideoRecorder(
        path=str(tmp_path),
        tag="test",
        video_name="recording",
        fps=30,
        compress=False,
    )
    recorder(np.zeros((8, 8, 3), dtype=np.uint8))

    path1 = recorder.stop()
    path2 = recorder.stop()

    assert path1 is not None
    assert path2 == path1
    assert release_calls["n"] == 1


def test_stop_without_frames_returns_none(tmp_path):
    recorder = VideoRecorder(
        path=str(tmp_path),
        tag="test",
        video_name="recording",
        fps=30,
        compress=False,
    )
    assert recorder.stop() is None


def test_debug_overlay_draws_motion_name_in_top_left(tmp_path, monkeypatch):
    drawn_text = []

    class DummyWriter:
        def write(self, _frame):
            return None

        def release(self):
            return None

    cv2 = types.ModuleType("cv2")
    cv2.COLOR_RGB2BGR = 0
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.cvtColor = lambda frame, _code: frame
    cv2.VideoWriter_fourcc = lambda *_args: 0
    cv2.VideoWriter = lambda *_args, **_kwargs: DummyWriter()

    def put_text(frame, text, origin, *_args):
        drawn_text.append((text, origin))
        return frame

    cv2.putText = put_text
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    recorder = VideoRecorder(
        path=str(tmp_path),
        tag="test",
        video_name="recording",
        fps=30,
        compress=False,
    )
    recorder(
        np.zeros((120, 320, 3), dtype=np.uint8),
        debug_info={"motion_name": "KIT/walk_01", "traj_no": 3},
    )

    assert drawn_text[0] == ("motion:KIT/walk_01", (10, 30))
    assert drawn_text[1][1] == (10, 60)
