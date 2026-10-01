"""Display and headless rendering utilities."""

import os
import subprocess
import sys


def detect_headless_environment() -> bool:
    """
    Detect if running in a headless environment (no display available).

    Returns:
        True if headless rendering should be used, False otherwise.
    """

    if sys.platform == "darwin":
        return False 

    if "DISPLAY" not in os.environ:
        return True

    try:
        result = subprocess.run(["xset", "q"], capture_output=True, text=True, timeout=2)
        return result.returncode != 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return True


def setup_headless_rendering() -> None:
    """
    Set up environment for headless rendering using EGL.

    This configures MuJoCo to use EGL for offscreen rendering when
    no display is available.
    """
    backend = os.environ.setdefault("MUJOCO_GL", "egl")
    if backend == "egl":
        print("No display detected - enabling headless rendering with EGL")
        print("   Set MUJOCO_GL=egl for headless rendering")
    else:
        print(f"No display detected - preserving MUJOCO_GL={backend} for headless rendering")


def setup_headless_rendering_if_needed() -> None:
    """
    Automatically set up headless rendering if no display is available.

    This is a convenience function that combines detection and setup.
    """
    if detect_headless_environment():
        setup_headless_rendering()
