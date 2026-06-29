__all__ = ["TrajectoryViserViewer"]


def __getattr__(name: str):
    if name == "TrajectoryViserViewer":
        from .trajectory_viewer import TrajectoryViserViewer

        return TrajectoryViserViewer
    raise AttributeError(name)


# C3D viewer available via: python -m musclemimic.web_viewer.c3d_viewer
