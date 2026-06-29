"""Interactive C3D motion capture viewer using Viser.

Visualizes marker trajectories and skeleton from C3D files (e.g. OpenBiomechanics).
Renders markers as spheres, bones as line segments, with playback controls and a frame scrubber.

Usage:
    uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer <path_to_c3d_file>
    uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer <path_to_c3d_file> --trail 10 --port 8080
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser

from .c3d.markers import load_c3d_markers

# ---------------------------------------------------------------------------
# Skeleton topology: Plug-in Gait marker connections
# Each tuple is (marker_a, marker_b, segment_name) for color grouping.
# Connections that reference missing markers are silently skipped.
# ---------------------------------------------------------------------------

# fmt: off
SKELETON_CONNECTIONS = [
    # Head
    ("LFHD", "RFHD", "head"), ("LBHD", "RBHD", "head"),
    ("LFHD", "LBHD", "head"), ("RFHD", "RBHD", "head"),

    # Torso
    ("C7", "CLAV", "torso"), ("C7", "T10", "torso"),
    ("CLAV", "STRN", "torso"), ("T10", "STRN", "torso"),
    ("CLAV", "LSHO", "torso"), ("CLAV", "RSHO", "torso"),
    ("C7", "LSHO", "torso"), ("C7", "RSHO", "torso"),
    ("STRN", "LASI", "torso"), ("STRN", "RASI", "torso"),

    # Pelvis
    ("LASI", "RASI", "pelvis"), ("LPSI", "RPSI", "pelvis"),
    ("LASI", "LPSI", "pelvis"), ("RASI", "RPSI", "pelvis"),

    # Left arm
    ("LSHO", "LUPA", "left"), ("LUPA", "LELB", "left"),
    ("LELB", "LMELB", "left"), ("LELB", "LFRM", "left"),
    ("LFRM", "LWRA", "left"), ("LFRM", "LWRB", "left"),
    ("LWRA", "LWRB", "left"), ("LWRA", "LFIN", "left"),

    # Right arm
    ("RSHO", "RUPA", "right"), ("RUPA", "RELB", "right"),
    ("RELB", "RMELB", "right"), ("RELB", "RFRM", "right"),
    ("RFRM", "RWRA", "right"), ("RFRM", "RWRB", "right"),
    ("RWRA", "RWRB", "right"), ("RWRA", "RFIN", "right"),

    # Left leg
    ("LASI", "LTHI", "left"), ("LTHI", "LKNE", "left"),
    ("LKNE", "LMKNE", "left"), ("LKNE", "LTIB", "left"),
    ("LTIB", "LANK", "left"), ("LANK", "LMANK", "left"),
    ("LANK", "LTOE", "left"), ("LANK", "LHEE", "left"),

    # Right leg
    ("RASI", "RTHI", "right"), ("RTHI", "RKNE", "right"),
    ("RKNE", "RMKNE", "right"), ("RKNE", "RTIB", "right"),
    ("RTIB", "RANK", "right"), ("RANK", "RMANK", "right"),
    ("RANK", "RTOE", "right"), ("RANK", "RHEE", "right"),
]
# fmt: on

SEGMENT_COLORS = {
    "head": np.array([220, 220, 220], dtype=np.uint8),
    "torso": np.array([200, 200, 200], dtype=np.uint8),
    "pelvis": np.array([180, 180, 180], dtype=np.uint8),
    "left": np.array([70, 130, 230], dtype=np.uint8),  # blue
    "right": np.array([230, 70, 70], dtype=np.uint8),  # red
    "bat": np.array([180, 140, 60], dtype=np.uint8),  # gold
}


def _viewer_label(label: str) -> str:
    """Normalize raw C3D labels for display without applying retargeting aliases."""
    return str(label).replace(" ", "").split(":")[-1].upper()


def _load_c3d(path: str) -> tuple[np.ndarray, list[str], float]:
    """Load a C3D file and return (positions, labels, frame_rate).

    Returns:
        positions: (N_frames, N_markers, 3) float32 array in meters
        labels: list of marker name strings
        frame_rate: capture rate in Hz
    """
    return load_c3d_markers(path)


def _build_bone_indices(
    labels: list[str],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Map skeleton connections to marker index pairs. Returns (index_pairs, colors)."""
    viewer_labels = [_viewer_label(name) for name in labels]
    label_map = {name: idx for idx, name in enumerate(viewer_labels)}
    pairs = []
    colors = []

    for a, b, segment in SKELETON_CONNECTIONS:
        if a in label_map and b in label_map:
            pairs.append((label_map[a], label_map[b]))
            colors.append(SEGMENT_COLORS[segment])

    # Bat markers (Marker1..Marker10) — connect sequentially
    bat_markers = sorted(
        [label for label in viewer_labels if label.startswith("MARKER") and label[6:].isdigit()],
        key=lambda x: int(x[6:]),
    )
    for i in range(len(bat_markers) - 1):
        pairs.append((label_map[bat_markers[i]], label_map[bat_markers[i + 1]]))
        colors.append(SEGMENT_COLORS["bat"])

    if not pairs:
        return [], np.empty((0, 3), dtype=np.uint8)
    return pairs, np.array(colors, dtype=np.uint8)


def _marker_colors(labels: list[str]) -> np.ndarray:
    """Assign a color per marker: left=blue, right=red, center=gray, bat=gold."""
    colors = np.full((len(labels), 3), 200, dtype=np.uint8)  # default gray
    for i, name in enumerate(labels):
        viewer_name = _viewer_label(name)
        if viewer_name.startswith("L"):
            colors[i] = [70, 130, 230]
        elif viewer_name.startswith("R"):
            colors[i] = [230, 70, 70]
        elif viewer_name.startswith("MARKER"):
            colors[i] = [180, 140, 60]
    return colors


def main():
    parser = argparse.ArgumentParser(description="Interactive C3D viewer (Viser)")
    parser.add_argument("c3d_file", help="Path to .c3d file")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--trail", type=int, default=0, help="Trail length in frames (0=off)")
    parser.add_argument("--marker-size", type=float, default=0.012, help="Marker sphere radius (m)")
    args = parser.parse_args()

    # Load data
    positions, labels, frame_rate = _load_c3d(args.c3d_file)
    n_frames, n_markers, _ = positions.shape
    bone_pairs, bone_colors = _build_bone_indices(labels)
    marker_colors = _marker_colors(labels)
    if len(marker_colors) != n_markers:
        padded_marker_colors = np.full((n_markers, 3), 200, dtype=np.uint8)
        n_labeled = min(len(marker_colors), n_markers)
        padded_marker_colors[:n_labeled] = marker_colors[:n_labeled]
        marker_colors = padded_marker_colors

    print(f"Loaded {args.c3d_file}")
    print(f"  {n_markers} markers, {n_frames} frames @ {frame_rate} Hz")
    print(f"  Duration: {n_frames / frame_rate:.2f}s")
    print(f"  Skeleton bones: {len(bone_pairs)}")

    # Setup Viser
    server = viser.ViserServer(label="C3D Viewer", port=args.port)
    server.scene.configure_environment_map(environment_intensity=0.6)

    # State
    paused = [True]
    current_frame = [0]
    speed = [1.0]
    loop = [True]

    # --- GUI ---
    tabs = server.gui.add_tab_group()
    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
        status_html = server.gui.add_html("")

        with server.gui.add_folder("Playback"):
            frame_slider = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)

            @frame_slider.on_update
            def _(_ev) -> None:
                current_frame[0] = int(frame_slider.value)

            pause_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)

            @pause_button.on_click
            def _(_ev) -> None:
                paused[0] = not paused[0]
                pause_button.label = "Pause" if not paused[0] else "Play"
                pause_button.icon = viser.Icon.PLAYER_PAUSE if not paused[0] else viser.Icon.PLAYER_PLAY

            speed_buttons = server.gui.add_button_group("Speed", options=["0.25x", "0.5x", "1x", "2x"])

            @speed_buttons.on_click
            def _(event) -> None:
                speed[0] = float(event.target.value.replace("x", ""))
                _update_status()

            loop_checkbox = server.gui.add_checkbox("Loop", initial_value=True)

            @loop_checkbox.on_update
            def _(_ev) -> None:
                loop[0] = loop_checkbox.value

        with server.gui.add_folder("Display"):
            marker_size_slider = server.gui.add_slider(
                "Marker size", min=0.003, max=0.03, step=0.001, initial_value=args.marker_size
            )
            trail_slider = server.gui.add_slider(
                "Trail length", min=0, max=min(60, n_frames), step=1, initial_value=args.trail
            )
            show_bones = server.gui.add_checkbox("Show skeleton", initial_value=True)
            show_markers = server.gui.add_checkbox("Show markers", initial_value=True)

        with server.gui.add_folder("Info"):
            server.gui.add_html(
                f"<b>File:</b> {Path(args.c3d_file).name}<br>"
                f"<b>Markers:</b> {n_markers}<br>"
                f"<b>Frames:</b> {n_frames}<br>"
                f"<b>Rate:</b> {frame_rate} Hz<br>"
                f"<b>Duration:</b> {n_frames / frame_rate:.2f}s"
            )

    def _update_status():
        state = "Paused" if paused[0] else "Playing"
        f = current_frame[0]
        t = f / frame_rate
        status_html.content = (
            f"<b>{state}</b> &nbsp; Frame {f}/{n_frames - 1} &nbsp; t={t:.3f}s &nbsp; Speed: {speed[0]:.2f}x"
        )

    # --- Helper ---
    def _valid_marker_mask(frame_pos: np.ndarray) -> np.ndarray:
        return ~(np.isnan(frame_pos).any(axis=-1) | np.all(np.isclose(frame_pos, 0.0), axis=-1))

    def _visible_marker_geometry(frame_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = _valid_marker_mask(frame_pos)
        return frame_pos[valid].astype(np.float32), marker_colors[valid]

    def _build_bone_geometry(
        frame_pos: np.ndarray,
        pairs: list[tuple[int, int]],
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = _valid_marker_mask(frame_pos)
        pts = []
        cols = []
        for (a, b), color in zip(pairs, colors, strict=True):
            if valid[a] and valid[b]:
                pts.append([frame_pos[a], frame_pos[b]])
                cols.append([color, color])
        if not pts:
            return np.empty((0, 2, 3), dtype=np.float32), np.empty((0, 2, 3), dtype=np.uint8)
        return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8)

    # --- Ground grid ---
    server.scene.add_grid(
        "/ground",
        width=6.0,
        height=6.0,
        plane="xy",
        cell_color=(180, 180, 180),
        section_color=(120, 120, 120),
        cell_thickness=1,
        section_thickness=2,
    )

    # --- Scene handles ---
    # Markers as point cloud (efficient for many markers)
    marker_pts, marker_cols = _visible_marker_geometry(positions[0])
    marker_handle = server.scene.add_point_cloud(
        "/markers",
        points=marker_pts,
        colors=marker_cols,
        point_size=args.marker_size,
        point_shape="rounded",
    )

    # Bones as line segments
    bone_handle = None
    if bone_pairs:
        bone_pts, bone_cols = _build_bone_geometry(positions[0], bone_pairs, bone_colors)
        bone_handle = server.scene.add_line_segments(
            "/bones",
            points=bone_pts,
            colors=bone_cols,
            line_width=3,
        )

    # Trail handle (created on demand)
    trail_handle = None

    def _update_frame(frame_idx: int):
        nonlocal trail_handle
        pos = positions[frame_idx]

        with server.atomic():
            # Update markers
            if show_markers.value:
                marker_pts, marker_cols = _visible_marker_geometry(pos)
                marker_handle.points = marker_pts
                marker_handle.colors = marker_cols
                marker_handle.point_size = marker_size_slider.value
                marker_handle.visible = True
            else:
                marker_handle.visible = False

            # Update bones
            if bone_handle is not None:
                if show_bones.value:
                    bone_pts, bone_cols = _build_bone_geometry(pos, bone_pairs, bone_colors)
                    bone_handle.points = bone_pts
                    bone_handle.colors = bone_cols
                    bone_handle.visible = True
                else:
                    bone_handle.visible = False

            # Update trail
            trail_len = int(trail_slider.value)
            if trail_len > 0 and frame_idx > 0:
                start = max(0, frame_idx - trail_len)
                trail_positions = positions[start : frame_idx + 1]  # (T, M, 3)
                t_len = trail_positions.shape[0]
                alphas = np.linspace(0.1, 0.8, t_len)
                trail_pts_list = []
                trail_cols_list = []
                for t, frame_positions in enumerate(trail_positions):
                    valid = _valid_marker_mask(frame_positions)
                    trail_pts_list.append(frame_positions[valid])
                    trail_cols_list.append((marker_colors[valid] * alphas[t]).astype(np.uint8))
                trail_pts = np.concatenate(trail_pts_list, axis=0).astype(np.float32)
                trail_cols = np.concatenate(trail_cols_list, axis=0).astype(np.uint8)

                if trail_handle is not None:
                    trail_handle.points = trail_pts
                    trail_handle.colors = trail_cols
                    trail_handle.point_size = marker_size_slider.value * 0.5
                    trail_handle.visible = True
                else:
                    trail_handle = server.scene.add_point_cloud(
                        "/trail",
                        points=trail_pts,
                        colors=trail_cols,
                        point_size=marker_size_slider.value * 0.5,
                        point_shape="rounded",
                    )
            elif trail_handle is not None:
                trail_handle.visible = False

        server.flush()
        _update_status()

    # Initial render
    _update_frame(0)
    _update_status()

    print(f"\nViewer running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")

    # --- Main loop ---
    # Decouple display rate (60 fps) from capture rate (often 360 Hz).
    # Compute data frame from wall-clock playback time for correct speed control.
    display_fps = 60.0
    display_dt = 1.0 / display_fps
    playback_time = [0.0]  # seconds into the recording

    try:
        while True:
            t0 = time.time()

            if not paused[0]:
                playback_time[0] += display_dt * speed[0]
                total_duration = n_frames / frame_rate
                if loop[0]:
                    playback_time[0] %= total_duration
                else:
                    if playback_time[0] >= total_duration:
                        playback_time[0] = total_duration - 1.0 / frame_rate
                        paused[0] = True
                        pause_button.label = "Play"
                        pause_button.icon = viser.Icon.PLAYER_PLAY
                f = int(playback_time[0] * frame_rate) % n_frames
                current_frame[0] = f
                frame_slider.value = f
                _update_frame(f)
            else:
                # Still respond to slider scrubbing
                f = int(frame_slider.value)
                if f != current_frame[0]:
                    current_frame[0] = f
                    playback_time[0] = f / frame_rate
                    _update_frame(f)

            elapsed = time.time() - t0
            sleep_time = max(0.0, display_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping viewer...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
