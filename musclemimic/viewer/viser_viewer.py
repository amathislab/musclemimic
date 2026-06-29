"""Viser-based policy rollout viewer for MuscleMimic MuJoCo environments.

The viewer streams prebuilt body meshes by updating body poses each frame.
Tendon visualization is delegated to ``ViserTendonLineRenderer``, which uses
MuJoCo's own tendon visualization geoms and renders them as Viser line segments.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .viser_tendons import ViserTendonLineRenderer


class ViserViewer:
    def __init__(
        self,
        env,
        agent_conf,
        agent_state,
        deterministic: bool = True,
        frame_rate: float = 60.0,
        include_collision: bool = False,
    ) -> None:
        """Construct a viewer.

        Args:
            env: MuJoCo environment instance (CPU, not MJX)
            agent_conf: PPOJax agent configuration (network + config)
            agent_state: PPOJax agent state with trained params
            deterministic: If True, set log_std to -inf for deterministic actions
            frame_rate: Target FPS for visualization
            include_collision: If True, display collision geoms instead of visual geoms
        """
        self.env = env
        self.agent_conf = agent_conf
        self.agent_state = agent_state
        self.deterministic = deterministic
        self.frame_rate = frame_rate
        self.include_collision = include_collision

        # Created in _setup_viser(), after the concrete MuJoCo model is known.
        self._server = None
        self._handles = None
        self._tendon_renderer = None
        self._paused = False
        self._time_multiplier = 1.0

    def _prepare_policy(self):
        """Create a jitted policy call matching MuscleMimic eval semantics."""

        def sample_actions(ts, obs, _rng):
            if hasattr(obs, "ndim") and obs.ndim == 1:
                obs_b = jnp.atleast_2d(obs)
            else:
                obs_b = obs
            vars_in = {"params": ts.params, "run_stats": ts.run_stats}
            y, updates = self.agent_conf.network.apply(vars_in, obs_b, mutable=["run_stats"])
            pi, _ = y
            ts_out = ts.replace(run_stats=updates["run_stats"])
            a = pi.sample(seed=_rng)
            if hasattr(a, "ndim") and a.ndim > 1 and a.shape[0] == 1:
                a = a[0]
            return a, ts_out

        train_state = self.agent_state.train_state
        if self.deterministic:
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        # Handle multi-seed agent state if present
        config = self.agent_conf.config.experiment
        if getattr(config, "n_seeds", 1) > 1:
            # Default to seed 0 for viewer
            train_state = jax.tree.map(lambda x: x[0], train_state)

        rng = jax.random.key(0)
        plcy_call = jax.jit(sample_actions)
        return plcy_call, train_state, rng

    def _setup_viser(self, model: mujoco.MjModel):
        try:
            import viser  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Viser is not installed. Install optional extras with: pip install '.[viser]'") from e

        # Mesh construction depends on optional visualization dependencies.
        from .viser_utils import build_body_meshes

        self._server = viser.ViserServer(label="musclemimic")
        self._server.scene.configure_environment_map(environment_intensity=0.8)

        tabs = self._server.gui.add_tab_group()
        with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
            self._status_html = self._server.gui.add_html("")
            with self._server.gui.add_folder("Simulation"):
                self._pause_button = self._server.gui.add_button(
                    "Play" if self._paused else "Pause",
                    icon=viser.Icon.PLAYER_PLAY if self._paused else viser.Icon.PLAYER_PAUSE,
                )

                @self._pause_button.on_click
                def _(_ev) -> None:
                    self._paused = not self._paused
                    self._pause_button.label = "Play" if self._paused else "Pause"
                    self._pause_button.icon = viser.Icon.PLAYER_PLAY if self._paused else viser.Icon.PLAYER_PAUSE
                    self._update_status()

                reset_button = self._server.gui.add_button("Reset Environment")

                @reset_button.on_click
                def _(_ev) -> None:
                    self.env.reset()
                    mujoco.mj_forward(model, self._get_data())
                    self._sync_meshes(model, self._get_data())
                    self._update_status()

                speed_buttons = self._server.gui.add_button_group("Speed", options=["Slower", "Faster"])

                @speed_buttons.on_click
                def _(event) -> None:
                    if event.target.value == "Slower":
                        self._time_multiplier = max(0.1, self._time_multiplier / 2.0)
                    else:
                        self._time_multiplier = min(4.0, self._time_multiplier * 2.0)
                    self._update_status()

            if model.ntendon > 0:
                with self._server.gui.add_folder("Tendons"):
                    tendon_show = self._server.gui.add_checkbox("Show", initial_value=True)

                    @tendon_show.on_update
                    def _(_ev) -> None:
                        if self._tendon_renderer is None:
                            return
                        self._tendon_renderer.set_visible(bool(tendon_show.value))
                        self._tendon_renderer.update(self._get_data())

        self._server.scene.add_grid(
            "/ground",
            width=10.0,
            height=10.0,
            width_segments=20,
            height_segments=20,
            plane="xy",
            cell_color=(180, 180, 180),
            section_color=(120, 120, 120),
            cell_thickness=1,
            section_thickness=2,
        )

        body_meshes = build_body_meshes(model, include_collision=self.include_collision)
        handles = {}
        with self._server.atomic():
            for body_id, mesh in body_meshes.items():
                # Batched handles keep per-frame pose updates as array assignments.
                handle = self._server.scene.add_batched_meshes_trimesh(
                    f"/bodies/{body_id}",
                    mesh,
                    batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]]),
                    batched_positions=np.array([[0.0, 0.0, 0.0]]),
                    lod="off",
                    visible=True,
                )
                handles[body_id] = handle
        self._handles = handles
        if model.ntendon > 0:
            self._tendon_renderer = ViserTendonLineRenderer(model, self._server, path="/tendons")
            self._tendon_renderer.update(self._get_data())
        self._update_status()

    def _get_data(self) -> mujoco.MjData:
        """Return the MuJoCo data object through common environment wrappers."""
        if hasattr(self.env, "data"):
            return self.env.data
        if hasattr(self.env, "env"):
            cur = self.env
            while hasattr(cur, "env"):
                cur = cur.env
                if hasattr(cur, "data"):
                    return cur.data
        raise RuntimeError("Could not access MuJoCo data from environment")

    def _update_status(self):
        if self._server is None:
            return
        status = f"<b>Paused:</b> {self._paused} &nbsp; <b>Speed:</b> {self._time_multiplier:.2f}x"
        self._status_html.content = status

    def _sync_meshes(self, model: mujoco.MjModel, data: mujoco.MjData, update_tendons: bool = True) -> None:
        import numpy as _np

        body_xpos = _np.array(data.xpos)
        body_xquat = _np.array(data.xquat)

        with self._server.atomic():
            for body_id, handle in self._handles.items():
                if body_id >= len(body_xpos):
                    continue
                pos = body_xpos[body_id]
                quat = body_xquat[body_id]  # wxyz
                handle.batched_positions = _np.array([pos], dtype=float)
                handle.batched_wxyzs = _np.array([quat], dtype=float)
            if update_tendons and self._tendon_renderer is not None:
                self._tendon_renderer.update(data)
        self._server.flush()

    def _sync_tendons(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model
        if self._tendon_renderer is None:
            return
        self._tendon_renderer.update(data)

    def run(self, n_steps: int | None = None) -> None:
        """Main loop: sample actions, step env, and stream transforms to Viser."""
        model = self.env.model if hasattr(self.env, "model") else self.env.env.model
        data = self._get_data()

        plcy_call, train_state, rng = self._prepare_policy()

        obs = self.env.reset()
        mujoco.mj_forward(model, data)

        self._setup_viser(model)
        self._sync_meshes(model, data)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        target_dt = 1.0 / float(self.frame_rate)

        step_count = 0
        try:
            while step_count < n_steps:
                t0 = time.time()

                if not self._paused:
                    rng, _rng = jax.random.split(rng)
                    action, train_state = plcy_call(train_state, obs, _rng)
                    action = jnp.atleast_2d(action)
                    obs, _r, _abs, done, _info = self.env.step(action)

                    mujoco.mj_forward(model, data)
                    self._sync_meshes(model, data)

                    if done:
                        obs = self.env.reset()
                        mujoco.mj_forward(model, data)
                        self._sync_meshes(model, data)

                    step_count += 1

                elapsed = time.time() - t0
                to_sleep = max(0.0, (target_dt / self._time_multiplier) - elapsed)
                if to_sleep > 0:
                    time.sleep(to_sleep)

        except KeyboardInterrupt:
            pass
        finally:
            try:
                if self._server is not None:
                    self._server.stop()
            except Exception:
                pass
