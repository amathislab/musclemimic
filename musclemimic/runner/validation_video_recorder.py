"""Validation video recording utilities for training workflows."""

import re
import time
from collections.abc import Mapping, Sequence

from omegaconf import OmegaConf

from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.task_factories import TaskFactory
from musclemimic.algorithms import PPOJax
from musclemimic.utils import setup_headless_rendering_if_needed


def validation_video_due(validation_number: int, frequency: int) -> bool:
    """Return whether a scheduled validation should also record video."""
    return frequency > 0 and validation_number > 0 and validation_number % frequency == 0


class ValidationVideoRecorder:
    """
    Host-side utility to record short evaluation rollouts during training.

    It reconstructs a standalone env (MJX by default) with headless rendering and
    invokes the selected algorithm's play_policy(record=True) to write a video using the built-in
    VideoRecorder. Designed to be called from the training logging callback.
    """

    def __init__(
        self,
        video_dir: str,
        frequency: int = 10,
        length: int = 500,
        deterministic: bool = True,
        named_motions: Sequence[Mapping[str, str] | str] | None = None,
    ):
        """
        Args:
            video_dir: Base directory where videos are written.
            frequency: Record every N validation callbacks.
            length: Number of steps to record per episode.
            deterministic: Use deterministic policy for reproducibility.
            named_motions: Optional ``name``/``path`` mappings or
                ``name=dataset/path`` strings. When supplied, record one
                episode for each pinned motion instead of selecting implicitly
                from the training dataset.
        """
        self.video_dir = video_dir
        self.frequency = max(1, int(frequency))
        self.length = max(1, int(length))
        self.deterministic = deterministic
        self.named_motions = self._normalize_named_motions(named_motions)

    @staticmethod
    def _normalize_named_motions(
        named_motions: Sequence[Mapping[str, str] | str] | None,
    ) -> tuple[tuple[str, str], ...]:
        if not named_motions:
            return ()
        normalized = []
        names = set()
        for index, item in enumerate(named_motions):
            if OmegaConf.is_config(item):
                item = OmegaConf.to_container(item, resolve=True)
            if isinstance(item, str):
                name, separator, path = item.partition("=")
                if not separator:
                    raise ValueError(f"named_motions[{index}] must use name=dataset/path syntax")
                name = name.strip()
                path = path.strip()
            elif isinstance(item, Mapping):
                name = str(item.get("name", "")).strip()
                path = str(item.get("path", "")).strip()
            else:
                raise ValueError(f"named_motions[{index}] must be a mapping or string")
            if not name or not path:
                raise ValueError(f"named_motions[{index}] requires non-empty name and path")
            if name in names:
                raise ValueError(f"named_motions contains duplicate name {name!r}")
            names.add(name)
            normalized.append((name, path))
        return tuple(normalized)

    @staticmethod
    def _safe_tag_component(value: str) -> str:
        component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
        return component or "motion"

    def _build_env_params(self, agent_conf, tag: str) -> dict:
        """Clone training env params and add recording-specific settings."""
        # Copy training env params.
        env_params = dict(agent_conf.config.experiment.env_params)

        # Switch to the MuJoCo CPU env for recording.
        env_name = env_params.get("env_name", "")
        if isinstance(env_name, str) and env_name.startswith("Mjx"):
            env_params["env_name"] = env_name.replace("Mjx", "", 1)
        # Drop MJX-only parameters.
        for k in ("mjx_backend", "num_envs", "nconmax", "njmax"):
            if k in env_params:
                env_params.pop(k, None)

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
        sites = goal_params.get("sites_for_mimic", [])
        if "Bimanual" in env_name:
            env_params["goal_type"] = "GoalBimanualTrajMimicv2"
            if sites:
                env_params["goal_params"]["sites_for_mimic"] = sites
        elif "MyoFullBody" in env_name:
            # Fullbody uses GoalTrajMimicv2.
            env_params["goal_type"] = "GoalTrajMimicv2"
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

    def _build_task_params(self, agent_conf, motion_path: str | None = None) -> dict:
        """Clone task parameters and optionally pin the dataset to one motion."""
        raw_task_params = agent_conf.config.experiment.task_factory.params
        if OmegaConf.is_config(raw_task_params):
            task_params = OmegaConf.to_container(raw_task_params, resolve=True)
        else:
            task_params = dict(raw_task_params) if raw_task_params else {}

        amass_conf = task_params.get("amass_dataset_conf")
        if isinstance(amass_conf, dict):
            amass_conf = dict(amass_conf)
            if motion_path is None:
                # Preserve the historical behavior for unpinned recordings.
                amass_conf.setdefault("max_motions", 3)
            else:
                amass_conf["rel_dataset_path"] = [motion_path]
                amass_conf["dataset_group"] = None
                amass_conf["max_motions"] = 1
            task_params["amass_dataset_conf"] = amass_conf
        elif motion_path is not None:
            raise ValueError("named validation motions require task_factory.params.amass_dataset_conf")
        return task_params

    def _on_environment_reset(self, env, motion_path: str | None) -> None:
        """Inspect a recorder environment at the exact state used for playback.

        Subclasses can override this hook to collect reset diagnostics.  The
        default is deliberately empty so existing recorders retain their
        behavior.  Playback performs the same deterministic reset immediately
        afterwards.
        """

    def _record_one_episode(
        self,
        agent_conf,
        agent_state,
        *,
        tag: str,
        motion_path: str | None,
        algorithm_cls,
    ) -> str | None:
        factory = TaskFactory.get_factory_cls(agent_conf.config.experiment.task_factory.name)
        env_params = self._build_env_params(agent_conf, tag)
        task_params = self._build_task_params(agent_conf, motion_path)

        # Isolate StatefulObject indices for the recorder env.
        saved_instances = StatefulObject._instances.copy()
        StatefulObject._instances.clear()
        env = None
        try:
            motion_description = f", motion={motion_path}" if motion_path is not None else ""
            print(f"[ValidationVideo] Building eval env for recording (tag={tag}{motion_description})...")
            try:
                env = factory.make(**env_params, **task_params)
            except Exception as e:
                print(f"[ValidationVideo] ERROR: Failed to create environment: {type(e).__name__}: {e}")
                import traceback

                traceback.print_exc()
                raise

            fps = env_params["recorder_params"]["fps"]
            print(f"[ValidationVideo] Eval env ready; starting rollout for {self.length} steps @ {fps} fps")

            # Diagnostic subclasses may inspect an equivalent frame-zero state
            # before playback. Avoid the extra reset for ordinary recorders.
            if type(self)._on_environment_reset is not ValidationVideoRecorder._on_environment_reset:
                env.reset()
                self._on_environment_reset(env, motion_path)

            # Keep isolation active through the internal reset in play_policy.
            algorithm_cls.play_policy(
                env,
                agent_conf,
                agent_state,
                n_envs=1,
                n_steps=self.length,
                render=True,  # must be True for recording to emit frames
                record=True,
                deterministic=self.deterministic,
                use_mujoco=True,
                wrap_env=True,
                train_state_seed=0,
                stop_after_first_episode=True,
            )
        finally:
            # Restore global state even on failure.
            if env is not None:
                env.stop()
            StatefulObject._instances = saved_instances

        return env.video_file_path if env is not None else None

    def record_episode(
        self,
        agent_conf,
        agent_state,
        validation_number: int,
        timestep: int | None = None,
        algorithm_cls=None,
    ) -> str | dict[str, str] | None:
        """
        Record one complete validation episode per configured motion if the
        frequency condition matches.

        Args:
            agent_conf: PPO agent configuration (contains network and saved config).
            agent_state: PPO agent state; only params and run_stats are used.
            validation_number: Current validation counter (1-based).
            timestep: Global training timestep for naming.

        Returns:
            One path for legacy single-motion recording, a name-to-path mapping
            for a configured panel, or None when skipped or unsuccessful.
        """
        if validation_number % self.frequency != 0:
            return None
        algorithm_cls = algorithm_cls or PPOJax

        setup_headless_rendering_if_needed()

        # Build a recording tag.
        time_tag = time.strftime("%Y%m%d_%H%M%S")
        base_tag = f"validation_{validation_number}_t{timestep if timestep is not None else 0}_{time_tag}"

        if self.named_motions:
            video_paths = {}
            for name, motion_path in self.named_motions:
                tag = f"{base_tag}_{self._safe_tag_component(name)}"
                try:
                    video_path = self._record_one_episode(
                        agent_conf,
                        agent_state,
                        tag=tag,
                        motion_path=motion_path,
                        algorithm_cls=algorithm_cls,
                    )
                except Exception as error:
                    # A diagnostic panel must never stop training, and one bad
                    # motion should not suppress the remaining videos.
                    print(f"[ValidationVideo] WARNING: {name!r} render failed: {error}")
                    continue
                if video_path:
                    video_paths[name] = video_path
            return video_paths or None

        return self._record_one_episode(
            agent_conf,
            agent_state,
            tag=base_tag,
            motion_path=None,
            algorithm_cls=algorithm_cls,
        )
