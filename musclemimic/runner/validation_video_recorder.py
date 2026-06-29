"""Validation video recording utilities for training workflows."""

import time

from omegaconf import OmegaConf

from musclemimic.algorithms import PPOJax
from musclemimic.utils import setup_headless_rendering_if_needed
from loco_mujoco.core.stateful_object import StatefulObject
from loco_mujoco.task_factories import TaskFactory


class ValidationVideoRecorder:
    """
    Host-side utility to record short evaluation rollouts during training.

    It reconstructs a standalone env (MJX by default) with headless rendering and
    invokes PPOJax.play_policy(record=True) to write a video using the built-in
    VideoRecorder. Designed to be called from the training logging callback.
    """

    def __init__(
        self, video_dir: str, frequency: int = 10, length: int = 500, deterministic: bool = True
    ):
        """
        Args:
            video_dir: Base directory where videos are written.
            frequency: Record every N validation callbacks.
            length: Number of steps to record per episode.
            deterministic: Use deterministic policy for reproducibility.
        """
        self.video_dir = video_dir
        self.frequency = max(1, int(frequency))
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
        fps = int(round(1.0 / control_dt))
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

    def record_episode(
        self, agent_conf, agent_state, validation_number: int, timestep: int | None = None
    ) -> str | None:
        """
        Record a single short rollout if the frequency condition matches.

        Args:
            agent_conf: PPO agent configuration (contains network and saved config).
            agent_state: PPO agent state; only params and run_stats are used.
            validation_number: Current validation counter (1-based).
            timestep: Global training timestep for naming.

        Returns:
            Path to the recorded video file if available, else None.
        """
        if validation_number % self.frequency != 0:
            return None

        setup_headless_rendering_if_needed()

        # Always use MuJoCo CPU env for evaluation visualization
        use_mujoco = True

        # Build a recording tag.
        time_tag = time.strftime("%Y%m%d_%H%M%S")
        tag = f"validation_{validation_number}_t{timestep if timestep is not None else 0}_{time_tag}"

        # Build the evaluation environment.
        factory = TaskFactory.get_factory_cls(agent_conf.config.experiment.task_factory.name)
        env_params = self._build_env_params(agent_conf, tag)
        raw_task_params = agent_conf.config.experiment.task_factory.params
        if OmegaConf.is_config(raw_task_params):
            task_params = OmegaConf.to_container(raw_task_params, resolve=True)
        else:
            task_params = dict(raw_task_params) if raw_task_params else {}

        # Cap large AMASS datasets for short recordings.
        amass_conf = task_params.get("amass_dataset_conf")
        if isinstance(amass_conf, dict):
            amass_conf = dict(amass_conf)
            amass_conf.setdefault("max_motions", 3)
            task_params["amass_dataset_conf"] = amass_conf

        # Isolate StatefulObject indices for the recorder env.
        saved_instances = StatefulObject._instances.copy()
        StatefulObject._instances.clear()
        env = None
        try:
            print(f"[ValidationVideo] Building eval env for recording (tag={tag})...")
            # Create the recorder environment.
            try:
                env = factory.make(**env_params, **task_params)
            except Exception as e:
                print(f"[ValidationVideo] ERROR: Failed to create environment: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise

            fps = env_params["recorder_params"]["fps"]
            print(f"[ValidationVideo] Eval env ready; starting rollout for {self.length} steps @ {fps} fps")

            # Keep isolation active through the internal reset in play_policy.
            PPOJax.play_policy(
                env,
                agent_conf,
                agent_state,
                n_envs=1,
                n_steps=self.length,
                render=True,  # must be True for recording to emit frames
                record=True,
                deterministic=self.deterministic,
                use_mujoco=use_mujoco,
                wrap_env=True,
                train_state_seed=0,
            )
        finally:
            # Restore global state even on failure.
            if env is not None:
                env.stop()
            StatefulObject._instances = saved_instances

        # Return the recorded video path when available.
        return env.video_file_path if env is not None else None
