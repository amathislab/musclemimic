from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf, open_dict

from musclemimic.algorithms.common.checkpoint_manager import UnifiedCheckpointManager
from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.algorithms.common.inference import (
    initialize_policy_state,
    make_policy_action_fn,
)
from musclemimic.algorithms.ppo.checkpoint import create_agent_state_from_orbax
from musclemimic.algorithms.ppo.inference import (
    ObservationHistoryBuffer,
)
from musclemimic.algorithms.ppo.runner import _run_validation
from musclemimic.runner.engine import build_metrics_handler, instantiate_validation_env
from musclemimic.utils import detect_headless_environment, setup_headless_rendering_if_needed
from musclemimic.utils.metrics import VALIDATION_STEP_METRIC_KEYS, flatten_validation_metrics


def add_common_eval_args(parser: argparse.ArgumentParser, default_n_envs: int) -> None:
    parser.add_argument("--path", type=str, required=True, help="Path to the agent pkl file")
    parser.add_argument("--use_mujoco", action="store_true", help="Use MuJoCo for evaluation instead of Mjx")
    parser.add_argument(
        "--mujoco_viewer",
        action="store_true",
        help="Use default MuJoCo viewer with control overlays (requires --use_mujoco)",
    )
    parser.add_argument(
        "--viser_viewer",
        action="store_true",
        help="Use Viser web-based 3D viewer (requires --use_mujoco)",
    )
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of evaluation steps (default: 1000)")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=default_n_envs,
        help="Number of MJX environments for rollouts (ignored for MuJoCo)",
    )
    parser.add_argument("--no_render", action="store_true", help="Disable rendering (headless mode)")
    parser.add_argument(
        "--strict_termination",
        action="store_true",
        help="Use training termination thresholds (may cause early termination)",
    )
    parser.add_argument("--record", action="store_true", help="Record rollout videos during evaluation")
    parser.add_argument(
        "--record_dir",
        type=str,
        default="./eval_recordings",
        help="Directory to save recorded videos (default: ./eval_recordings)",
    )
    parser.add_argument(
        "--export_trajectory",
        action="store_true",
        help="Export trajectory data as NPZ file (joint angles, velocities, accelerations, site positions)",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default="./trajectory_data",
        help="Directory to save trajectory NPZ files (default: ./trajectory_data)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy evaluation (default: deterministic)",
    )
    parser.add_argument(
        "--train_state_seed",
        type=int,
        default=0,
        help="Seed index for multi-seed checkpoints (default: 0)",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=0,
        help="RNG seed for stochastic action sampling during evaluation (default: 0)",
    )

    parser.add_argument(
        "--traj_index",
        type=int,
        default=None,
        help="Force evaluation on a specific trajectory index (0-based)",
    )
    parser.add_argument(
        "--traj_start_step",
        type=int,
        default=None,
        help="Start step within the trajectory (requires --traj_index)",
    )
    parser.add_argument(
        "--list_trajs",
        action="store_true",
        help="List available trajectories (indices and lengths) and exit",
    )
    parser.add_argument(
        "--list_trajs_limit",
        type=int,
        default=50,
        help="Max trajectories to show with --list_trajs (default: 50)",
    )

    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Compute validation metrics using training validation settings",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Only compute metrics (skip policy playback/rendering)",
    )
    parser.add_argument(
        "--metrics_steps",
        type=int,
        default=None,
        help="Override validation num_steps for metrics",
    )
    parser.add_argument(
        "--metrics_envs",
        type=int,
        default=8,
        help="Override validation num_envs for metrics",
    )
    metrics_group = parser.add_mutually_exclusive_group()
    metrics_group.add_argument(
        "--metrics_deterministic",
        action="store_true",
        help="Force deterministic policy for metrics evaluation",
    )
    metrics_group.add_argument(
        "--metrics_stochastic",
        action="store_true",
        help="Force stochastic policy for metrics evaluation",
    )


def normalize_eval_args(args: argparse.Namespace) -> None:
    if args.metrics_only:
        args.metrics = True
    if args.traj_start_step is not None and args.traj_index is None:
        raise ValueError("--traj_start_step requires --traj_index")


def apply_configured_evaluation_backend(args: argparse.Namespace, config) -> bool:
    """Apply a checkpoint's evaluation backend, returning whether it changed the CLI."""
    backend = config.experiment.get("evaluation_backend", None)
    if backend is None:
        return False
    if backend != "mujoco":
        raise ValueError(f"Unsupported evaluation_backend: {backend!r}")

    changed = not args.use_mujoco
    args.use_mujoco = True
    return changed


def validate_viewer_args(args: argparse.Namespace) -> str | None:
    if args.mujoco_viewer and not args.use_mujoco:
        return "Error: --mujoco_viewer requires --use_mujoco"
    if args.mujoco_viewer and args.no_render:
        return "Error: --mujoco_viewer is incompatible with --no_render"
    if args.viser_viewer and not args.use_mujoco:
        return "Error: --viser_viewer currently supports only --use_mujoco"
    if args.viser_viewer and args.mujoco_viewer:
        return "Error: --viser_viewer is incompatible with --mujoco_viewer"
    return None


def setup_headless(args: argparse.Namespace) -> tuple[bool, str | None]:
    is_headless = detect_headless_environment()
    if is_headless and not args.no_render and not args.mujoco_viewer and not args.viser_viewer:
        setup_headless_rendering_if_needed()
        args.no_render = True
        print("   Automatically enabling --no_render for headless environment")
    elif is_headless and args.mujoco_viewer:
        return True, "Error: MuJoCo viewer requested but no display available"
    return is_headless, None


def load_checkpoint(path: str):
    """Load checkpoint from local path or HuggingFace.

    Supports:
        - Local path: /path/to/checkpoint_30000
        - Local parent dir: /path/to/checkpoints/ (picks latest)
        - HuggingFace: hf://username/repo-name
    """
    # Handle HuggingFace URLs and canonicalize paths
    from musclemimic.runner.checkpointing import _canonicalize_resume_path

    resolved_path = _canonicalize_resume_path(path)

    checkpoint_dir = os.path.dirname(resolved_path) if os.path.isdir(resolved_path) else os.path.dirname(os.path.abspath(resolved_path))
    manager = UnifiedCheckpointManager(checkpoint_dir, max_to_keep=5)
    try:
        (loaded_conf, loaded_state), metadata = manager.load_checkpoint(resolved_path)
    finally:
        manager.close()
    config = OmegaConf.create(loaded_conf)
    agent_state = SimpleNamespace(train_state=loaded_state)
    return config, agent_state, metadata


def apply_temporal_params(config) -> float:
    training_timestep = config.experiment.env_params.get("timestep", 0.002)
    training_n_substeps = config.experiment.env_params.get("n_substeps", 5)
    training_control_dt = training_timestep * training_n_substeps
    config.experiment.env_params["timestep"] = training_timestep
    config.experiment.env_params["n_substeps"] = training_n_substeps
    return training_control_dt


def configure_goal_visualization(config, args, goal_type_v2: str, is_mjx_env: bool) -> None:
    env_params = config.experiment.env_params
    if "goal_params" not in env_params:
        return
    goal_params = env_params["goal_params"]
    goal_params["visualize_goal"] = not args.no_render or args.record
    if "enable_enhanced_visualization" not in goal_params:
        goal_params["enable_enhanced_visualization"] = True
    if "target_geom_rgba" not in goal_params:
        goal_params["target_geom_rgba"] = [0.471, 0.38, 0.812, 0.6]

    original_sites = env_params.get("goal_params", {}).get("sites_for_mimic", [])

    if not args.no_render or args.mujoco_viewer or args.record:
        if is_mjx_env and not args.use_mujoco:
            print(f"MJX backend: forcing enhanced {goal_type_v2} for ghost robot")
        else:
            print("   Enabling enhanced ghost robot visualization")
        env_params["goal_type"] = goal_type_v2
        if original_sites:
            print(f"   Preserving training sites_for_mimic: {original_sites}")
            env_params["goal_params"]["sites_for_mimic"] = original_sites
        else:
            print("   Using default sites_for_mimic (no custom sites in training config)")
    else:
        print("   Headless mode: using training goal class for efficiency")


def apply_trajectory_selection(config, traj_index: int | None, traj_start_step: int | None) -> None:
    if traj_index is None:
        return
    if traj_start_step is None:
        traj_start_step = 0
    th_params = dict(config.experiment.env_params.get("th_params", {}))
    th_params["random_start"] = False
    th_params["fixed_start_conf"] = [int(traj_index), int(traj_start_step)]
    config.experiment.env_params["th_params"] = th_params


def configure_recording(
    env_params: dict, args: argparse.Namespace, control_dt: float, motion_name: str | None = None
) -> str:
    """Configure recording parameters. Returns the recording tag (folder name)."""
    if not args.record:
        return ""
    tag = motion_name if motion_name else "evaluation_recording"
    fps = int(round(1.0 / control_dt))
    record_params = {
        "path": args.record_dir,
        "tag": tag,
        "video_name": f"eval_{env_params.get('env_name', 'unknown')}",
        "fps": fps,
        "compress": True,
    }
    env_params["recorder_params"] = record_params
    env_params["headless"] = bool(args.no_render)
    return tag


def align_agent_state(agent_state, agent_conf):
    """Restore the checkpoint state using the selected algorithm's typed state."""
    from musclemimic.algorithms.flashsac.checkpoint import restore_train_state
    from musclemimic.algorithms.flashsac.config import FlashSACAgentConf
    from musclemimic.algorithms.flashsac.state import FlashSACAgentState
    from musclemimic.algorithms.ppo.config import PPOAgentConf, PPOAgentState

    state_data = agent_state.train_state
    if isinstance(agent_conf, FlashSACAgentConf):
        return FlashSACAgentState(
            train_state=restore_train_state(state_data, agent_conf)
        )
    if isinstance(agent_conf, PPOAgentConf):
        restored = create_agent_state_from_orbax(state_data)
        return PPOAgentState(
            train_state=restored.train_state.replace(
                apply_fn=agent_conf.network.apply,
                tx=agent_conf.tx,
            )
        )
    raise TypeError(f"Unsupported agent configuration: {type(agent_conf).__name__}")


def verify_env_dt(env, expected_dt: float) -> None:
    actual_dt = env.dt
    print("\n=== ENVIRONMENT VERIFICATION ===")
    print(f"   Expected control_dt: {expected_dt:.6f}")
    print(f"   Actual environment dt: {actual_dt:.6f}")
    if abs(actual_dt - expected_dt) > 1e-6:
        print(f"WARNING: Environment dt mismatch! Expected {expected_dt:.6f}, got {actual_dt:.6f}")
        print("This may cause ghost robot synchronization issues")
    else:
        print("Environment temporal parameters correctly configured")


def check_trajectory_sync(env) -> None:
    if not hasattr(env, "th") or env.th is None:
        return
    traj_freq = env.th.traj.info.frequency
    required_freq = 1.0 / env.dt
    sync_ratio = traj_freq / required_freq
    print(f"   Trajectory frequency: {traj_freq:.1f} Hz")
    print(f"   Required frequency: {required_freq:.1f} Hz")
    print(f"   Sync ratio: {sync_ratio:.3f}")
    if abs(sync_ratio - 1.0) > 0.01:
        print("WARNING: Trajectory-policy frequency mismatch!")
        print(f"Ghost robot will move at {sync_ratio:.2f}x speed relative to policy")
    else:
        print("Trajectory synchronized with policy - ghost robot will move correctly")


def validate_traj_index(env, traj_index: int | None) -> None:
    if traj_index is None:
        return
    if env.th is None:
        raise ValueError("Trajectory index specified but environment has no trajectory handler")
    n_traj = int(env.th.n_trajectories)
    if traj_index < 0 or traj_index >= n_traj:
        raise ValueError(f"Trajectory index {traj_index} out of range (0-{n_traj - 1})")


def format_trajectory_listing(env, limit: int | None) -> list[str]:
    if env.th is None:
        return ["No trajectory handler attached to environment."]
    n_traj = int(env.th.n_trajectories)
    freq = float(env.th.traj.info.frequency)
    lines = [f"Trajectories: {n_traj} (frequency: {freq:.1f} Hz)"]
    if limit is None or limit <= 0:
        limit = n_traj
    show = min(n_traj, limit)
    for i in range(show):
        length = int(env.th.len_trajectory(i))
        duration = length * float(env.dt)
        lines.append(f"  {i}: {length} steps ({duration:.2f}s)")
    if show < n_traj:
        lines.append(f"... {n_traj - show} more trajectories not shown (use --list_trajs_limit)")
    return lines


def run_validation_metrics(
    config,
    agent_state,
    agent_conf,
    train_state_seed: int | None,
    num_steps: int | None = None,
    num_envs: int | None = None,
    deterministic_override: bool | None = None,
    eval_seed: int = 0,
):
    validation_cfg = config.experiment.get("validation", None)
    if validation_cfg is None:
        raise ValueError("Validation config missing; cannot compute metrics.")
    original_deterministic = bool(validation_cfg.get("deterministic", False))
    with open_dict(config.experiment.validation):
        config.experiment.validation.active = True
        if num_steps is not None:
            config.experiment.validation.num_steps = int(num_steps)
        if num_envs is not None:
            config.experiment.validation.num_envs = int(num_envs)
        if deterministic_override is not None:
            config.experiment.validation.deterministic = bool(deterministic_override)

    val_env = instantiate_validation_env(config)
    if val_env is None:
        raise ValueError("Validation environment could not be created.")
    val_env = wrap_env(val_env, config.experiment)
    mh = build_metrics_handler(config, val_env)
    if mh is None:
        raise ValueError("Metrics handler not available; check validation settings in config.")

    train_state = agent_state.train_state
    exp_cfg = config.experiment

    if exp_cfg.n_seeds > 1:
        if train_state_seed is None:
            raise ValueError("Loaded train state has multiple seeds; specify --train_state_seed.")
        train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

    rng = jax.random.key(eval_seed)

    from musclemimic.algorithms.flashsac.config import FlashSACAgentConf

    if isinstance(agent_conf, FlashSACAgentConf):
        from musclemimic.algorithms.flashsac.validation import run_validation

        validation_metrics, _ = run_validation(
            agent_conf,
            train_state,
            rng,
            val_env,
            config.experiment,
            mh,
            jnp.asarray(config.experiment.validation_interval),
        )
    else:
        validation_metrics, _ = _run_validation(
            train_state=train_state,
            val_rng=rng,
            val_env=val_env,
            config=config.experiment,
            mh=mh,
            counter=jnp.asarray(0),
            network=agent_conf.network,
        )
    if deterministic_override is not None:
        with open_dict(config.experiment.validation):
            config.experiment.validation.deterministic = original_deterministic
    enabled_measures = getattr(config.experiment.validation, 'measures', None)
    enabled_quantities = getattr(config.experiment.validation, 'quantities', None)
    return validation_metrics, flatten_validation_metrics(validation_metrics, enabled_measures, enabled_quantities)


def run_validation_metrics_mjx_all(
    env,
    agent_conf,
    agent_state,
    deterministic: bool = True,
    train_state_seed: int | None = None,
    num_envs: int | None = None,
    eval_seed: int = 0,
) -> dict[str, float]:
    """Evaluate every trajectory with the MJX GPU path."""
    from musclemimic.core.wrappers import LogWrapper, NStepWrapper, VecEnv

    config = agent_conf.config.experiment
    train_state = agent_state.train_state

    if config.n_seeds > 1:
        if train_state_seed is None:
            raise ValueError("Loaded train state has multiple seeds; specify --train_state_seed.")
        train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

    # Ensure trajectory data is on JAX backend
    if hasattr(env, "th") and env.th is not None and env.th.is_numpy:
        env.th.to_jax()

    # Raise horizon so long trajectories aren't truncated by _mjx_is_done
    max_traj_len = max(int(env.th.len_trajectory(i)) for i in range(env.th.n_trajectories))
    if env.info.horizon < max_traj_len:
        env._mdp_info.horizon = max_traj_len

    n_traj = int(env.th.n_trajectories)
    val_cfg = config.get("validation", {})
    if num_envs is None:
        num_envs = int(val_cfg.get("num_envs", config.get("num_envs", 64)))
    else:
        num_envs = int(num_envs)
    num_envs = max(1, min(num_envs, n_traj))

    inner_env = env
    if "len_obs_history" in config and config.len_obs_history > 1:
        split_goal = config.get("split_goal", False)
        inner_env = NStepWrapper(inner_env, config.len_obs_history, split_goal=split_goal)

    val_env = LogWrapper(VecEnv(inner_env))

    print(f"MJX evaluate_all: {env.th.n_trajectories} trajectories, "
          f"batch_size={num_envs}, max_traj_len={max_traj_len}")

    from musclemimic.algorithms.flashsac.config import FlashSACAgentConf

    if isinstance(agent_conf, FlashSACAgentConf):
        from musclemimic.algorithms.flashsac.validation import run_validation_all

        metrics = run_validation_all(
            agent_conf=agent_conf,
            train_state=train_state,
            trajectory_env=inner_env,
            env=val_env,
            num_envs=num_envs,
            deterministic=deterministic,
            eval_seed=eval_seed,
        )
    else:
        from musclemimic.algorithms.ppo.runner import _run_validation_all

        metrics = _run_validation_all(
            network=agent_conf.network,
            params=train_state.params,
            run_stats=train_state.run_stats,
            traj_env=inner_env,
            val_env=val_env,
            num_envs=num_envs,
            deterministic=deterministic,
            eval_seed=eval_seed,
        )
    return metrics


def _prepare_cpu_evaluate_all(env) -> int:
    env.th.random_start = False
    env.th.use_fixed_start = True
    env.th.start_from_random_step = False
    return int(env.th.n_trajectories)


def _reset_cpu_validation_episode(env, traj_idx: int | None):
    if traj_idx is not None:
        env.th.fixed_start_conf = [traj_idx, 0]
    return env.reset()


def _accumulate_cpu_step_metrics(step_metric_sums, step_metric_counts, info) -> None:
    if not isinstance(info, dict):
        return
    for key in VALIDATION_STEP_METRIC_KEYS:
        if key in info:
            step_metric_sums[key] += float(info[key])
            step_metric_counts[key] += 1


def run_validation_metrics_mujoco(
    env,
    agent_conf,
    agent_state,
    num_steps: int = 500,
    deterministic: bool = True,
    evaluate_all: bool = False,
    train_state_seed: int | None = None,
    eval_seed: int = 0,
) -> dict[str, float]:
    """
    Run validation metrics collection using MuJoCo CPU backend.
    This is a simpler version that doesn't use JAX scan, compatible with MuJoCo CPU.

    Returns a dict of metrics similar to flatten_validation_metrics output.
    """
    config = agent_conf.config.experiment
    train_state = agent_state.train_state

    if config.n_seeds > 1:
        if train_state_seed is None:
            raise ValueError("Loaded train state has multiple seeds; specify train_state_seed.")
        train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

    policy_fn = make_policy_action_fn(agent_conf, deterministic=deterministic)
    rng = jax.random.key(eval_seed)

    step_metric_sums = {key: 0.0 for key in VALIDATION_STEP_METRIC_KEYS}
    step_metric_counts = {key: 0 for key in VALIDATION_STEP_METRIC_KEYS}

    early_termination_count = 0
    total_episodes = 0
    total_episode_return = 0.0
    total_episode_length = 0.0
    total_frame = 0.0
    covered_frame = 0.0

    if evaluate_all:
        num_episodes = _prepare_cpu_evaluate_all(env)
        episode_traj_indices = range(num_episodes)
    else:
        num_episodes = num_steps
        episode_traj_indices = [None] * num_episodes

    print(f"Running MuJoCo validation for {num_episodes} episodes...")

    for traj_idx in episode_traj_indices:
        obs = _reset_cpu_validation_episode(env, traj_idx)
        policy_state, policy_reset = initialize_policy_state(agent_conf, obs)

        episode_return = 0.0
        episode_length = 0

        while True:
            rng, _rng = jax.random.split(rng)
            action, train_state, policy_state = policy_fn(
                train_state, obs, _rng, policy_state, policy_reset
            )
            action = jnp.atleast_2d(action)

            obs, reward, absorbing, done, info = env.step(action)

            episode_return += float(reward)
            episode_length += 1

            _accumulate_cpu_step_metrics(step_metric_sums, step_metric_counts, info)

            if done:
                traj_no = int(info["traj_no"])
                print(f"finished traj {traj_no} | len={episode_length} | return={episode_return:.6f}", flush=True)

                total_episodes += 1
                total_episode_return += episode_return
                total_episode_length += episode_length

                subtraj_step = int(info.get("subtraj_step_no", 0))
                traj_len = int(info.get("traj_len", 1))

                total_frame += traj_len
                covered_frame += subtraj_step + 1
                if subtraj_step < traj_len - 1:
                    early_termination_count += 1
                    print(f"  EARLY termination at {subtraj_step}/{traj_len}", flush=True)

                break

    # Compute summary metrics
    metrics = {}

    # Episode-level metrics
    if total_episodes > 0:
        metrics["val_mean_episode_return"] = float(total_episode_return / total_episodes)
        metrics["val_mean_episode_length"] = float(total_episode_length / total_episodes)
        metrics["val_early_termination_count"] = float(early_termination_count)
        metrics["val_early_termination_rate"] = float(early_termination_count / total_episodes)
        metrics["val_frame_coverage"] = float(covered_frame / total_frame) if total_frame > 0 else 0.0
        metrics["val_total_frame"] = float(total_frame) if total_frame > 0 else 0.0
    else:
        metrics["val_mean_episode_return"] = 0.0
        metrics["val_mean_episode_length"] = float(num_steps)
        metrics["val_early_termination_count"] = 0.0
        metrics["val_early_termination_rate"] = 0.0

    # Step-level metrics (mean over all steps)
    for key in VALIDATION_STEP_METRIC_KEYS:
        count = step_metric_counts[key]
        metrics[f"val_{key}"] = float(step_metric_sums[key] / count) if count > 0 else 0.0

    print(f"Completed {total_episodes} episodes, {early_termination_count} early terminations")
    return metrics


def run_with_mujoco_viewer(env, agent_conf, agent_state, n_steps=None, deterministic=False, train_state_seed=None):
    """
    Run evaluation using the default MuJoCo viewer with control signal overlays.
    This provides a richer visualization experience with force/torque displays.
    """
    print("Using default MuJoCo viewer with control signal overlays...")

    config = agent_conf.config.experiment
    train_state = agent_state.train_state

    if config.n_seeds > 1:
        assert train_state_seed is not None, (
            "Loaded train state has multiple seeds. Please specify train_state_seed for replay."
        )
        train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

    # Create observation history buffer if needed
    len_obs_history = getattr(config, "len_obs_history", 1)
    split_goal = getattr(config, "split_goal", False)
    state_indices = None
    goal_indices = None
    if split_goal:
        if not hasattr(env, "obs_container"):
            raise ValueError("split_goal=True requires env.obs_container with goal group indices")
        goal_indices = env.obs_container.get_obs_ind_by_group("goal")
        if goal_indices.size == 0:
            raise ValueError("split_goal=True requires goal observations grouped as 'goal'")
        raw_obs_dim = env.info.observation_space.shape[0]
        goal_indices = np.asarray(goal_indices, dtype=int)
        state_mask = np.ones(raw_obs_dim, dtype=bool)
        state_mask[goal_indices] = False
        state_indices = np.arange(raw_obs_dim, dtype=int)[state_mask]

    obs_buffer = (
        ObservationHistoryBuffer(
            len_obs_history,
            split_goal=split_goal,
            state_indices=state_indices,
            goal_indices=goal_indices,
        )
        if len_obs_history > 1
        else None
    )

    rng = jax.random.key(0)
    plcy_call = make_policy_action_fn(agent_conf, deterministic=deterministic)

    obs = env.reset()
    if obs_buffer is not None:
        obs = obs_buffer.reset(obs)
    policy_state, policy_reset = initialize_policy_state(agent_conf, obs)

    if n_steps is None:
        n_steps = np.iinfo(np.int32).max

    if hasattr(env, "model") and hasattr(env, "data"):
        model = env.model
        data = env.data
    elif hasattr(env, "env") and hasattr(env.env, "model") and hasattr(env.env, "data"):
        model = env.env.model
        data = env.env.data
    else:
        curr_env = env
        while hasattr(curr_env, "env"):
            curr_env = curr_env.env
            if hasattr(curr_env, "model") and hasattr(curr_env, "data"):
                model = curr_env.model
                data = curr_env.data
                break
        else:
            raise RuntimeError("Could not access MuJoCo model and data from environment")

    print(f"Found MuJoCo model with {model.nq} DoF, {model.nu} actuators, {model.ntendon} tendons")

    has_ghost_robot = hasattr(env, "_goal") and hasattr(env._goal, "_n_visual_geoms") and env._goal._n_visual_geoms > 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo viewer launched. Press ESC to exit.")
        if has_ghost_robot:
            print("Ghost robot visualization enabled - purple reference trajectory visible")
        else:
            print("Ghost robot not available - consider using appropriate goal type")
        print("Viewer features available:")
        print("- Right-click and drag to rotate view")
        print("- Scroll to zoom")
        print("- Double-click on bodies to select and view details")
        print("- Press 'F' to toggle force visualization")
        print("- Press 'T' to toggle torque visualization")
        print("- Press 'C' to toggle contact forces")
        print("- Press 'H' to show all keyboard shortcuts")

        step_count = 0

        while viewer.is_running() and step_count < n_steps:
            rng, _rng = jax.random.split(rng)
            action, train_state, policy_state = plcy_call(
                train_state, obs, _rng, policy_state, policy_reset
            )
            action = jnp.atleast_2d(action)

            obs, reward, absorbing, done, info = env.step(action)
            policy_reset = jnp.asarray([bool(done)], dtype=bool)
            if obs_buffer is not None:
                obs = obs_buffer.step(obs)

            viewer.sync()

            step_count += 1

            if done:
                print(f"Episode ended at step {step_count}, resetting...")
                obs = env.reset()
                if obs_buffer is not None:
                    obs = obs_buffer.reset(obs)
                policy_state, policy_reset = initialize_policy_state(agent_conf, obs)

            time.sleep(0.001)

        print(f"Evaluation completed after {step_count} steps")


def run_with_trajectory_export(
    env,
    agent_conf,
    agent_state,
    n_steps=None,
    deterministic=False,
    use_mujoco=False,
    train_state_seed=None,
    trajectory_dir="./trajectory_data",
    file_prefix="episodes",
    evaluate_all=False,
    evaluate_all_passes=1,
    motion_group=None,
):
    """
    Run evaluation and export trajectory data for individual episodes in a single NPZ file.

    When ``evaluate_all`` is True, the exporter walks the entire dataset in order,
    running one episode per trajectory (traj 0, 1, ..., n_trajectories-1) starting
    from the beginning of each. ``evaluate_all_passes`` repeats that full sweep that
    many times (default 1); with stochastic action sampling each pass yields a
    different rollout, which is useful for measuring variability. Otherwise trajectory
    selection follows the environment's trajectory-handler settings (e.g. random start).

    ``motion_group`` (the dataset split the rollouts were run over, e.g.
    ``GAIT120_TESTING_SUBJECTS_TRIALS0104``) is only used to name the output: when set, a
    second copy of the NPZ is written with the motion-group name concatenated into the
    filename, so rollouts of the same policy over different motion sets don't overwrite each
    other and can be told apart. The un-tagged file is always written too.
    """
    print("Running evaluation with multi-episode trajectory export...")

    os.makedirs(trajectory_dir, exist_ok=True)

    config = agent_conf.config.experiment
    train_state = agent_state.train_state

    if config.n_seeds > 1:
        assert train_state_seed is not None, (
            "Loaded train state has multiple seeds. Please specify train_state_seed for replay."
        )
        train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

    # Create observation history buffer if needed
    len_obs_history = getattr(config, "len_obs_history", 1)
    split_goal = getattr(config, "split_goal", False)
    state_indices = None
    goal_indices = None
    if split_goal:
        if not hasattr(env, "obs_container"):
            raise ValueError("split_goal=True requires env.obs_container with goal group indices")
        goal_indices = env.obs_container.get_obs_ind_by_group("goal")
        if goal_indices.size == 0:
            raise ValueError("split_goal=True requires goal observations grouped as 'goal'")
        raw_obs_dim = env.info.observation_space.shape[0]
        goal_indices = np.asarray(goal_indices, dtype=int)
        state_mask = np.ones(raw_obs_dim, dtype=bool)
        state_mask[goal_indices] = False
        state_indices = np.arange(raw_obs_dim, dtype=int)[state_mask]

    obs_buffer = (
        ObservationHistoryBuffer(
            len_obs_history,
            split_goal=split_goal,
            state_indices=state_indices,
            goal_indices=goal_indices,
        )
        if len_obs_history > 1
        else None
    )

    rng = jax.random.key(0)
    plcy_call = make_policy_action_fn(agent_conf, deterministic=deterministic)

    env_data = None
    if hasattr(env, "data"):
        env_data = env.data
    elif hasattr(env, "env") and hasattr(env.env, "data"):
        env_data = env.env.data
    else:
        curr_env = env
        while hasattr(curr_env, "env"):
            curr_env = curr_env.env
            if hasattr(curr_env, "data"):
                env_data = curr_env.data
                break

    if env_data is None:
        raise RuntimeError("Cannot access environment data for trajectory export.")

    model = None
    if hasattr(env, "model"):
        model = env.model
    elif hasattr(env, "env") and hasattr(env.env, "model"):
        model = env.env.model
    else:
        curr_env = env
        while hasattr(curr_env, "env"):
            curr_env = curr_env.env
            if hasattr(curr_env, "model"):
                model = curr_env.model
                break

    if model is None:
        raise RuntimeError("Cannot access MuJoCo model from environment.")

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    joint_names = [name for name in joint_names if name is not None]

    if n_steps is None:
        n_steps = np.iinfo(np.int32).max

    # When evaluating the whole dataset, iterate trajectories sequentially
    # (traj 0..n-1) starting from the beginning of each, one episode per traj,
    # repeating the full sweep `evaluate_all_passes` times.
    num_trajectories = None
    num_evaluate_all_episodes = None
    if evaluate_all:
        num_trajectories = _prepare_cpu_evaluate_all(env)
        n_passes = max(1, int(evaluate_all_passes))
        num_evaluate_all_episodes = num_trajectories * n_passes
        # Ignore the global step budget in evaluate-all mode: each trajectory runs
        # to completion and the sweep ends once every trajectory has been covered.
        n_steps = np.iinfo(np.int32).max
        print(f"Evaluate-all: iterating {num_trajectories} trajectories sequentially from the beginning, "
              f"{n_passes} pass(es) ({num_evaluate_all_episodes} episodes total)")

    print(f"Collecting episodes for {len(joint_names)} joints over {n_steps} total steps")
    print(f"Environment dt: {env.dt:.6f}")

    all_episodes = {}
    episode_metadata = []

    total_step_count = 0
    episode_id = 0
    start_time = time.time()

    touch_indexes = []
    for name, obs in env.obs_container.items():
        if name in ["touch_r_foot", "touch_r_toes", "touch_l_foot", "touch_l_toes"]:
            touch_indexes.append(obs.obs_ind)

    while total_step_count < n_steps:
        # Stop once every trajectory has been evaluated the requested number of passes.
        if num_evaluate_all_episodes is not None and episode_id >= num_evaluate_all_episodes:
            break

        print(f"\n--- Starting Episode {episode_id} ---")

        if num_trajectories is not None:
            # Wrap the trajectory index so each pass replays traj 0..n-1 in order.
            obs = _reset_cpu_validation_episode(env, episode_id % num_trajectories)
        else:
            obs = env.reset()
        if obs_buffer is not None:
            obs = obs_buffer.reset(obs)
        policy_state, policy_reset = initialize_policy_state(agent_conf, obs)

        print(f"Trajectory {env.th.new_traj_no}")
        print(f"Starting from {env.th.new_subtraj_step_no_init}/ {env.th.len_trajectory(env.th.new_traj_no) - 1}")
        episode_traj_qpos = []
        for t in range(env.th.len_trajectory(env.th.new_traj_no)):
            episode_traj_qpos.append(env.th.traj.data.get(env.th.new_traj_no, t, np).qpos)

        episode_traj_id = env.th.new_traj_no
        episode_traj_length = env.th.len_trajectory(episode_traj_id)

        # episode_observations = []
        episode_touch_observations = []
        episode_policy_actions = []
        episode_muscle_commands = []
        episode_muscle_activations = []
        episode_rewards = []
        episode_joint_positions = []
        episode_joint_velocities = []
        episode_joint_accelerations = []
        episode_timesteps = []

        episode_step_count = 0
        episode_reward = 0.0
        episode_start_time = time.time()

        while total_step_count < n_steps:
            rng, _rng = jax.random.split(rng)
            action, train_state, policy_state = plcy_call(
                train_state, obs, _rng, policy_state, policy_reset
            )
            action = jnp.atleast_2d(action)

            # episode_observations.append(np.array(obs))
            episode_touch_observations.append(np.array(obs)[touch_indexes].flatten())
            episode_policy_actions.append(np.array(action).flatten())
            episode_joint_positions.append(np.array(env_data.qpos))
            episode_joint_velocities.append(np.array(env_data.qvel))
            episode_joint_accelerations.append(np.array(env_data.qacc))
            episode_timesteps.append(episode_step_count * env.dt)

            obs, reward, absorbing, done, info = env.step(action)
            policy_reset = jnp.asarray([bool(done)], dtype=bool)
            if obs_buffer is not None:
                obs = obs_buffer.step(obs)

            episode_muscle_commands.append(np.array(env_data.ctrl))
            episode_muscle_activations.append(np.array(env_data.act))

            episode_rewards.append(float(reward))
            episode_reward += float(reward)
            episode_step_count += 1
            total_step_count += 1

            if done:
                print(f"  Episode {episode_id} completed: {episode_step_count} steps, reward: {episode_reward:.3f}")
                break

        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        episode_fps = episode_step_count / episode_duration if episode_duration > 0 else 0

        episode_prefix = f"episode_{episode_id}"
        all_episodes[f"{episode_prefix}_traj_id"] = np.array(episode_traj_id)
        all_episodes[f"{episode_prefix}_traj_length"] = np.array(episode_traj_length)
        all_episodes[f"{episode_prefix}_traj_qpos"] = np.array(episode_traj_qpos)
        all_episodes[f"{episode_prefix}_joint_positions"] = np.array(episode_joint_positions)
        all_episodes[f"{episode_prefix}_joint_velocities"] = np.array(episode_joint_velocities)
        all_episodes[f"{episode_prefix}_joint_accelerations"] = np.array(episode_joint_accelerations)
        # all_episodes[f"{episode_prefix}_observations"] = np.array(episode_observations)
        all_episodes[f"{episode_prefix}_touch_observations"] = np.array(episode_touch_observations)
        all_episodes[f"{episode_prefix}_policy_actions"] = np.array(episode_policy_actions)
        all_episodes[f"{episode_prefix}_muscle_commands"] = np.array(episode_muscle_commands)
        all_episodes[f"{episode_prefix}_muscle_activations"] = np.array(episode_muscle_activations)
        all_episodes[f"{episode_prefix}_rewards"] = np.array(episode_rewards)
        all_episodes[f"{episode_prefix}_timesteps"] = np.array(episode_timesteps)

        episode_info = {
            "episode_id": episode_id,
            "episode_steps": episode_step_count,
            "episode_reward": episode_reward,
            "episode_duration": episode_duration,
            "episode_fps": episode_fps,
            "terminated": done,
        }
        episode_metadata.append(episode_info)

        episode_id += 1

        if total_step_count >= n_steps:
            break

    end_time = time.time()
    total_duration = end_time - start_time
    overall_fps = total_step_count / total_duration if total_duration > 0 else 0

    print("\n=== COLLECTION SUMMARY ===")
    print(f"Total episodes collected: {episode_id}")
    print(f"Total steps: {total_step_count}")
    print(f"Total duration: {total_duration:.2f}s ({overall_fps:.1f} FPS)")
    print(f"Total reward: {sum(ep['episode_reward'] for ep in episode_metadata):.3f}")

    all_episodes.update(
        {
            "n_episodes": episode_id,
            "total_steps": total_step_count,
            "total_duration": total_duration,
            "total_fps": overall_fps,
            "dt": env.dt,
            "joint_names": joint_names,
            "env_name": env.__class__.__name__,
            "backend": "MuJoCo" if use_mujoco else "MJX",
            "episode_metadata": episode_metadata,
        }
    )

    backend_name = "mujoco" if use_mujoco else "mjx"
    filename = f"{file_prefix}_{backend_name}_{motion_group}.npz"
    filepath = os.path.join(trajectory_dir, filename)

    np.savez_compressed(filepath, **all_episodes)

    print("\n=== TRAJECTORY EXPORT COMPLETE ===")
    print(f"File: {filepath}")
    print(f"Episodes: {episode_id} episodes stored")
    print("Data structure:")
    for i in range(episode_id):
        ep_steps = len(all_episodes[f"episode_{i}_joint_positions"])
        policy_range = (
            np.min(all_episodes[f"episode_{i}_policy_actions"]),
            np.max(all_episodes[f"episode_{i}_policy_actions"]),
        )
        muscle_range = (
            np.min(all_episodes[f"episode_{i}_muscle_commands"]),
            np.max(all_episodes[f"episode_{i}_muscle_commands"]),
        )
        print(f"  Episode {i}: {ep_steps} steps")
        print(f"    Policy actions range: [{policy_range[0]:.3f}, {policy_range[1]:.3f}]")
        print(f"    Muscle commands range: [{muscle_range[0]:.3f}, {muscle_range[1]:.3f}]")
    print(f"Joint names: {len(joint_names)} joints")
    print(f"File size: {os.path.getsize(filepath) / 1024:.1f} KB")

    return filepath, all_episodes
