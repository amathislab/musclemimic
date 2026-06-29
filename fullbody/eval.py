"""
Fullbody policy evaluation and visualization entrypoint.

Common workflows:

1. Basic MJX playback
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123

2. MuJoCo playback or viewer
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --use_mujoco
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --use_mujoco --mujoco_viewer

3. Record a rollout
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --record

4. Inspect available trajectories
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --list_trajs

5. Replay a specific trajectory
   uv run python fullbody/eval.py --path outputs/.../checkpoint_123 --traj_index 3 --traj_start_step 0

6. Run validation metrics for the KIT testing group
   uv run python fullbody/eval.py \
     --path hf://amathislab/mm-base-ll \
     --metrics --metrics_only \
     --motion_group KIT_KINESIS_TESTING_MOTIONS \
     --terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler \
     --root_deviation_threshold 0.5 \
     --eval_seed 0

7. Evaluate all motions with MJX/GPU metrics
   uv run python fullbody/eval.py \
     --path hf://amathislab/mm-base-ll \
     --evaluate_all --metrics --metrics_only \
     --motion_group KIT_KINESIS_TESTING_MOTIONS \
     --terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler \
     --root_deviation_threshold 0.5 \
     --metrics_envs 8 \
     --eval_seed 0

8. Evaluate all motions with MuJoCo/CPU metrics
   uv run python fullbody/eval.py \
     --path hf://amathislab/mm-base-ll \
     --use_mujoco \
     --evaluate_all --metrics --metrics_only \
     --motion_group KIT_KINESIS_TESTING_MOTIONS \
     --terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler \
     --root_deviation_threshold 0.5 \
     --eval_seed 0

Useful notes:
- `--traj_index` is 0-based.
- `--motion_group` loads an entire dataset split; use `--motion_path` for a single motion.
- `--metrics_envs` only affects MJX metrics, not MuJoCo metrics.
- Evaluation defaults to `validation.terminal_state_type` / `validation.terminal_state_params`; use `--strict_termination` to keep training-time termination settings.
- Fullbody metrics default to stochastic evaluation; use `--metrics_deterministic` to override.
- `--eval_seed` controls the RNG seed for stochastic action sampling (default: 0). Use different values to test variance across random seeds.
- `--train_state_seed` selects which set of network weights to evaluate when a checkpoint was trained with multiple random seeds (n_seeds > 1). For standard single-seed training this flag has no effect and can be omitted.
- `--terminal_state_type MeanRelativeSiteDeviationWithRootTerminalStateHandler` terminates when relative site tracking drifts too far, with extra root-position and optional root-orientation guards.
- `--mean_site_deviation_threshold`, `--root_deviation_threshold`, `--root_orientation_threshold`, and `--root_site` apply to compatible fullbody terminal handlers.
- `--root_deviation_threshold 0.5` sets that root-position tolerance to 0.5 meters.
"""

import argparse
import os
import sys

from omegaconf import OmegaConf

from fullbody._eval_terminal import apply_eval_terminal_defaults, apply_terminal_cli_overrides
from musclemimic.algorithms import PPOJax
from musclemimic.runner.eval_utils import (
    add_common_eval_args,
    normalize_eval_args,
    validate_viewer_args,
    setup_headless,
    load_checkpoint,
    apply_temporal_params,
    apply_trajectory_selection,
    configure_goal_visualization,
    configure_recording,
    align_agent_state,
    validate_traj_index,
    format_trajectory_listing,
    verify_env_dt,
    check_trajectory_sync,
    run_validation_metrics,
    run_validation_metrics_mjx_all,
    run_validation_metrics_mujoco,
    run_with_mujoco_viewer,
    run_with_trajectory_export,
)
from loco_mujoco.task_factories import TaskFactory

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
from jax import config as jax_config

jax_config.update("jax_default_matmul_precision", "high")


def apply_eval_init_state_overrides(env_params: dict, config) -> None:
    validation_cfg = config.experiment.get("validation", {})
    val_init_state_type = validation_cfg.get("init_state_type", None)
    if val_init_state_type is not None:
        env_params["init_state_type"] = val_init_state_type
        env_params["init_state_params"] = dict(validation_cfg.get("init_state_params", {}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate, visualize, or record a fullbody PPO policy.")
    add_common_eval_args(parser, default_n_envs=10)
    parser.add_argument(
        "--hfield_length",
        type=int,
        default=None,
        help="Override terrain heightfield resolution (default: use training config). Lower values reduce contacts.",
    )
    parser.add_argument(
        "--motion_path",
        type=str,
        nargs="+",
        default=None,
        help="Override motion path for the dataset (default: use training config).",
    )
    parser.add_argument(
        "--motion_group",
        type=str,
        default=None,
        help="Override motion group for the dataset (default: use training config).",
    )
    parser.add_argument(
        "--start_from_beginning",
        default=False,
        action="store_true",
        help="Start evaluation from the beginning of the motion (default: False).",
    )
    parser.add_argument(
        "--evaluate_all",
        default=False,
        action="store_true",
        help="Evaluate all trajectories in the dataset (default: False).",
    )
    parser.add_argument(
        "--no_termination",
        default=False,
        action="store_true",
        help="Disable early termination (run for full n_steps regardless of falls).",
    )
    parser.add_argument(
        "--terminal_state_type",
        type=str,
        default=None,
        help="Override terminal state handler type (e.g., MeanRelativeSiteDeviationWithRootTerminalStateHandler)",
    )
    parser.add_argument(
        "--mean_site_deviation_threshold",
        type=float,
        default=None,
        help="Override mean site deviation threshold for compatible fullbody terminal handlers.",
    )
    parser.add_argument(
        "--root_deviation_threshold",
        type=float,
        default=None,
        help="Override root position deviation threshold for compatible fullbody terminal handlers.",
    )
    parser.add_argument(
        "--root_orientation_threshold",
        type=float,
        default=None,
        help="Override root orientation deviation threshold in radians for compatible fullbody terminal handlers.",
    )
    parser.add_argument(
        "--root_site",
        type=str,
        default=None,
        help="Override the root site name used by compatible fullbody terminal handlers.",
    )
    parser.add_argument(
        "--n_substeps",
        type=int,
        default=None,
        help="Override n_substeps (control frequency). Default 5 = 100Hz, 10 = 50Hz.",
    )

    args = parser.parse_args()

    try:
        normalize_eval_args(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    viewer_err = validate_viewer_args(args)
    if viewer_err:
        print(viewer_err)
        return 1

    is_headless, headless_err = setup_headless(args)
    if headless_err:
        print(headless_err)
        print("   Use --no_render for headless operation")
        return 1

    config, agent_state, _metadata = load_checkpoint(args.path)
    OmegaConf.set_struct(config, False)

    # Restore training configuration
    print("=== RESTORING TRAINING CONFIGURATION ===")
    training_env_params = config.experiment.env_params
    config.experiment.env_params = training_env_params.copy()

    env_name = config.experiment.env_params.get("env_name")
    goal_type = config.experiment.env_params.get("goal_type")
    goal_params = config.experiment.env_params.get("goal_params", {})

    print("Restored training environment configuration:")
    print(f"   Environment: {env_name}")
    print(f"   Goal type: {goal_type}")
    print(f"   Goal params: {goal_params}")

    # Evaluation-specific overrides
    config.experiment.env_params["headless"] = args.no_render
    apply_trajectory_selection(config, args.traj_index, args.traj_start_step)

    # Preserve training temporal parameters
    training_timestep = config.experiment.env_params.get("timestep", 0.002)
    training_n_substeps = config.experiment.env_params.get("n_substeps", 5)
    training_control_dt = apply_temporal_params(config)

    print("\n=== TRAINING TEMPORAL CONFIGURATION ===")
    print(f"   Training timestep: {training_timestep}")
    print(f"   Training n_substeps: {training_n_substeps}")
    print(f"   Training control_dt: {training_control_dt}")

    # Override n_substeps if specified (for testing different control frequencies)
    if args.n_substeps is not None:
        config.experiment.env_params["n_substeps"] = args.n_substeps
        new_control_dt = training_timestep * args.n_substeps
        print(f"\n=== OVERRIDE: n_substeps={args.n_substeps} ===")
        print(f"   New control_dt: {new_control_dt} ({1/new_control_dt:.0f}Hz)")

        # Scale n_step_stride to maintain same lookahead time window
        goal_params = config.experiment.env_params.get("goal_params", {})
        old_stride = goal_params.get("n_step_stride")
        if old_stride is not None:
            # lookahead_time = stride * control_dt (keep constant)
            new_stride = int(round(old_stride * training_control_dt / new_control_dt))
            new_stride = max(1, new_stride)
            config.experiment.env_params["goal_params"]["n_step_stride"] = new_stride
            print(f"   n_step_stride: {old_stride} -> {new_stride} (preserving {old_stride * training_control_dt:.2f}s lookahead)")

    # Fullbody-specific visualization and terminal overrides
    if "MyoFullBody" in env_name:
        print("\nConfiguring MyoFullBody evaluation:")
        configure_goal_visualization(config, args, "GoalTrajMimicv2", is_mjx_env="Mjx" in env_name)
        print(f"   Training goal_type: {goal_type}")
        print(f"   Evaluation goal_type: {config.experiment.env_params.get('goal_type')}")
        print(
            "   Training sites_for_mimic: "
            f"{config.experiment.env_params.get('goal_params', {}).get('sites_for_mimic', 'Not specified')}"
        )
        print(f"   Evaluation headless mode: {args.no_render}")
        print(
            "   Enhanced visualization geometries: "
            f"{config.experiment.env_params.get('goal_params', {}).get('n_visual_geoms', 'Default')}"
        )

    # Terrain override
    if args.hfield_length is not None:
        if "terrain_params" not in config.experiment.env_params:
            config.experiment.env_params["terrain_params"] = {}
        config.experiment.env_params["terrain_params"]["hfield_length"] = args.hfield_length
        print(f"Terrain override: hfield_length={args.hfield_length} (cell size: {8.0/args.hfield_length:.2f}m)")

    # Handle start_from_beginning option: start each episode from initial
    # timestep of a random motion
    if args.start_from_beginning:
        if "th_params" not in config.experiment.env_params:
            config.experiment.env_params.th_params = {}
        config.experiment.env_params.th_params.start_from_random_step = False

    if args.metrics and args.evaluate_all:
        print("\nEvaluating all trajectories in the dataset for validation metrics.")

        if "th_params" not in config.experiment.env_params:
            config.experiment.env_params.th_params = {}

        config.experiment.env_params.th_params.random_start = False
        config.experiment.env_params.th_params.fixed_start_conf = [0, 0]
        config.experiment.env_params.th_params.start_from_random_step = False


    # Override motion path if specified
    if args.motion_path is not None:
        motion_paths = args.motion_path if isinstance(args.motion_path, list) else [args.motion_path]
        config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path = motion_paths
        config.experiment.task_factory.params.amass_dataset_conf.dataset_group = None
        print(f"Motion path override: {motion_paths}")
    elif args.motion_group is not None:
        config.experiment.task_factory.params.amass_dataset_conf.dataset_group = args.motion_group
        print(f"Motion group override: {args.motion_group}")

    play_env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    apply_eval_init_state_overrides(play_env_params, config)
    # Terminal state configuration priority: CLI > validation config > training config.
    apply_eval_terminal_defaults(play_env_params, config, args.strict_termination)
    try:
        apply_terminal_cli_overrides(play_env_params, args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # Final runtime env overrides.
    if args.use_mujoco and "Mjx" in play_env_params.get("env_name", ""):
        play_env_params["env_name"] = play_env_params["env_name"].replace("Mjx", "")
    if not args.use_mujoco:
        play_env_params["num_envs"] = int(args.num_envs)

    # Compute actual control_dt (with override if specified)
    actual_control_dt = training_timestep * (args.n_substeps if args.n_substeps else training_n_substeps)

    if args.record:
        # Extract motion name from dataset config for recording folder name
        motion_paths = config.experiment.task_factory.params.amass_dataset_conf.get("rel_dataset_path", [])
        motion_name = None
        if motion_paths:
            # Handle nested list/ListConfig - flatten to get first string path
            first_path = motion_paths
            while hasattr(first_path, "__iter__") and not isinstance(first_path, str):
                first_path = list(first_path)[0] if first_path else None
            if first_path:
                # Use full path with / replaced by _ (e.g., "KIT/9/walking_run07_poses" -> "KIT_9_walking_run07_poses")
                motion_name = str(first_path).replace("/", "_")
        record_tag = configure_recording(play_env_params, args, actual_control_dt, motion_name)
        print(f"Recording to: {args.record_dir}/{record_tag}/ @ {int(1/actual_control_dt)}fps")

    # Create environment.
    # task_factory.params contributes factory-specific inputs such as dataset selection.
    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    merged_params = {**play_env_params, **task_params}
    try:
        env = factory.make(**merged_params)
    except Exception as exc:
        print(f"Error creating environment: {exc!s}")
        print("Exception traceback:")
        import traceback

        traceback.print_exc()
        return 1

    if args.list_trajs:
        for line in format_trajectory_listing(env, args.list_trajs_limit):
            print(line)
        return 0

    validate_traj_index(env, args.traj_index)

    # Build agent configuration and align state for inference
    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)

    # Verify env timing and trajectory sync
    verify_env_dt(env, actual_control_dt)
    if "MyoFullBody" in env_name:
        check_trajectory_sync(env)

    # Optional metrics (training-style validation)
    if args.metrics:
        metrics_deterministic = False  # Default to stochastic for fullbody eval
        if args.metrics_deterministic:
            metrics_deterministic = True
        elif args.metrics_stochastic:
            metrics_deterministic = False

        if args.use_mujoco:
            # Use MuJoCo CPU-compatible metrics collection (uses the already created env)
            print("Note: Using MuJoCo (CPU) environment for validation metrics")
            metrics_dict = run_validation_metrics_mujoco(
                env,
                agent_conf,
                agent_state,
                num_steps=args.metrics_steps or 500,
                deterministic=metrics_deterministic,
                train_state_seed=args.train_state_seed,
                evaluate_all=args.evaluate_all,
                eval_seed=args.eval_seed,
            )
        elif args.evaluate_all:
            # MJX GPU evaluate_all: per-trajectory evaluation without AutoResetWrapper
            metrics_dict = run_validation_metrics_mjx_all(
                env,
                agent_conf,
                agent_state,
                deterministic=metrics_deterministic,
                train_state_seed=args.train_state_seed,
                num_envs=args.metrics_envs,
                eval_seed=args.eval_seed,
            )
        else:
            # Use JAX-based validation (MJX environment)
            _validation_summary, metrics_dict = run_validation_metrics(
                config,
                agent_state,
                train_state_seed=args.train_state_seed,
                num_steps=args.metrics_steps,
                num_envs=args.metrics_envs,
                deterministic_override=metrics_deterministic,
                eval_seed=args.eval_seed,
            )

        print("\n=== VALIDATION METRICS ===")
        for key in sorted(metrics_dict.keys()):
            print(f"{key}: {metrics_dict[key]:.6f}")
        if args.metrics_only:
            return 0

    # Run evaluation
    print(f"\nStarting evaluation for {args.n_steps} steps...")
    if args.record:
        print("Recording enabled")
        if is_headless:
            print("  Recording will use headless EGL rendering (no display window)")
    if args.export_trajectory:
        print(f"Trajectory export enabled - saving to {args.trajectory_dir}")

    try:
        enable_render = not args.no_render or args.record

        if args.export_trajectory:
            print("Running evaluation with trajectory export...")
            if args.no_render:
                print("  Headless mode enabled for trajectory export")
            trajectory_filepath, trajectory_data = run_with_trajectory_export(
                env,
                agent_conf,
                agent_state,
                n_steps=args.n_steps,
                deterministic=not args.stochastic,
                use_mujoco=args.use_mujoco,
                train_state_seed=args.train_state_seed,
                trajectory_dir=args.trajectory_dir,
                file_prefix="myofullbody_episodes",
            )

        elif args.use_mujoco and args.mujoco_viewer:
            print("Running MuJoCo evaluation with default viewer...")
            run_with_mujoco_viewer(
                env,
                agent_conf,
                agent_state,
                n_steps=args.n_steps,
                deterministic=not args.stochastic,
                train_state_seed=args.train_state_seed,
            )
        elif args.viser_viewer:
            print("Launching Viser web viewer at http://localhost:8080")
            from musclemimic.viewer import ViserViewer

            viewer = ViserViewer(env, agent_conf, agent_state, deterministic=not args.stochastic)
            viewer.run(n_steps=args.n_steps)
        elif args.use_mujoco:
            print("Running MuJoCo evaluation...")
            PPOJax.play_policy_mujoco(
                env,
                agent_conf,
                agent_state,
                deterministic=not args.stochastic,
                n_steps=args.n_steps,
                render=enable_render,
                record=args.record,
                train_state_seed=args.train_state_seed,
            )
        else:
            print("Running MJX evaluation...")
            mjx_play_envs = 1 if args.traj_index is not None else int(args.num_envs)
            use_sequential_mjx = args.traj_index is not None and not args.evaluate_all
            PPOJax.play_policy(
                env,
                agent_conf,
                agent_state,
                deterministic=not args.stochastic,
                n_steps=args.n_steps,
                n_envs=mjx_play_envs,
                render=enable_render,
                record=args.record,
                train_state_seed=args.train_state_seed,
                sequential_mjx=use_sequential_mjx,
            )

        print("\nEvaluation completed successfully!")
        if args.record:
            print(f"Video recording saved to: {args.record_dir}/{record_tag}/")
            if hasattr(env, "video_file_path") and env.video_file_path:
                print(f"Video saved as: {env.video_file_path}")
        if args.export_trajectory:
            print(f"Trajectory data saved to: {trajectory_filepath}")

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    except Exception as exc:
        print(f"Error during evaluation: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
