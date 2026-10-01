#!/usr/bin/env python3
"""
MyoBimanualArm Evaluation Script

Usage examples:
- Basic evaluation:
    python bimanual/eval.py --path outputs/.../checkpoint_123
- MuJoCo evaluation:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --use_mujoco
- Default MuJoCo viewer:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --use_mujoco --mujoco_viewer
- Recording:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --record
- Recording without early termination:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --record --no_termination
- Stochastic policy:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --stochastic
- Trajectory export:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --export_trajectory
- Headless mode:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --no_render
- List trajectories:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --list_trajs
- Evaluate a specific motion:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --traj_index 3 --traj_start_step 0
- Compute validation metrics:
    python bimanual/eval.py --path outputs/.../checkpoint_123 --metrics
"""

import argparse
import os
import sys

from omegaconf import OmegaConf

from loco_mujoco.task_factories import TaskFactory
from musclemimic.runner.engine import pick_algorithm
from musclemimic.runner.eval_utils import (
    add_common_eval_args,
    align_agent_state,
    apply_temporal_params,
    apply_trajectory_selection,
    check_trajectory_sync,
    configure_goal_visualization,
    configure_recording,
    format_trajectory_listing,
    load_checkpoint,
    normalize_eval_args,
    run_validation_metrics,
    run_validation_metrics_mujoco,
    run_with_mujoco_viewer,
    run_with_trajectory_export,
    setup_headless,
    validate_traj_index,
    validate_viewer_args,
    verify_env_dt,
)

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
from jax import config as jax_config

jax_config.update("jax_default_matmul_precision", "high")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fullbody or bimanual policy evaluation.")
    add_common_eval_args(parser, default_n_envs=5)

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
        "--no_termination",
        default=False,
        action="store_true",
        help="Disable early termination (run for full n_steps regardless of falls).",
    )
    parser.add_argument(
        "--terminal_state_type",
        type=str,
        default=None,
        help="Override terminal state handler type (e.g., EnhancedBimanualTerminalStateHandler)",
    )
    parser.add_argument(
        "--max_site_deviation",
        type=float,
        default=None,
        help="Override max site deviation threshold for terminal state handler",
    )
    parser.add_argument(
        "--enable_reference_check",
        type=int,
        default=None,
        help="Override reference check (0/1) for terminal state handler",
    )
    parser.add_argument(
        "--site_deviation_mode",
        type=str,
        default=None,
        choices=["mean", "max"],
        help="Override site deviation mode for terminal handler (mean/max).",
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
    algorithm_cls = pick_algorithm(config)

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
        print(f"   New control_dt: {new_control_dt} ({1 / new_control_dt:.0f}Hz)")

        # Scale n_step_stride to maintain same lookahead time window
        goal_params = config.experiment.env_params.get("goal_params", {})
        old_stride = goal_params.get("n_step_stride")
        if old_stride is not None:
            new_stride = round(old_stride * training_control_dt / new_control_dt)
            new_stride = max(1, new_stride)
            config.experiment.env_params["goal_params"]["n_step_stride"] = new_stride
            print(
                f"   n_step_stride: {old_stride} -> {new_stride} "
                f"(preserving {old_stride * training_control_dt:.2f}s lookahead)"
            )

    # Bimanual-specific visualization configuration
    if env_name and "MyoBimanualArm" in env_name:
        print("\nConfiguring MyoBimanualArm evaluation:")
        configure_goal_visualization(config, args, "GoalBimanualTrajMimicv2", is_mjx_env="Mjx" in env_name)
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

    if args.motion_path is not None:
        motion_paths = args.motion_path if isinstance(args.motion_path, list) else [args.motion_path]
        config.experiment.task_factory.params.amass_dataset_conf.rel_dataset_path = motion_paths
        config.experiment.task_factory.params.amass_dataset_conf.dataset_group = None
        print(f"Motion path override: {motion_paths}")
    elif args.motion_group is not None:
        config.experiment.task_factory.params.amass_dataset_conf.dataset_group = args.motion_group
        print(f"Motion group override: {args.motion_group}")

    if args.start_from_beginning:
        if "th_params" not in config.experiment.env_params:
            config.experiment.env_params.th_params = {}
        config.experiment.env_params.th_params.start_from_random_step = False
        print("Start-from-beginning enabled: th_params.start_from_random_step=False")

    if args.no_termination:
        config.experiment.task_factory.params.terminal_state_type = "NoTerminalStateHandler"
        print("Early termination disabled")

    if args.terminal_state_type is not None:
        config.experiment.task_factory.params.terminal_state_type = args.terminal_state_type
        print(f"Terminal state handler override: {args.terminal_state_type}")

    if args.max_site_deviation is not None:
        if "terminal_state_params" not in config.experiment.task_factory.params:
            config.experiment.task_factory.params.terminal_state_params = {}
        config.experiment.task_factory.params.terminal_state_params.max_site_deviation = float(args.max_site_deviation)
        print(f"Max site deviation override: {args.max_site_deviation}")

    if args.enable_reference_check is not None:
        if "terminal_state_params" not in config.experiment.task_factory.params:
            config.experiment.task_factory.params.terminal_state_params = {}
        config.experiment.task_factory.params.terminal_state_params.enable_reference_check = bool(
            int(args.enable_reference_check)
        )
        print(f"Enable reference check override: {bool(int(args.enable_reference_check))}")

    if args.site_deviation_mode is not None:
        if "terminal_state_params" not in config.experiment.task_factory.params:
            config.experiment.task_factory.params.terminal_state_params = {}
        config.experiment.task_factory.params.terminal_state_params.site_deviation_mode = str(args.site_deviation_mode)
        print(f"Site deviation mode override: {args.site_deviation_mode}")

    play_env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    if args.use_mujoco and "Mjx" in play_env_params.get("env_name", ""):
        play_env_params["env_name"] = play_env_params["env_name"].replace("Mjx", "")
    if not args.use_mujoco:
        play_env_params["num_envs"] = int(args.num_envs)

    actual_control_dt = training_timestep * (args.n_substeps if args.n_substeps else training_n_substeps)

    record_tag = None
    if args.record:
        # Extract motion name for recording folder name (if available)
        motion_paths = config.experiment.task_factory.params.amass_dataset_conf.get("rel_dataset_path", [])
        motion_name = None
        if motion_paths:
            first_path = motion_paths
            while hasattr(first_path, "__iter__") and not isinstance(first_path, str):
                first_path = next(iter(first_path), None)
            if first_path:
                motion_name = str(first_path).replace("/", "_")

        record_tag = configure_recording(play_env_params, args, actual_control_dt, motion_name)
        print(f"Recording to: {args.record_dir}/{record_tag}/ @ {int(1 / actual_control_dt)}fps")

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
    agent_conf = algorithm_cls.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)

    # Verify env timing and trajectory sync
    verify_env_dt(env, actual_control_dt)
    if env_name and "MyoBimanualArm" in env_name:
        check_trajectory_sync(env)

    # Optional metrics (training-style validation)
    if args.metrics:
        metrics_deterministic = True  # Default to deterministic
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
                eval_seed=args.eval_seed,
            )
        else:
            # Use JAX-based validation (MJX environment)
            _validation_summary, metrics_dict = run_validation_metrics(
                config,
                agent_state,
                agent_conf,
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
            trajectory_filepath, _trajectory_data = run_with_trajectory_export(
                env,
                agent_conf,
                agent_state,
                n_steps=args.n_steps,
                deterministic=not args.stochastic,
                use_mujoco=args.use_mujoco,
                train_state_seed=args.train_state_seed,
                trajectory_dir=args.trajectory_dir,
                file_prefix="bimanual_episodes",
                motion_group=args.motion_group,
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
            algorithm_cls.play_policy_mujoco(
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
            algorithm_cls.play_policy(
                env,
                agent_conf,
                agent_state,
                deterministic=not args.stochastic,
                n_steps=args.n_steps,
                n_envs=int(args.num_envs),
                render=enable_render,
                record=args.record,
                train_state_seed=args.train_state_seed,
                sequential_mjx=False,
            )

        print("\nEvaluation completed successfully!")
        if args.record and record_tag is not None:
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
