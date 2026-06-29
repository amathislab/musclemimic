# Example usage:
#   uv run examples/retargeting/retarget_visualize.py --motion "KIT/6/WalkInCounterClockwiseCircle06_1_poses"  --record
#   uv run --extra c3d --extra smpl --extra gmr examples/retargeting/retarget_visualize.py --c3d-file path/to/file.c3d --c3d-model-path /path/to/smplx
import argparse

from loco_mujoco.task_factories import AMASSDatasetConf, CustomDatasetConf, ImitationFactory
from musclemimic.utils import detect_headless_environment, setup_headless_rendering


def get_video_name_from_motion(motion_name: str) -> str:
    """Convert motion path to video name (e.g., 'KIT/3/tennis_forehand_right04_poses' -> 'KIT_3_tennis_forehand_right04_poses')."""
    return motion_name.replace("/", "_")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retargeting and visualization for muscle-based environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        choices=["MyoBimanualArm", "MyoFullBody"],
        default="MyoFullBody",
        help="Model type to use",
    )

    # Motion configuration
    parser.add_argument(
        "--motion",
        action="append",
        default=None,
        help="Motion name (e.g., 'KIT/3/tennis_forehand_right04_poses'). Can be passed multiple times.",
    )
    parser.add_argument(
        "--dataset-group",
        type=str,
        default=None,
        help="Dataset group from loco_mujoco.smpl.const",
    )
    parser.add_argument(
        "--c3d-file",
        type=str,
        default=None,
        help="Path to a C3D file. Fits SMPL, retargets, and visualizes on the musculoskeletal model.",
    )

    # Retargeting method
    parser.add_argument(
        "--retargeting-method",
        choices=["smpl", "gmr"],
        default=None,
        help="Retargeting method. Defaults to smpl for AMASS motions and gmr for C3D files.",
    )

    # C3D fitting configuration
    c3d_group = parser.add_argument_group("C3D fitting")
    c3d_group.add_argument(
        "--c3d-surface-model",
        choices=["smplh", "smplx"],
        default="smplx",
        help="Body surface model used by C3D marker fitting",
    )
    c3d_group.add_argument(
        "--c3d-gender",
        choices=["neutral", "male", "female"],
        default="male",
        help="Body model gender used by C3D marker fitting",
    )
    c3d_group.add_argument("--c3d-model-path", default=None, help="SMPL-X/SMPL-H model path for C3D fitting")
    c3d_group.add_argument(
        "--c3d-dataset-name",
        default=None,
        help="Optional relative name for the saved converted C3D trajectory",
    )
    c3d_group.add_argument(
        "--clear-c3d-cache",
        action="store_true",
        default=False,
        help="Ignore existing converted C3D and SMPL-fit caches and recompute them",
    )
    c3d_group.add_argument(
        "--retarget-smpl-model-path",
        default=None,
        help="SMPL-H/SMPL model path for direct --retargeting-method smpl",
    )
    c3d_group.add_argument("--pose-body-prior-path", default=None, help="Optional MoSh++ pose_body_prior.pkl")
    c3d_group.add_argument("--head-marker-corr-path", default=None, help="Optional MoSh++ ssm_head_marker_corr.npz")
    c3d_group.add_argument(
        "--stage1-shape-solver",
        choices=["joint_dogleg", "joint_dogleg_jax"],
        default="joint_dogleg_jax",
        help="Stage-I C3D fitting solver",
    )
    c3d_group.add_argument("--stage1-iters", type=int, default=4, help="Stage-I iteration budget")
    c3d_group.add_argument("--stage2-iters", type=int, default=80, help="Stage-II per-frame iteration budget")
    c3d_group.add_argument(
        "--optimize-toes",
        action="store_true",
        default=False,
        help="Allow C3D fitting to optimize SMPL L_Foot/R_Foot rotations",
    )

    # GMR-specific configuration
    gmr_group = parser.add_argument_group("GMR Configuration (only applies when --retargeting-method=gmr)")
    gmr_group.add_argument("--gmr-src-human", default="smplh", help="Source human model for GMR")
    gmr_group.add_argument("--gmr-target-fps", type=int, default=30, help="Target FPS for GMR retargeting")
    gmr_group.add_argument("--gmr-solver", default="daqp", help="IK solver for GMR (e.g., daqp, qpswift)")
    gmr_group.add_argument("--gmr-damping", type=float, default=0.5, help="Damping factor for GMR solver")
    gmr_group.add_argument(
        "--gmr-offset-to-ground", action="store_true", default=False, help="Offset trajectory to ground"
    )
    gmr_group.add_argument("--gmr-no-offset-to-ground", dest="gmr_offset_to_ground", action="store_false")
    gmr_group.add_argument("--gmr-use-velocity-limit", action="store_true", default=False, help="Use velocity limits")
    gmr_group.add_argument("--gmr-verbose", action="store_true", default=False, help="Enable verbose GMR output")

    # Visualization configuration
    parser.add_argument("--record", action="store_true", default=False, help="Record video of trajectory playback")
    parser.add_argument(
        "--output-dir", type=str, default="./retargeting_recordings", help="Output directory for recordings"
    )
    parser.add_argument("--video-name", type=str, default="retargeted", help="Name for output video file")
    parser.add_argument("--n-episodes", type=int, default=2, help="Number of episodes to play/record")
    parser.add_argument("--n-steps", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--no-render", action="store_true", default=False, help="Disable rendering (dry run)")

    # Terrain randomization
    terrain_group = parser.add_argument_group("Terrain Configuration")
    terrain_group.add_argument(
        "--terrain", action="store_true", default=False, help="Enable RoughTerrain randomization"
    )
    terrain_group.add_argument(
        "--terrain-height", type=float, default=0.03, help="Max terrain height variation (meters)"
    )
    terrain_group.add_argument(
        "--terrain-platform", type=float, default=1.5, help="Flat platform size in center (meters)"
    )

    return parser.parse_args()


def _c3d_to_trajectory(args):
    """Convert C3D file to a retargeted trajectory via SMPL fitting."""
    import logging

    from musclemimic.web_viewer.c3d_pipeline import retarget_c3d_to_trajectory

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("c3d_pipeline")

    try:
        trajectory, _analysis = retarget_c3d_to_trajectory(
            args.c3d_file,
            args.model,
            retargeting_method=args.retargeting_method,
            gmr_config={
                "src_human": args.gmr_src_human,
                "target_fps": args.gmr_target_fps,
                "solver": args.gmr_solver,
                "damping": args.gmr_damping,
                "offset_to_ground": args.gmr_offset_to_ground,
                "use_velocity_limit": args.gmr_use_velocity_limit,
                "verbose": args.gmr_verbose,
            },
            logger=log,
            optimize_toes=args.optimize_toes,
            pose_body_prior_path=args.pose_body_prior_path,
            head_marker_corr_path=args.head_marker_corr_path,
            surface_model_type=args.c3d_surface_model,
            gender=args.c3d_gender,
            c3d_fit_model_path=args.c3d_model_path,
            retarget_smpl_model_path=args.retarget_smpl_model_path,
            stage1_shape_solver=args.stage1_shape_solver,
            stage1_iters=args.stage1_iters,
            stage2_iters=args.stage2_iters,
            converted_c3d_name=args.c3d_dataset_name,
            clear_cache=args.clear_c3d_cache,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return trajectory


def main():
    args = parse_arguments()

    # Detect headless environment before imports
    is_headless = detect_headless_environment()
    if is_headless:
        setup_headless_rendering()

    # Build dataset configuration
    motions = args.motion if args.motion is not None else []
    modes = [bool(motions), bool(args.dataset_group), bool(args.c3d_file)]
    if sum(modes) != 1:
        raise SystemExit("Pass exactly one of --motion, --dataset-group, or --c3d-file.")

    retargeting_method = args.retargeting_method or ("gmr" if args.c3d_file else "smpl")
    args.retargeting_method = retargeting_method

    custom_traj = None
    if args.c3d_file:
        custom_traj = _c3d_to_trajectory(args)
        print(f"Model: {args.model}")
        print(f"Source: C3D file → {args.c3d_file}")
        print(f"Retargeting Method: {retargeting_method.upper()}")
    else:
        if motions:
            dataset_conf = AMASSDatasetConf(motions)
        else:
            dataset_conf = AMASSDatasetConf(dataset_group=args.dataset_group)

        # Configure retargeting method
        dataset_conf.retargeting_method = retargeting_method
        if retargeting_method == "gmr":
            dataset_conf.gmr_config = {
                "src_human": args.gmr_src_human,
                "target_fps": args.gmr_target_fps,
                "solver": args.gmr_solver,
                "damping": args.gmr_damping,
                "offset_to_ground": args.gmr_offset_to_ground,
                "use_velocity_limit": args.gmr_use_velocity_limit,
                "verbose": args.gmr_verbose,
            }

        print(f"Model: {args.model}")
        print(f"Retargeting Method: {retargeting_method.upper()}")
        if motions:
            print("Motions:")
            for m in motions:
                print(f"  - {m}")
        if retargeting_method == "gmr":
            print(f"  GMR Solver: {args.gmr_solver}, FPS: {args.gmr_target_fps}")

    # Configure model-specific parameters (camera uses env defaults like validation_video_recorder)
    env_params = {
        "env_params": {"timestep": 0.002, "n_substeps": 5},
        "th_params": {"random_start": False, "fixed_start_conf": (0, 0)},
        "headless": is_headless,
    }
    if custom_traj is not None:
        env_params["custom_dataset_conf"] = CustomDatasetConf(traj=custom_traj)
    else:
        env_params["amass_dataset_conf"] = dataset_conf

    # Add terrain randomization if enabled
    if args.terrain:
        print("\nTerrain Randomization: ON")
        print(f"  Height range: ±{args.terrain_height}m")
        print(f"  Platform size: {args.terrain_platform}m")
        env_params["terrain_type"] = "RoughTerrain"
        env_params["terrain_params"] = {
            "inner_platform_size_in_meters": args.terrain_platform,
            "random_min_height": -args.terrain_height,
            "random_max_height": args.terrain_height,
            "random_step": 0.005,
            "random_downsampled_scale": 0.4,
        }

    # Goal params for visualization
    goal_params = {
        "visualize_goal": False,
        "enable_enhanced_visualization": True,
        "target_geom_rgba": [0.471, 0.38, 0.812, 0.6],
    }

    if args.model == "MyoBimanualArm":
        env_params["goal_type"] = "GoalBimanualTrajMimicv2"
        env_params["goal_params"] = goal_params
    elif args.model == "MyoFullBody":
        env_params["goal_type"] = "GoalTrajMimicv2"
        env_params["goal_params"] = goal_params

    # Create environment
    env = ImitationFactory.make(args.model, **env_params)

    # Print trajectory info
    print("\nTrajectory Info:")
    print(f"  Control frequency: {1.0 / env.dt:.1f} Hz")
    print(f"  Trajectory frequency: {env.th.traj.info.frequency:.1f} Hz")
    print(f"  Trajectory length: {len(env.th.traj.data.qpos)} frames")

    if args.no_render:
        print("\nDry run complete (--no-render specified)")
        return

    # Determine video name from motion if not explicitly set
    video_name = args.video_name
    if motions and video_name == "retargeted":
        video_name = get_video_name_from_motion(motions[0]) if len(motions) == 1 else f"{len(motions)}_motions"

    # Record or play trajectory
    recorder_params = None
    if args.record:
        recorder_params = {
            "path": args.output_dir,
            "tag": f"{args.model.lower()}_retargeted",
            "video_name": video_name,
            "compress": True,
        }
        fps = int(1.0 / env.dt)
        print(f"\nRecording to: {args.output_dir}/{args.model.lower()}_retargeted/{video_name}.mp4 ({fps} FPS)")

    env.play_trajectory(
        n_episodes=args.n_episodes,
        n_steps_per_episode=args.n_steps,
        render=True,
        record=args.record,
        recorder_params=recorder_params,
    )

    if args.record:
        print("Recording completed!")
    else:
        print("Playback completed!")


if __name__ == "__main__":
    main()
