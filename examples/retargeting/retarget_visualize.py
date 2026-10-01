# Example usage:
#   uv run examples/retargeting/retarget_visualize.py --motion "KIT/6/WalkInCounterClockwiseCircle06_1_poses"  --record
#   uv run --extra c3d --extra smpl --extra gmr examples/retargeting/retarget_visualize.py \
#       --c3d-file path/to/file.c3d --c3d-model-path /path/to/smplx
import argparse
import hashlib
from pathlib import Path

from loco_mujoco.task_factories import AMASSDatasetConf, CustomDatasetConf, ImitationFactory
from musclemimic.utils import detect_headless_environment, setup_headless_rendering


def get_video_name_from_motion(motion_name: str) -> str:
    """Convert a slash-separated motion path to a filesystem-safe video name."""
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
        choices=["smpl", "gmr", "terra", "omniretarget"],
        default=None,
        help="Retargeting method. Defaults to smpl for AMASS motions and gmr for C3D files.",
    )
    parser.add_argument(
        "--output-cache-subdir",
        default=None,
        help="Cache namespace independent of the algorithm (for example omniretarget).",
    )
    parser.add_argument(
        "--no-cache-download",
        action="store_true",
        help="Never fetch a missing GMR cache; require a local retargeting run.",
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
    gmr_group.add_argument(
        "--gmr-terrain",
        type=str,
        default=None,
        metavar="PATH",
        help="Terrain JSON whose local walkable height replaces GMR's hard-coded z=0 grounding target",
    )
    gmr_group.add_argument("--gmr-use-velocity-limit", action="store_true", default=False, help="Use velocity limits")
    gmr_group.add_argument("--gmr-verbose", action="store_true", default=False, help="Enable verbose GMR output")

    # TERRA owns this adapter and its extensions. The explicit ``omniretarget`` method
    # selects the contribution-free comparison profile.
    terra_group = parser.add_argument_group("TERRA adapter configuration (--retargeting-method=terra or omniretarget)")
    terra_group.add_argument(
        "--terra-no-fitted-shape",
        action="store_true",
        default=False,
        help="Pose SMPL with the subject's own betas and rescale by height (OmniRetarget behavior) "
        "instead of using MuscleMimic's fitted robot shape",
    )
    terra_group.add_argument("--terra-step-size", type=float, default=0.2, help="SQP trust region per iteration")
    terra_group.add_argument(
        "--terra-foot-sticking-tolerance", type=float, default=1e-3, help="Foot sticking tolerance in x, y (m)"
    )
    terra_group.add_argument(
        "--terra-penetration-tolerance",
        type=float,
        default=None,
        help="Non-penetration constraint tolerance (m). Defaults to 0.9 mm on terrain and "
        "the historical 1.0 mm on flat ground.",
    )
    terra_group.add_argument(
        "--terra-orient-weight",
        type=float,
        default=None,
        help="Weight of the mimic-site orientation cost (0 disables it, recovering the "
        "position-only interaction mesh). Defaults to DEFAULT_ORIENT_WEIGHT.",
    )
    terra_group.add_argument(
        "--terra-foot-orient-weight",
        type=float,
        default=None,
        help="Orientation-cost weight for the four foot mimic sites. Defaults to 0.1 on "
        "flat ground and --terra-orient-weight on terrain; set it independently to override "
        "the scene-class default.",
    )
    terra_group.add_argument(
        "--terra-axial-smooth-weight",
        type=float,
        default=None,
        help="Frame-to-frame damping on the axial DOFs. Not a substitute for the orientation "
        "cost, which removes their null space outright; this holds the shoulder's "
        "elv_angle/shoulder_rot parameterisation singularity, which no task-space cost can.",
    )
    terra_group.add_argument(
        "--terra-trunk-smooth-weight", type=float, default=None, help="Frame-to-frame damping on the trunk DOFs"
    )
    terra_group.add_argument(
        "--terra-foot-mode",
        choices=["anchored", "omniretarget", "off"],
        default=None,
        help="How planted feet are held. 'anchored' (default) pins each foot to its "
        "touchdown position with a ramped cost; 'omniretarget' uses OmniRetarget's "
        "frame-relative rate limit; 'off' applies no foot constraint (diagnostic - "
        "non-penetration is separate, so this does not reintroduce ground penetration).",
    )
    terra_group.add_argument(
        "--terra-foot-planted-speed",
        type=float,
        default=None,
        help="Speed below which a foot counts as planted (m/s). OmniRetarget's default is "
        "frame-rate dependent and becomes 1.0 m/s at 100 Hz.",
    )
    terra_group.add_argument(
        "--terra-warmup-frames",
        type=int,
        default=None,
        help="Static copies of frame 0 prepended before solving and discarded afterwards, "
        "so the SQP converges before the motion starts. 0 reproduces the untreated "
        "opening transient.",
    )
    terra_group.add_argument(
        "--terra-coupler-weight",
        type=float,
        default=None,
        help="Weight on the model's joint-coupler equalities (0 disables). OmniRetarget "
        "optimises qpos with no knowledge of them; smpl/gmr satisfy them to ~0.05 deg.",
    )
    terra_group.add_argument(
        "--terra-anchor-gate",
        choices=["strict", "omniretarget"],
        default=None,
        help="Which contact test gates the anchored foot constraint. 'strict' = toe slow "
        "AND low; 'omniretarget' = the shipped speed-only flag. Separates the gate from the "
        "anchoring so the two can be ablated independently.",
    )
    terra_group.add_argument(
        "--terra-foot-anchor-weight", type=float, default=None, help="Weight of the foot anchor cost"
    )
    terra_group.add_argument(
        "--terra-foot-velocity-weight",
        type=float,
        default=None,
        help="Weight on horizontal foot displacement above the configured speed limit",
    )
    terra_group.add_argument(
        "--terra-foot-velocity-tracking-weight",
        type=float,
        default=None,
        help="Weight on all horizontal foot displacement during source sticking intervals",
    )
    terra_group.add_argument(
        "--terra-foot-velocity-limit",
        type=float,
        default=None,
        help="Unpenalized horizontal foot speed in m/s during source sticking intervals",
    )
    terra_group.add_argument(
        "--terra-terrain",
        type=str,
        default=None,
        metavar="AUTO_OR_PATH",
        help="Retarget onto terrain instead of flat ground. 'auto' reconstructs it from the "
        "motion's own foot contacts; otherwise a path to a terrain JSON (as written by "
        "the TERRA repository's scripts/terra/fit_terrain.py). The boxes are built into the environment, so the "
        "solver and the trained policy see the same geometry.",
    )
    terra_group.add_argument(
        "--terra-nonpen-mode",
        type=str,
        default=None,
        choices=["scene", "off"],
        help="Non-penetration against the floor and terrain. Defaults to 'scene' for both "
        "flat and non-flat scenes; 'off' is the explicit ablation.",
    )
    terra_group.add_argument(
        "--terra-clearance-mode",
        type=str,
        default=None,
        choices=["source", "off"],
        help="Require the soles to clear the terrain during swing by a fraction of the "
        "clearance the source motion itself achieved. Defaults to 'source' with terrain "
        "and 'off' without. Needs --terra-nonpen-mode scene.",
    )
    terra_group.add_argument(
        "--terra-clearance-fraction",
        type=float,
        default=None,
        help="Fraction of the source's own swing clearance to require (default 0.7)",
    )
    terra_group.add_argument(
        "--terra-clearance-cap",
        type=float,
        default=None,
        help="Most clearance the swing requirement may ask for, m (default 0.15). Below "
        "~0.07 it binds on every stair swing, since the source lifts 250-340 mm there.",
    )
    terra_group.add_argument(
        "--terra-clearance-weight",
        type=float,
        default=None,
        help="Weight of the swing-clearance shortfall cost (default 1500)",
    )
    terra_group.add_argument(
        "--terra-clearance-max-recovery-per-iter",
        type=float,
        default=None,
        help="Most clearance shortfall one SQP iteration may recover, m (default 0.01)",
    )
    terra_group.add_argument(
        "--terra-clearance-surface-ramp-seconds",
        type=float,
        default=None,
        help="Half-width of terrain surface-height smoothing, seconds (default 0.05)",
    )
    terra_group.add_argument(
        "--terra-min-swing-clearance",
        type=float,
        default=None,
        help="Clearance asked of a swinging sole over the surface within reach even where "
        "the source's own sole demonstrated none, m (default 0, disabled). This "
        "experimental sole-geom formulation is retained for ablations.",
    )
    terra_group.add_argument(
        "--terra-swing-target-lift",
        type=float,
        default=None,
        help="Translate the source ankle and toe targets together while a swing passes "
        "beside a tread nose, m (default 0.06). Unlike the sole-clearance "
        "minimum this preserves foot pitch while routing the foot around the ledge.",
    )
    terra_group.add_argument(
        "--terra-swing-target-ramp-frames",
        type=int,
        default=None,
        help="Frames used to ease route target and QP authority in/out (default 12).",
    )
    terra_group.add_argument(
        "--terra-swing-target-weight",
        type=float,
        default=None,
        help="Weight on the direct ankle/toe vertical route targets (default 1500).",
    )
    terra_group.add_argument(
        "--terra-swing-target-max-recovery-per-iter",
        type=float,
        default=None,
        help="Absolute route-target residual one SQP iteration is asked to recover, m (default 0.004).",
    )
    terra_group.add_argument(
        "--terra-mtp-smooth-weight",
        type=float,
        default=None,
        help="Frame-to-frame damping on the toe joints. They move no tracked landmark at all, "
        "so nothing else in the objective resolves them and the swing-clearance cost "
        "makes them flap. Defaults to 10 with terrain, off without.",
    )
    terra_group.add_argument(
        "--terra-selfpen-weight",
        type=float,
        default=None,
        help="Weight of the left/right overlap cost (default 1000 on flat ground, 20000 on terrain)",
    )
    terra_group.add_argument(
        "--terra-selfpen-max-recovery-per-iter",
        type=float,
        default=None,
        help="Inter-leg overlap one SQP iteration is asked to undo, m. The terrain "
        "default is DEFAULT_SELF_COLLISION_MAX_RECOVERY_PER_ITER.",
    )
    terra_group.add_argument(
        "--terra-nonpen-max-recovery",
        type=float,
        default=None,
        help="Penetration one SQP iteration is asked to undo, m (default 0.01). Larger gives "
        "the hard non-penetration row more authority against the added costs; too large "
        "and a cold start goes infeasible.",
    )
    terra_group.add_argument(
        "--terra-nonpen-engage-frame",
        type=int,
        default=None,
        help="Frame from which non-penetration is enforced, counted from the start of the "
        "warm-up (default: warmup_frames - 5, so the constraint first sees a settled "
        "pose). Lower it for a clip that *starts* standing on raised terrain: the "
        "opening pose is below the surface and five warm-up frames may not lift it out "
        "before the first stance latches.",
    )
    terra_group.add_argument(
        "--terra-selfpen-mode",
        type=str,
        default=None,
        choices=["legs", "off"],
        help="Keep the left and right leg segments out of each other. Defaults to 'legs' "
        "for both flat ground and terrain.",
    )
    terra_group.add_argument(
        "--terra-sole-offset",
        type=str,
        default=None,
        choices=["on", "off"],
        help="Raise the foot landmark targets so matching them puts the robot's soles where "
        "the human's soles were, rather than its joint centres where the human's joint "
        "centres were. Defaults to 'on' with terrain and 'off' without, which preserves "
        "the measured flat-ground behaviour.",
    )
    terra_group.add_argument(
        "--terra-foot-orient",
        type=str,
        default=None,
        choices=["on", "off"],
        help="Re-datum the foot orientation targets against the source's own flat-footed "
        "frames, removing the constant dorsiflexion the SMPL-to-robot alignment carries. "
        "Defaults to 'on' for both flat and terrain motions; flat ground uses a lower "
        "foot-site orientation weight.",
    )
    terra_group.add_argument(
        "--terra-extra-landmark",
        nargs=2,
        action="append",
        metavar=("SMPL_JOINT", "BODY"),
        default=None,
        help="Add a position landmark, e.g. --terra-extra-landmark Spine lumbar1. Repeatable.",
    )

    # Visualization configuration
    parser.add_argument("--record", action="store_true", default=False, help="Record video of trajectory playback")
    parser.add_argument(
        "--output-dir", type=str, default="./retargeting_recordings", help="Output directory for recordings"
    )
    parser.add_argument("--video-name", type=str, default="retargeted", help="Name for output video file")
    parser.add_argument(
        "--video-tag",
        type=str,
        default=None,
        help="Subdirectory under --output-dir for the video. May be nested, e.g. "
        "'myofullbody_retargeted/anchored', to keep each config version's clips together. "
        "Defaults to '<model>_retargeted'.",
    )
    parser.add_argument("--n-episodes", type=int, default=2, help="Number of episodes to play/record")
    parser.add_argument("--n-steps", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--no-render", action="store_true", default=False, help="Disable rendering (dry run)")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Re-retarget even if a cache exists. Use --no-cache-download for a local-only "
        "GMR run that cannot fall back to the published HuggingFace cache.",
    )

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
                "use_fitted_shape": True,
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
    terrain_spec = None

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
        dataset_conf.output_cache_subdir = args.output_cache_subdir
        dataset_conf.clear_cache = args.clear_cache
        dataset_conf.allow_cache_download = not args.no_cache_download
        if retargeting_method == "gmr":
            dataset_conf.gmr_config = {
                "src_human": args.gmr_src_human,
                "target_fps": args.gmr_target_fps,
                "solver": args.gmr_solver,
                "damping": args.gmr_damping,
                "offset_to_ground": args.gmr_offset_to_ground,
                "use_fitted_shape": True,
                "use_velocity_limit": args.gmr_use_velocity_limit,
                "verbose": args.gmr_verbose,
            }
            if args.gmr_terrain is not None:
                from loco_mujoco.core.terrain import TerrainSpec

                if len(motions) != 1:
                    raise SystemExit(f"--gmr-terrain needs exactly one --motion, and {len(motions)} were given.")
                terrain_path = Path(args.gmr_terrain).resolve()
                terrain_bytes = terrain_path.read_bytes()
                terrain_spec = TerrainSpec.load(terrain_path)
                dataset_conf.gmr_config.update(
                    terrain=terrain_spec.to_dict(),
                    terrain_source_path=str(terrain_path),
                    terrain_sha256=hashlib.sha256(terrain_bytes).hexdigest(),
                )
                print(f"\nGMR terrain: {len(terrain_spec)} box(es) from {terrain_path}")
        elif retargeting_method in {"terra", "omniretarget"}:
            dataset_conf.terra_config = {
                "use_fitted_shape": not args.terra_no_fitted_shape,
                "step_size": args.terra_step_size,
                "foot_sticking_tolerance": args.terra_foot_sticking_tolerance,
            }
            # Only forward what was actually passed, so `fit_terra_motion`'s defaults stay
            # the single source of truth for anything left alone.
            for key, value in (
                ("orient_weight", args.terra_orient_weight),
                ("foot_orient_weight", args.terra_foot_orient_weight),
                ("axial_smooth_weight", args.terra_axial_smooth_weight),
                ("trunk_smooth_weight", args.terra_trunk_smooth_weight),
                ("extra_landmarks", dict(args.terra_extra_landmark) if args.terra_extra_landmark else None),
                ("foot_planted_speed", args.terra_foot_planted_speed),
                ("foot_mode", args.terra_foot_mode),
                ("anchor_gate", args.terra_anchor_gate),
                ("coupler_weight", args.terra_coupler_weight),
                ("warmup_frames", args.terra_warmup_frames),
                ("foot_anchor_weight", args.terra_foot_anchor_weight),
                ("foot_velocity_weight", args.terra_foot_velocity_weight),
                ("foot_velocity_tracking_weight", args.terra_foot_velocity_tracking_weight),
                ("foot_velocity_limit", args.terra_foot_velocity_limit),
                ("nonpen_mode", args.terra_nonpen_mode),
                ("penetration_tolerance", args.terra_penetration_tolerance),
                ("sole_offset_mode", args.terra_sole_offset),
                ("foot_orient_mode", args.terra_foot_orient),
                ("selfpen_mode", args.terra_selfpen_mode),
                ("clearance_mode", args.terra_clearance_mode),
                ("clearance_fraction", args.terra_clearance_fraction),
                ("clearance_cap", args.terra_clearance_cap),
                ("clearance_weight", args.terra_clearance_weight),
                ("clearance_max_recovery_per_iter", args.terra_clearance_max_recovery_per_iter),
                ("clearance_surface_ramp_seconds", args.terra_clearance_surface_ramp_seconds),
                ("min_swing_clearance", args.terra_min_swing_clearance),
                ("swing_target_lift", args.terra_swing_target_lift),
                ("swing_target_ramp_frames", args.terra_swing_target_ramp_frames),
                ("swing_target_weight", args.terra_swing_target_weight),
                ("swing_target_max_recovery_per_iter", args.terra_swing_target_max_recovery_per_iter),
                ("mtp_smooth_weight", args.terra_mtp_smooth_weight),
                ("selfpen_weight", args.terra_selfpen_weight),
                ("selfpen_max_recovery_per_iter", args.terra_selfpen_max_recovery_per_iter),
                ("nonpen_max_recovery", args.terra_nonpen_max_recovery),
                ("nonpen_engage_frame", args.terra_nonpen_engage_frame),
            ):
                if value is not None:
                    dataset_conf.terra_config[key] = value

            dataset_conf.terra_config["method_profile"] = retargeting_method

            # Resolve the terrain once, here, and hand the *same* spec to the retargeter
            # and to the environment below. Fitting it separately in each place would
            # leave two descriptions free to drift apart on any parameter change - and a
            # policy trained against terrain a few centimetres from the one the motion
            # was retargeted onto fails in a way that looks like a bad reward.
            if args.terra_terrain is not None:
                from loco_mujoco.core.terrain import TerrainSpec
                from terra import terrain_for_motion

                if len(motions) != 1:
                    raise SystemExit(
                        "--terra-terrain needs exactly one --motion: the terrain is reconstructed "
                        f"from a specific motion's foot contacts, and {len(motions)} were given."
                    )
                if args.terra_terrain == "auto":
                    print(f"\nReconstructing terrain from {motions[0]} ...")
                    terrain_spec, _fit, terrain_check = terrain_for_motion(
                        motions[0],
                        env_name=args.model,
                        use_fitted_shape=dataset_conf.terra_config["use_fitted_shape"],
                    )
                    print(
                        f"  {len(terrain_spec)} box(es); validation "
                        f"{'PASS' if terrain_check['passed'] else 'FAIL'} "
                        f"(contact error max {terrain_check['raised_contact_error_max'] * 1000:.1f} mm, "
                        f"penetration {terrain_check['max_penetration'] * 1000:.1f} mm)"
                    )
                else:
                    terrain_spec = TerrainSpec.load(args.terra_terrain)
                    print(f"\nTerrain: {len(terrain_spec)} box(es) from {args.terra_terrain}")
                    terrain_path = Path(args.terra_terrain).resolve()
                    terrain_bytes = terrain_path.read_bytes()
                    dataset_conf.terra_config["terrain_source_path"] = str(terrain_path)
                    dataset_conf.terra_config["terrain_sha256"] = hashlib.sha256(terrain_bytes).hexdigest()
                dataset_conf.terra_config["terrain"] = terrain_spec.to_dict()

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

    # Reconstructed terrain: the same spec the motion was retargeted onto, so playback and
    # anything trained here stand on the geometry the solver actually constrained against.
    if terrain_spec is not None and not terrain_spec.is_flat:
        env_params["terrain_type"] = "BoxTerrain"
        env_params["terrain_params"] = terrain_spec.to_env_params()

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
            "tag": args.video_tag or f"{args.model.lower()}_retargeted",
            "video_name": video_name,
            "compress": True,
        }
        fps = int(1.0 / env.dt)
        print(f"\nRecording to: {args.output_dir}/{recorder_params['tag']}/{video_name}.mp4 ({fps} FPS)")

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
