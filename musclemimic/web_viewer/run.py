from __future__ import annotations

import argparse

from loco_mujoco.task_factories import AMASSDatasetConf, CustomDatasetConf, ImitationFactory

from .trajectory_viewer import TrajectoryViserViewer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Web-based trajectory viewer for retargeted MuscleMimic motions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["MyoBimanualArm", "MyoFullBody"],
        default="MyoFullBody",
        help="Model type to visualize.",
    )
    parser.add_argument(
        "--motion",
        action="append",
        default=None,
        help="Motion name (repeatable).",
    )
    parser.add_argument(
        "--dataset-group",
        type=str,
        default=None,
        help="Dataset group from loco_mujoco.smpl.const.",
    )
    parser.add_argument(
        "--c3d-file",
        type=str,
        default=None,
        help="Path to a C3D file. Fits SMPL, retargets, and visualizes on the musculoskeletal model.",
    )
    parser.add_argument(
        "--c3d-dataset-name",
        type=str,
        default=None,
        help=(
            "Optional relative name for the saved converted C3D trajectory. "
            "Defaults to a sanitized source path plus a short pipeline hash."
        ),
    )
    parser.add_argument(
        "--clear-c3d-cache",
        action="store_true",
        default=False,
        help="Ignore existing converted C3D and SMPL-fit caches and recompute them.",
    )
    parser.add_argument(
        "--retargeting-method",
        choices=["smpl", "gmr"],
        default="gmr",
        help="Retargeting method.",
    )
    parser.add_argument(
        "--optimize-toes",
        action="store_true",
        default=False,
        help=(
            "Allow the C3D->SMPL fit to optimize SMPL L_Foot/R_Foot rotations. "
            "Default matches moshpp's optimize_toes=False; turn ON for gait data "
            "with explicit LTOE/RTOE markers to avoid ankle over-rotation."
        ),
    )
    parser.add_argument(
        "--pose-body-prior-path",
        type=str,
        default=None,
        help="Optional MoSh++ pose_body_prior.pkl. Defaults to MUSCLEMIMIC_MOSHPP_ASSETS_PATH when configured.",
    )
    parser.add_argument(
        "--head-marker-corr-path",
        type=str,
        default=None,
        help="Optional MoSh++ ssm_head_marker_corr.npz. Defaults to MUSCLEMIMIC_MOSHPP_ASSETS_PATH when configured.",
    )
    parser.add_argument(
        "--c3d-surface-model",
        choices=["smplh", "smplx"],
        default="smplx",
        help="Body surface model used by the C3D marker fit.",
    )
    parser.add_argument(
        "--c3d-gender",
        type=str,
        choices=["neutral", "male", "female"],
        default="male",
        help="Body model gender used by the C3D marker fit.",
    )
    parser.add_argument(
        "--c3d-model-path",
        type=str,
        default=None,
        help="SMPL-X/SMPL-H model path used only for C3D marker fitting. Defaults to MUSCLEMIMIC_C3D_MODEL_PATH.",
    )
    parser.add_argument(
        "--retarget-smpl-model-path",
        type=str,
        default=None,
        help="SMPL-H/SMPL model path used only by the SMPL retargeting backend.",
    )
    parser.add_argument(
        "--stage1-shape-solver",
        choices=["joint_dogleg", "joint_dogleg_jax"],
        default="joint_dogleg_jax",
        help=(
            "Stage-I solver variant. joint_dogleg mirrors MoSh++'s residual trust-region solve; "
            "joint_dogleg_jax is the default JAX implementation and uses GPU when available."
        ),
    )
    parser.add_argument(
        "--stage1-iters",
        type=int,
        default=4,
        help="Stage-I iteration budget. Values below 25 map to at least 25 dogleg steps per annealing phase.",
    )
    parser.add_argument(
        "--stage2-iters",
        type=int,
        default=80,
        help="Stage-II per-frame pose fitting iterations.",
    )
    parser.add_argument("--gmr-src-human", default="smplh", help="Source human model for GMR.")
    parser.add_argument("--gmr-target-fps", type=int, default=30, help="Target FPS for GMR retargeting.")
    parser.add_argument("--gmr-solver", default="daqp", help="IK solver for GMR.")
    parser.add_argument("--gmr-damping", type=float, default=0.5, help="Damping factor for GMR.")
    parser.add_argument(
        "--gmr-offset-to-ground",
        action="store_true",
        default=False,
        help="Offset the trajectory to the ground plane.",
    )
    parser.add_argument(
        "--gmr-use-velocity-limit",
        action="store_true",
        default=False,
        help="Use GMR velocity limits.",
    )
    parser.add_argument(
        "--include-collision",
        action="store_true",
        default=False,
        help="Render collision geoms instead of visual geoms.",
    )
    return parser.parse_args()


def _make_env_from_c3d(args) -> tuple:
    """Fit SMPL to C3D markers, retarget, and create environment."""
    import logging
    from pathlib import Path

    from .c3d_pipeline import retarget_c3d_to_trajectory

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

    # Step 3: Create environment with the trajectory
    log.info("[3/3] Loading into viewer...")
    env = ImitationFactory.make(
        args.model,
        custom_dataset_conf=CustomDatasetConf(traj=trajectory),
        env_params={"timestep": 0.002, "n_substeps": 5},
    )

    label = Path(args.c3d_file).stem
    return env, [label]


def main() -> None:
    args = parse_args()

    modes = [bool(args.motion), bool(args.dataset_group), bool(args.c3d_file)]
    if sum(modes) != 1:
        raise SystemExit("Pass exactly one of --motion, --dataset-group, or --c3d-file.")

    if args.c3d_file:
        env, motion_labels = _make_env_from_c3d(args)
    else:
        motions = args.motion or []
        if motions:
            dataset_conf = AMASSDatasetConf(motions)
        else:
            dataset_conf = AMASSDatasetConf(dataset_group=args.dataset_group)

        dataset_conf.retargeting_method = args.retargeting_method
        if args.retargeting_method == "gmr":
            dataset_conf.gmr_config = {
                "src_human": args.gmr_src_human,
                "target_fps": args.gmr_target_fps,
                "solver": args.gmr_solver,
                "damping": args.gmr_damping,
                "offset_to_ground": args.gmr_offset_to_ground,
                "use_velocity_limit": args.gmr_use_velocity_limit,
            }

        env = ImitationFactory.make(
            args.model,
            amass_dataset_conf=dataset_conf,
            env_params={"timestep": 0.002, "n_substeps": 5},
        )
        motion_labels = (
            list(dict.fromkeys(motions)) if motions else [f"Trajectory {i}" for i in range(env.th.n_trajectories)]
        )

    TrajectoryViserViewer(env, include_collision=args.include_collision).run(motion_labels)


if __name__ == "__main__":
    main()
