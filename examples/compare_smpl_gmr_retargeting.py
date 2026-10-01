from musclemimic.utils import setup_headless_rendering_if_needed, detect_headless_environment

# Setup rendering before other imports
setup_headless_rendering_if_needed()

from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf


def create_env(retargeting_method="smpl", motion_file="KIT/3/tennis_forehand_right04_poses"):
    """
    Create MyoFullBody environment with specified retargeting method.

    Args:
        retargeting_method: "smpl" (optimization-based) or "gmr" (IK-based)
        motion_file: AMASS motion file to retarget

    Returns:
        Environment with retargeted trajectory loaded
    """
    is_headless = detect_headless_environment()
    amass_conf = AMASSDatasetConf([motion_file])

    # Add GMR configuration if using GMR retargeting
    if retargeting_method == "gmr":
        amass_conf.retargeting_method = "gmr"
        amass_conf.gmr_config = {
            "src_human": "smplh",
            "target_fps": 30,
            "solver": "daqp",
            "damping": 0.5,
            "offset_to_ground": True,
            "use_velocity_limit": False,
            "verbose": False,
        }

    env = ImitationFactory.make(
        "MyoFullBody",
        amass_dataset_conf=amass_conf,
        env_params={
            "timestep": 0.002,  # 2ms timestep
            "n_substeps": 5,    # 100Hz control frequency
        },
        default_camera_mode="static",
        camera_params={
            "static": {
                "distance": 3.0,
                "elevation": -15.0,
                "azimuth": 45.0,
                "lookat": [0.0, 0.0, 1.0]
            }
        },
        headless=is_headless,
    )

    return env


def print_trajectory_info(env, method_name):
    """Print trajectory statistics."""
    traj = env.th.traj
    duration = (len(traj.data.qpos) - 1) / traj.info.frequency

    print(f"\n{method_name.upper()} RETARGETING:")
    print(f"  Trajectory frames: {len(traj.data.qpos)}")
    print(f"  Trajectory frequency: {traj.info.frequency:.1f} Hz")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Control frequency: {1.0/env.dt:.1f} Hz")
    print(f"  qpos shape: {traj.data.qpos.shape}")
    print(f"  qvel shape: {traj.data.qvel.shape}")


def visualize_trajectory(env, method_name, n_episodes=2, record=True, video_name="tennis_forehand"):
    """Visualize and optionally record trajectory playback."""
    print(f"\nVisualizing {method_name} retargeting...")

    recorder_params = None
    if record:
        recorder_params = {
            "path": "./retargeting_recordings",
            "tag": f"myofullbody_{method_name}",
            "video_name": video_name,
            "compress": True
        }
        print(f"  Recording to: {recorder_params['path']}/{recorder_params['tag']}/")

    env.play_trajectory(
        n_episodes=n_episodes,
        n_steps_per_episode=400,
        render=True,
        record=record,
        recorder_params=recorder_params
    )

    if record:
        print(f"Recording saved")


def main():
    """Compare SMPL and GMR retargeting methods."""
    print("=" * 60)
    print("SMPL vs GMR Retargeting Comparison")
    print("=" * 60)

    motion_file = "KIT/3/tennis_forehand_left04_poses"

    # Compare both methods
    for method in ["smpl", "gmr"]:
        print(f"\n{'=' * 60}")
        print(f"Testing {method.upper()} retargeting...")
        print('=' * 60)

        # Create environment with specified retargeting method
        env = create_env(retargeting_method=method, motion_file=motion_file)

        # Print trajectory information
        print_trajectory_info(env, method)

        # Visualize and record
        visualize_trajectory(env, method, n_episodes=2, record=True, video_name=motion_file.split('/')[-1])


    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print("\nRecordings saved in ./retargeting_recordings/")
    print("  • myofullbody_smpl/tennis_forehand.mp4")
    print("  • myofullbody_gmr/tennis_forehand.mp4")
    print("\nCompare the videos to see the difference between:")
    print("  SMPL: Optimization-based retargeting")
    print("  GMR:  IK-based retargeting with equality constraints")


if __name__ == "__main__":
    main()
