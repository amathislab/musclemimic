import os
from typing import Union, List

from omegaconf import DictConfig
from huggingface_hub import hf_hub_download

import loco_mujoco
from loco_mujoco.smpl.retargeting import load_robot_conf_file
from musclemimic.environments import LocoEnv
from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryCacheType,
    TrajectoryHandler,
)
from loco_mujoco.utils import setup_logger


def extend_motion(
        env_name: str,
        robot_conf: DictConfig,
        traj: Trajectory,
        replace_qvel_with_finite_diff: bool,
) -> Trajectory:
    """
    Extend a motion trajectory to include more model-specific entities
    like body xpos, site positions, etc. and to match the environment's frequency.

    Args:
        env_name (str): Name of the environment.
        robot_conf (DictConfig): Configuration of the robot.
        traj (Trajectory): The original trajectory data.

    Returns:
        Trajectory: The extended trajectory.

    """

    assert traj.data.n_trajectories == 1

    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))

    return TrajectoryHandler(
        env._model,
        traj=traj,
        control_dt=env.dt,
        random_start=False,
        fixed_start_conf=(0, 0),
        cache_type=TrajectoryCacheType.NONE,
    ).traj


def load_lafan1_trajectory(
        env_name: str,
        dataset_name: Union[str, List[str]],
        max_steps: int = 100
) -> Trajectory:
    """
    Load a trajectory from the LAFAN1 dataset.

    Args:
        env_name (str): The name of the environment.
        dataset_name (Union[str, List[str]]): The name of the dataset(s) to load.
        max_steps (int, optional): The maximum number of steps to optimize for collisions. Defaults to 100.

    Returns:
        Trajectory: The loaded trajectory.

    """
    logger = setup_logger("lafan1", identifier="[LocoMuJoCo's LAFAN1 Retargeting Pipeline]")

    if "Mjx" in env_name:
        env_name = env_name.replace("Mjx", "")

    path_to_convert_lafan1_datasets = loco_mujoco.load_path_config().get("MUSCLEMIMIC_CONVERTED_LAFAN1_PATH")

    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    all_trajectories = []
    for d_name in dataset_name:

        if path_to_convert_lafan1_datasets:
            target_path_dataset = os.path.join(path_to_convert_lafan1_datasets, env_name, f"{d_name}.npz")
        else:
            target_path_dataset = None

        if path_to_convert_lafan1_datasets:
            # check if file exists
            if os.path.exists(target_path_dataset):
                logger.info(f"Found converted dataset at: {target_path_dataset}.")
                traj = Trajectory.load(target_path_dataset)
                all_trajectories.append(traj)
                continue

        # load the npz file
        d_name = d_name if d_name.endswith(".npz") else f"{d_name}.npz"

        file_path = hf_hub_download(
            repo_id="robfiras/loco-mujoco-datasets",
            filename=f"Lafan1/mocap/{env_name}/{d_name}",
            repo_type="dataset"
        )

        traj = Trajectory.load(file_path)

        # extend the motion to the desired length
        if not traj.data.is_complete:
            logger.info("Using Mujoco's kinematics to calculate other model-specific entities ...")
            traj = extend_motion(env_name, load_robot_conf_file(env_name), traj, replace_qvel_with_finite_diff=False)

        if path_to_convert_lafan1_datasets:
            traj.save(target_path_dataset)

        all_trajectories.append(traj)

    # concatenate trajectories
    if len(all_trajectories) == 1:
        trajectory = all_trajectories[0]
    else:
        logger.info("Concatenating trajectories ...")
        traj_datas = [t.data for t in all_trajectories]
        traj_infos = [t.info for t in all_trajectories]
        traj_data, traj_info = TrajectoryData.concatenate(traj_datas, traj_infos)
        trajectory = Trajectory(traj_info, traj_data)

    logger.info("Trajectory data loaded!")

    return trajectory


if __name__ == "__main__":
    from loco_mujoco.datasets.humanoids.LAFAN1 import LAFAN1_ALL_DATASETS

    traj = load_lafan1_trajectory("UnitreeH1", LAFAN1_ALL_DATASETS)
    print("Done!")
