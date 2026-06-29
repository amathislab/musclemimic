from dataclasses import dataclass
from functools import lru_cache

from omegaconf import ListConfig

from loco_mujoco.trajectory.dataclasses import Trajectory


@lru_cache(maxsize=1)
def get_amass_dataset_groups() -> dict[str, list[str]]:
    """Lazily load and return the AMASS dataset group mapping."""
    from loco_mujoco.smpl.const import (
        AMASS_BIMANUAL_MARGINAL_MOTIONS,
        AMASS_BIMANUAL_TEST_MOTIONS,
        AMASS_BIMANUAL_TRAIN_MOTIONS,
        AMASS_EKUT_DATASETS,
        AMASS_LOCOMOTION_DATASETS,
        AMASS_RANDOM_TRAINING_MOTIONS,
        AMASS_RANDOM_TRAINING_MOTIONS_2K,
        AMASS_TRANSITION_MOTIONS,
        KIT_KINESIS_TESTING_MOTIONS,
        KIT_KINESIS_TRAINING_MOTIONS,
        KIT_KINESIS_TRAINING_MOTIONS_MINT,
        KIT_KINESIS_TRAINING_MOTIONS_MINT_STRAIGHT_FORWARDS,
        KIT_KINESIS_TRANSITION_TRAINING_MOTIONS,
    )
    return {
        "AMASS_LOCOMOTION_DATASETS": AMASS_LOCOMOTION_DATASETS,
        "AMASS_EKUT_DATASETS": AMASS_EKUT_DATASETS,
        "AMASS_BIMANUAL_MARGINAL_MOTIONS": AMASS_BIMANUAL_MARGINAL_MOTIONS,
        "KIT_KINESIS_TRAINING_MOTIONS": KIT_KINESIS_TRAINING_MOTIONS,
        "KIT_KINESIS_TESTING_MOTIONS": KIT_KINESIS_TESTING_MOTIONS,
        "AMASS_TRANSITION_MOTIONS": AMASS_TRANSITION_MOTIONS,
        "KIT_KINESIS_TRANSITION_TRAINING_MOTIONS": KIT_KINESIS_TRANSITION_TRAINING_MOTIONS,
        "KIT_KINESIS_TRAINING_MOTIONS_MINT": KIT_KINESIS_TRAINING_MOTIONS_MINT,
        "KIT_KINESIS_TRAINING_MOTIONS_MINT_STRAIGHT_FORWARDS": KIT_KINESIS_TRAINING_MOTIONS_MINT_STRAIGHT_FORWARDS,
        "AMASS_RANDOM_TRAINING_MOTIONS": AMASS_RANDOM_TRAINING_MOTIONS,
        "AMASS_RANDOM_TRAINING_MOTIONS_2K": AMASS_RANDOM_TRAINING_MOTIONS_2K,
        "AMASS_BIMANUAL_TEST_MOTIONS": AMASS_BIMANUAL_TEST_MOTIONS,
        "AMASS_BIMANUAL_TRAIN_MOTIONS": AMASS_BIMANUAL_TRAIN_MOTIONS,
    }


def expand_amass_dataset_group_spec(dataset_group: str | list | None) -> list[str]:
    """Normalize dataset_group into a flat list of AMASS group names.

    Accepts a single group name, a '+'-joined string of names, or a list thereof.
    """
    if dataset_group is None:
        return []

    if isinstance(dataset_group, (ListConfig, list, tuple)):
        group_names = []
        for group in dataset_group:
            group_names.extend(expand_amass_dataset_group_spec(group))
        return group_names

    if isinstance(dataset_group, str):
        return [group.strip() for group in dataset_group.split("+") if group.strip()]

    raise TypeError(
        "AMASS dataset_group must be a string, a '+'-joined string, or a list of strings, "
        f"got {type(dataset_group).__name__}."
    )


# @dataclass
# class DefaultDatasetConf:
#     """
#     Configuration for loading default datasets provided by LocoMuJoCo.

#     Attributes:
#         dataset_type (str): The type of the dataset to load. Can be "mocap" or "pretrained".
#         task (str): The task to load.
#         debug (bool): Whether to load the dataset in debug mode.

#     """

#     task: Union[str, list]  = "walk"
#     dataset_type: str = "mocap"
#     debug: bool = False

#     def __post_init__(self):
#         assert self.dataset_type in ["mocap", "pretrained"], f"Unknown dataset type: {self.dataset_type}"


@dataclass
class AMASSDatasetConf:
    """
    Configuration for loading AMASS datasets.

    Attributes:
        rel_dataset_path (Union[str, list]): A relative path or a list of relative paths to
            load from the AMASS dataset.
        dataset_group (Union[str, list]): A predefined dataset group name, a list of group names,
            or a `GROUP_A + GROUP_B` combination string to load from AMASS.
        retargeting_method (str): The retargeting method to use (e.g., 'smpl', 'gmr'). Optional.
        gmr_config (dict): Configuration for GMR retargeting. Optional.
        max_motions (int, optional): If set, cap the number of motion paths loaded by
            sampling a subset of this size.
        clear_cache (bool): If True, overwrite existing cached retargeted files instead of loading them.
        trajectory_cache_type (str): Runtime trajectory cache layout. Use "sparse"
            to store only mimic site caches, "full" for all body/site caches, or
            "none" for qpos/qvel only.

    """
    rel_dataset_path: str | list = None
    dataset_group: str | list = None
    retargeting_method: str = None
    gmr_config: dict = None
    max_motions: int | None = None
    clear_cache: bool = False
    trajectory_cache_type: str = "sparse"

    def __post_init__(self):
        assert self.rel_dataset_path is not None or self.dataset_group is not None, ("Either `rel_dataset_path` or "
                                                                                     "`dataset_group` must be set.")
        if self.max_motions is not None:
            self.max_motions = int(self.max_motions)
            if self.max_motions <= 0:
                raise ValueError(f"max_motions must be > 0, got {self.max_motions}")


@dataclass
class C3DDatasetConf:
    """
    Configuration for loading converted C3D-derived trajectories.

    Attributes:
        rel_dataset_path (Union[str, list]): A relative path or a list of relative paths
            under the converted C3D cache root.
        retargeting_method (str): The retargeting method namespace used when saving
            the C3D trajectory. Defaults to 'gmr'.
        max_motions (int, optional): If set, cap the number of motion paths loaded by
            sampling a subset of this size.
        trajectory_cache_type (str): Runtime trajectory cache layout. Use "sparse"
            to store only mimic site caches, "full" for all body/site caches, or
            "none" for qpos/qvel only.

    """

    rel_dataset_path: str | list = None
    retargeting_method: str = "gmr"
    max_motions: int | None = None
    trajectory_cache_type: str = "sparse"

    def __post_init__(self):
        assert self.rel_dataset_path is not None, "`rel_dataset_path` must be set for converted C3D datasets."
        if self.retargeting_method not in {"smpl", "gmr"}:
            raise ValueError(f"Unknown C3D retargeting method: {self.retargeting_method}")
        if self.max_motions is not None:
            self.max_motions = int(self.max_motions)
            if self.max_motions <= 0:
                raise ValueError(f"max_motions must be > 0, got {self.max_motions}")


@dataclass
class LAFAN1DatasetConf:
    """
    Configuration for loading LAFAN1 datasets.

    Attributes:
        dataset_name (Union[str, list]): A name of a dataset or a list of dataset names to load from LAFAN1.
        dataset_group (str): A name of a predefined group of datasets to load from LAFAN1.

    ..note:: This datatset is loaded from the LocoMuJoCo's HuggingFace repository:
        https://huggingface.co/datasets/robfiras/loco-mujoco-datasets. It provides datasets for
        all humanoid environments.

    """

    dataset_name: str | list = None
    dataset_group: str = None
    trajectory_cache_type: str = "full"

    def __post_init__(self):
        assert self.dataset_name is not None or self.dataset_group is not None, ("Either `dataset_name` or "
                                                                                 "`dataset_group` must be set.")


@dataclass
class CustomDatasetConf:
    """
    Configuration for loading custom trajectories.

    Attributes:
        traj (Trajectory): A custom trajectory to load.

    """
    traj: Trajectory
    trajectory_cache_type: str = "full"
