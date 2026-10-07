from dataclasses import dataclass
from functools import lru_cache

from omegaconf import ListConfig

from loco_mujoco.smpl.gmr_cache_identity import GMR_CACHE_NAMESPACE
from loco_mujoco.smpl.motion_manifest import (
    load_amass_phc_motion_groups,
    load_amass_phc_quality_motion_groups,
)
from loco_mujoco.smpl.motion_spec import AMASSMotionSpec, as_amass_motion_spec
from loco_mujoco.trajectory.dataclasses import Trajectory

GMR_TRAINING_CACHE_NAMESPACE = GMR_CACHE_NAMESPACE


def resolve_training_gmr_cache_namespace(
    retargeting_method: str | None,
    requested_namespace: str | None,
) -> str | None:
    """Resolve the canonical GMR namespace used by imitation training."""
    if retargeting_method != "gmr":
        return requested_namespace
    resolved_namespace = requested_namespace or GMR_TRAINING_CACHE_NAMESPACE
    if resolved_namespace != GMR_TRAINING_CACHE_NAMESPACE:
        raise ValueError(f"GMR training data must use the canonical gmr_v6 cache namespace, got {resolved_namespace!r}")
    return resolved_namespace


@lru_cache(maxsize=1)
def get_amass_dataset_groups() -> dict[str, list[str]]:
    """Lazily load and return the AMASS dataset group mapping."""
    from loco_mujoco.smpl.const import (
        AMASS_BIMANUAL_MARGINAL_MOTIONS,
        AMASS_BIMANUAL_TEST_MOTIONS,
        AMASS_BIMANUAL_TRAIN_MOTIONS,
        AMASS_CLEANED_TRAINING_MOTIONS,
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

    groups = {
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
        "AMASS_CLEANED_TRAINING_MOTIONS": AMASS_CLEANED_TRAINING_MOTIONS,
        "AMASS_BIMANUAL_TEST_MOTIONS": AMASS_BIMANUAL_TEST_MOTIONS,
        "AMASS_BIMANUAL_TRAIN_MOTIONS": AMASS_BIMANUAL_TRAIN_MOTIONS,
    }
    groups.update(
        {group_name: [spec.path for spec in specs] for group_name, specs in load_amass_phc_motion_groups().items()}
    )
    groups.update(
        {
            group_name: [spec.path for spec in specs]
            for group_name, specs in load_amass_phc_quality_motion_groups().items()
        }
    )
    return groups


@lru_cache(maxsize=1)
def get_amass_motion_spec_groups() -> dict[str, list[AMASSMotionSpec]]:
    """Return AMASS groups with optional source-time bounds."""
    groups = {
        group_name: [as_amass_motion_spec(path) for path in paths]
        for group_name, paths in get_amass_dataset_groups().items()
    }
    groups.update({group_name: list(specs) for group_name, specs in load_amass_phc_motion_groups().items()})
    groups.update({group_name: list(specs) for group_name, specs in load_amass_phc_quality_motion_groups().items()})
    return groups


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
        rel_dataset_path: A relative path, an ``AMASSMotionSpec``, or a list of
            either type to load from the AMASS dataset.
        dataset_group (Union[str, list]): A predefined dataset group name, a list of group names,
            or a `GROUP_A + GROUP_B` combination string to load from AMASS.
        retargeting_method (str): The retargeting method to use (e.g., 'smpl', 'gmr'). Optional.
        gmr_config (dict): Configuration for GMR retargeting. Optional.
        gmr_cache_namespace: Cache namespace for precomputed GMR trajectories.
            GMR training is restricted to ``gmr_v6``.
        max_motions (int, optional): If set, cap the number of motion paths loaded by
            sampling a subset of this size.
        max_motions_seed (int): Seed used to select a reproducible capped subset.
        clear_cache (bool): If True, overwrite existing cached retargeted files instead of loading them.
        trajectory_cache_type (str): Runtime trajectory cache layout. Use "sparse"
            to store only mimic site caches, "full" for all body/site caches, or
            "none" for qpos/qvel only.
        write_materialized_cache (bool): If True, store and reuse persistent materialized AMASS caches.
            Defaults to False so existing and mixed-dataset configurations keep
            using the runtime loader.
        rebuild_materialized_cache (bool): Explicitly rebuild an existing materialized cache.

    """

    rel_dataset_path: str | AMASSMotionSpec | list = None
    dataset_group: str | list = None
    retargeting_method: str = None
    gmr_config: dict = None
    max_motions: int | None = None
    max_motions_seed: int = 0
    clear_cache: bool = False
    trajectory_cache_type: str = "sparse"
    write_materialized_cache: bool = False
    rebuild_materialized_cache: bool = False
    gmr_cache_namespace: str | None = None

    def __post_init__(self):
        assert self.rel_dataset_path is not None or self.dataset_group is not None, (
            "Either `rel_dataset_path` or `dataset_group` must be set."
        )
        if self.max_motions is not None:
            self.max_motions = int(self.max_motions)
            if self.max_motions <= 0:
                raise ValueError(f"max_motions must be > 0, got {self.max_motions}")
        self.max_motions_seed = int(self.max_motions_seed)
        if self.gmr_cache_namespace is not None:
            self.gmr_cache_namespace = str(self.gmr_cache_namespace)
            if (
                not self.gmr_cache_namespace
                or self.gmr_cache_namespace in {".", ".."}
                or "/" in self.gmr_cache_namespace
                or "\\" in self.gmr_cache_namespace
            ):
                raise ValueError(
                    f"gmr_cache_namespace must be one non-empty path component, got {self.gmr_cache_namespace!r}"
                )
        self.gmr_cache_namespace = resolve_training_gmr_cache_namespace(
            self.retargeting_method,
            self.gmr_cache_namespace,
        )


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
        assert self.dataset_name is not None or self.dataset_group is not None, (
            "Either `dataset_name` or `dataset_group` must be set."
        )


@dataclass
class CustomDatasetConf:
    """
    Configuration for loading custom trajectories.

    Attributes:
        traj (Trajectory): A custom trajectory to load.

    """

    traj: Trajectory
    trajectory_cache_type: str = "full"
