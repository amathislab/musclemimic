import random

from loco_mujoco.smpl.motion_spec import AMASSMotionSpec
from loco_mujoco.task_factories import imitation_factory
from loco_mujoco.task_factories.dataset_confs import AMASSDatasetConf
from loco_mujoco.task_factories.imitation_factory import ImitationFactory


def test_amass_max_motions_subset_is_reproducible(monkeypatch):
    motion_paths = [f"motion_{index}" for index in range(20)]
    monkeypatch.setattr(
        imitation_factory,
        "get_amass_dataset_groups",
        lambda: {"TEST_MOTIONS": motion_paths},
    )
    config = AMASSDatasetConf(
        dataset_group="TEST_MOTIONS",
        max_motions=5,
        max_motions_seed=7,
    )

    random.seed(1)
    first = ImitationFactory.select_amass_dataset_paths(config)
    random.seed(999)
    second = ImitationFactory.select_amass_dataset_paths(config)

    expected = sorted(random.Random(7).sample(sorted(motion_paths), 5))
    assert first == second == expected


def test_amass_group_selection_retains_phc_time_bounds(monkeypatch):
    motion_specs = [
        AMASSMotionSpec("motion_b", end_time_s=2.0),
        AMASSMotionSpec("motion_a", end_time_s=1.0),
    ]
    monkeypatch.setattr(
        imitation_factory,
        "get_amass_dataset_groups",
        lambda: {"TEST_MOTIONS": [spec.path for spec in motion_specs]},
    )
    monkeypatch.setattr(
        imitation_factory,
        "get_amass_motion_spec_groups",
        lambda: {"TEST_MOTIONS": motion_specs},
    )

    selected = ImitationFactory.select_amass_motion_specs(AMASSDatasetConf(dataset_group="TEST_MOTIONS"))

    assert selected == [motion_specs[1], motion_specs[0]]
