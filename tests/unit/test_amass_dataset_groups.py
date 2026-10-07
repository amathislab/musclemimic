from loco_mujoco.task_factories.dataset_confs import get_amass_dataset_groups


def test_cleaned_amass_training_group_is_complete_and_unique():
    motions = get_amass_dataset_groups()["AMASS_CLEANED_TRAINING_MOTIONS"]

    assert len(motions) == 9_796
    assert len(set(motions)) == len(motions)
    assert motions == sorted(motions)
    assert all(path and not path.startswith("/") and not path.endswith(".npz") for path in motions)
