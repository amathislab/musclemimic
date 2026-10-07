from omegaconf import OmegaConf

from fullbody.eval import _recording_motion_name


def test_recording_motion_name_uses_amass_path() -> None:
    task_params = OmegaConf.create(
        {
            "amass_dataset_conf": {"rel_dataset_path": ["KIT/9/walking_run07_poses"]},
        }
    )

    assert _recording_motion_name(task_params, traj_index=None) == "KIT_9_walking_run07_poses"


