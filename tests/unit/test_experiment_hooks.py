from unittest.mock import MagicMock, call

from musclemimic.runner.logging import ExperimentHooks


def test_named_validation_videos_use_separate_wandb_keys():
    wandb = MagicMock()
    wandb.Video.side_effect = lambda path, format: (path, format)

    ExperimentHooks().on_validation_video(
        True,
        wandb,
        {"stairs_up": "/tmp/up.mp4", "ramp_down": "/tmp/down.mp4"},
        123,
    )

    assert wandb.Video.call_args_list == [
        call("/tmp/up.mp4", format="mp4"),
        call("/tmp/down.mp4", format="mp4"),
    ]
    wandb.log.assert_called_once_with(
        {
            "Validation/Video/stairs_up": ("/tmp/up.mp4", "mp4"),
            "Validation/Video/ramp_down": ("/tmp/down.mp4", "mp4"),
        },
        step=123,
    )
