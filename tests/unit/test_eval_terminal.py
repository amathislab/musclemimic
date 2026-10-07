from types import SimpleNamespace

from omegaconf import OmegaConf

from fullbody._eval_terminal import (
    apply_eval_reward_termination_defaults,
    apply_terminal_cli_overrides,
)


def test_eval_reward_termination_uses_validation_defaults():
    env_params = {
        "reward_termination_type": "LowRewardTerminationHandler",
        "reward_termination_params": {"threshold": 0.968, "patience": 100},
    }
    config = OmegaConf.create(
        {
            "experiment": {
                "validation": {
                    "reward_termination_type": "NoRewardTerminationHandler",
                    "reward_termination_params": {},
                }
            }
        }
    )

    apply_eval_reward_termination_defaults(env_params, config, strict_termination=False)

    assert env_params["reward_termination_type"] == "NoRewardTerminationHandler"
    assert env_params["reward_termination_params"] == {}


def test_eval_reward_termination_strict_keeps_training_defaults():
    env_params = {
        "reward_termination_type": "LowRewardTerminationHandler",
        "reward_termination_params": {"threshold": 0.968, "patience": 100},
    }
    config = OmegaConf.create(
        {
            "experiment": {
                "validation": {
                    "reward_termination_type": "NoRewardTerminationHandler",
                    "reward_termination_params": {},
                }
            }
        }
    )

    apply_eval_reward_termination_defaults(env_params, config, strict_termination=True)

    assert env_params["reward_termination_type"] == "LowRewardTerminationHandler"
    assert env_params["reward_termination_params"] == {"threshold": 0.968, "patience": 100}


def test_eval_no_termination_disables_state_and_reward_termination():
    env_params = {
        "terminal_state_type": "MeanRelativeSiteDeviationTerminalStateHandler",
        "reward_termination_type": "LowRewardTerminationHandler",
        "reward_termination_params": {"threshold": 0.968, "patience": 100},
    }
    args = SimpleNamespace(
        no_termination=True,
        terminal_state_type=None,
        mean_site_deviation_threshold=None,
        root_deviation_threshold=None,
        root_orientation_threshold=None,
        root_site=None,
    )

    apply_terminal_cli_overrides(env_params, args)

    assert env_params["terminal_state_type"] == "NoTerminalStateHandler"
    assert env_params["reward_termination_type"] == "NoRewardTerminationHandler"
    assert env_params["reward_termination_params"] == {}
