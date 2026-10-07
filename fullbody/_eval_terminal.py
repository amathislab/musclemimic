from __future__ import annotations

import argparse
from typing import Any

from omegaconf import OmegaConf

from musclemimic.runner.validation_config import apply_validation_reward_termination

TERMINAL_PARAM_SUPPORT = {
    "MeanSiteDeviationTerminalStateHandler": {"mean_site_deviation_threshold"},
    "MeanRelativeSiteDeviationTerminalStateHandler": {"mean_site_deviation_threshold"},
    "MeanRelativeSiteDeviationWithRootTerminalStateHandler": {
        "mean_site_deviation_threshold",
        "root_deviation_threshold",
        "root_orientation_threshold",
        "root_site",
    },
}


def apply_eval_terminal_defaults(env_params: dict[str, Any], config: Any, strict_termination: bool) -> None:
    """Apply default terminal-state settings for fullbody evaluation."""
    if strict_termination:
        print("Using strict training terminal state configuration")
        return

    validation_cfg = config.experiment.get("validation", None)
    if validation_cfg is None:
        return

    terminal_state_type = validation_cfg.get("terminal_state_type", None)
    if terminal_state_type is not None:
        env_params["terminal_state_type"] = terminal_state_type

    terminal_state_params = validation_cfg.get("terminal_state_params", None)
    if terminal_state_params is not None:
        env_params["terminal_state_params"] = OmegaConf.to_container(terminal_state_params, resolve=True)

    print("Using validation terminal state configuration for evaluation")


def apply_eval_reward_termination_defaults(
    env_params: dict[str, Any],
    config: Any,
    strict_termination: bool,
) -> None:
    """Apply validation reward-termination settings for fullbody evaluation."""
    if strict_termination:
        return

    validation_cfg = config.experiment.get("validation", None)
    if validation_cfg is None:
        return

    apply_validation_reward_termination(env_params, validation_cfg)


def apply_terminal_cli_overrides(env_params: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI terminal-state overrides on top of evaluation defaults."""
    if args.no_termination:
        env_params["terminal_state_type"] = "NoTerminalStateHandler"
        env_params["reward_termination_type"] = "NoRewardTerminationHandler"
        env_params["reward_termination_params"] = {}
        print("Early termination disabled")

    if args.terminal_state_type is not None:
        env_params["terminal_state_type"] = args.terminal_state_type
        print(f"Terminal state handler override: {args.terminal_state_type}")

    terminal_overrides = _collect_terminal_overrides(args)
    if not terminal_overrides:
        return

    active_handler = env_params.get("terminal_state_type")
    supported = TERMINAL_PARAM_SUPPORT.get(active_handler, set())
    unsupported = sorted(set(terminal_overrides) - supported)
    if unsupported:
        flags = ", ".join(f"--{name}" for name in unsupported)
        raise ValueError(f"Terminal handler '{active_handler}' does not support {flags}.")

    terminal_state_params = env_params.get("terminal_state_params")
    if terminal_state_params is None:
        terminal_state_params = {}
        env_params["terminal_state_params"] = terminal_state_params
    for name, value in terminal_overrides.items():
        terminal_state_params[name] = value
        print(f"Terminal override: {name}={value}")


def _collect_terminal_overrides(args: argparse.Namespace) -> dict[str, Any]:
    terminal_overrides: dict[str, Any] = {}
    if args.mean_site_deviation_threshold is not None:
        terminal_overrides["mean_site_deviation_threshold"] = args.mean_site_deviation_threshold
    if args.root_deviation_threshold is not None:
        terminal_overrides["root_deviation_threshold"] = args.root_deviation_threshold
    if args.root_orientation_threshold is not None:
        terminal_overrides["root_orientation_threshold"] = args.root_orientation_threshold
    if args.root_site is not None:
        terminal_overrides["root_site"] = args.root_site
    return terminal_overrides
