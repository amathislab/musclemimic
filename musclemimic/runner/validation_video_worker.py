"""Subprocess entrypoint for asynchronous validation video recording.

CPU-only JAX cannot be enforced from inside this module: importing the
musclemimic package (which `python -m` does before main() runs) already
imports jax, and jax reads JAX_PLATFORMS once at import time. The spawning
AsyncValidationVideoRecorder therefore injects the CPU-only environment via
configure_validation_video_worker_env at spawn time; main() only verifies it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _write_result(path: str | Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: python -m musclemimic.runner.validation_video_worker <payload.json>")
        return 2

    if os.environ.get("JAX_PLATFORMS") != "cpu":
        print(
            "validation_video_worker must run with JAX_PLATFORMS=cpu set in its "
            "environment before launch (the spawning recorder does this).",
            file=sys.stderr,
        )
        return 2

    payload_path = Path(argv[0])
    with open(payload_path) as f:
        payload = json.load(f)

    result_path = payload["result_path"]
    try:
        import jax
        from omegaconf import OmegaConf

        from loco_mujoco.core.stateful_object import StatefulObject
        from musclemimic.runner.engine import pick_algorithm
        from musclemimic.runner.validation_video_checkpoint import (
            load_validation_policy_checkpoint,
        )
        from musclemimic.runner.validation_video_recorder import ValidationVideoRecorder
        from musclemimic.utils import setup_headless_rendering_if_needed

        setup_headless_rendering_if_needed()

        policy_checkpoint = load_validation_policy_checkpoint(
            payload["checkpoint_path"],
            timeout_s=float(payload["checkpoint_wait_timeout_s"]),
        )
        config = OmegaConf.create(policy_checkpoint.config)
        policy_state = jax.tree_util.tree_map(
            jax.numpy.asarray,
            policy_checkpoint.policy_state,
        )
        del policy_checkpoint

        recorder = ValidationVideoRecorder(
            video_dir=payload["video_dir"],
            length=int(payload["length"]),
            deterministic=bool(payload["deterministic"]),
        )
        config_holder = SimpleNamespace(config=config)

        saved_instances = StatefulObject._instances.copy()
        StatefulObject._instances.clear()
        env = None
        try:
            env, env_params = recorder.make_recording_env(
                config_holder,
                payload["tag"],
                validation_number=int(payload["validation_number"]),
            )
            algorithm_cls = pick_algorithm(config)
            agent_conf = algorithm_cls.init_agent_conf(env, config)
            agent_state = algorithm_cls.build_inference_state(policy_state, agent_conf)
            recorder.run_recording_rollout(
                algorithm_cls,
                env,
                agent_conf,
                agent_state,
                env_params,
            )
        finally:
            if env is not None:
                env.stop()
            StatefulObject._instances = saved_instances

        video_path = env.video_file_path if env is not None else None
        _write_result(
            result_path,
            {
                "ok": bool(video_path),
                "video_path": video_path,
                "validation_number": payload["validation_number"],
                "timestep": payload["timestep"],
            },
        )
        return 0 if video_path else 1
    except Exception as exc:
        import traceback

        traceback_text = traceback.format_exc()
        print(traceback_text, file=sys.stderr)
        _write_result(
            result_path,
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback_text,
                "validation_number": payload.get("validation_number"),
                "timestep": payload.get("timestep"),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
