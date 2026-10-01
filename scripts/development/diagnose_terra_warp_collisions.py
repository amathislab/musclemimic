"""Run one TERRA MJX-Warp reset with a selected subset of explicit geom pairs.

Each invocation is intentionally a separate process: a native Warp/CUDA crash cannot be
caught by Python.  This is a diagnostic, not a training entrypoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_indices(value: str, count: int) -> set[int]:
    if value == "all":
        return set(range(count))
    if value == "none":
        return set()

    result: set[int] = set()
    for part in value.split(","):
        if ":" in part:
            start, stop = part.split(":", 1)
            result.update(range(int(start), int(stop)))
        else:
            result.add(int(part))
    invalid = sorted(index for index in result if index < 0 or index >= count)
    if invalid:
        raise ValueError(f"Pair indices outside [0, {count}): {invalid}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-pairs", default="all", help="all, none, 0:20, or comma-separated indices")
    parser.add_argument(
        "--pair-kind",
        choices=("all", "capsule_only", "noncapsule"),
        default="all",
        help="Optionally filter by geom type; indices refer to MjSpec order, not compiled-model order.",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Converted-AMASS root (default: MuscleMimic's configured cache root).",
    )
    parser.add_argument("--cache-subdir", default="terra")
    parser.add_argument("--nconmax", type=int, default=192)
    args = parser.parse_args()

    import jax
    import mujoco
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from loco_mujoco.smpl.retargeting import get_converted_amass_dataset_path
    from loco_mujoco.task_factories import TaskFactory
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody

    original_modify = MyoFullBody._modify_spec_for_mjx

    def filter_pairs(self, spec):
        spec = original_modify(self, spec)
        pair_count = len(spec.pairs)
        keep = _parse_indices(args.keep_pairs, pair_count)
        if args.pair_kind != "all":
            geom_types = {geom.name: geom.type for geom in spec.geoms}
            capsule = mujoco.mjtGeom.mjGEOM_CAPSULE
            capsule_pairs = {
                index
                for index, pair in enumerate(spec.pairs)
                if geom_types[pair.geomname1] == capsule and geom_types[pair.geomname2] == capsule
            }
            keep &= capsule_pairs if args.pair_kind == "capsule_only" else set(range(pair_count)) - capsule_pairs
        # Re-index into the live MjSpec sequence on every deletion. MjSpec element
        # wrappers are index-backed, so retaining wrappers while the list shrinks can
        # make a later wrapper refer to a different pair.
        for index in reversed(range(pair_count)):
            if index not in keep:
                spec.delete(spec.pairs[index])
        print(f"PAIR_FILTER original={pair_count} retained={len(keep)} indices={sorted(keep)}", flush=True)
        return spec

    MyoFullBody._modify_spec_for_mjx = filter_pairs

    cache_root = args.cache_root or get_converted_amass_dataset_path()
    repo = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo / "fullbody"), version_base=None):
        config = compose(
            config_name="conf_terra_flashsac_single",
            overrides=[
                "experiment.env_params.num_envs=1",
                f"experiment.env_params.nconmax={args.nconmax}",
                f"experiment.task_factory.params.amass_dataset_conf.output_cache_subdir={args.cache_subdir}",
                f"experiment.task_factory.params.amass_dataset_conf.cache_root={cache_root}",
            ],
        )

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    task_params = OmegaConf.to_container(config.experiment.task_factory.params, resolve=True)
    print("CONSTRUCTING_ENV", flush=True)
    env = factory.make(**env_params, **task_params)
    model = env._model
    pair_names = []
    for index in range(model.npair):
        geom1 = int(model.pair_geom1[index])
        geom2 = int(model.pair_geom2[index])
        pair_names.append(
            (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            )
        )
    print(f"ENV_READY npair={model.npair} pairs={pair_names}", flush=True)

    env.th.to_jax()
    print("RESETTING", flush=True)
    state = env.mjx_reset(jax.random.PRNGKey(0))
    jax.block_until_ready(state.observation)
    print(f"RESET_OK observation_shape={state.observation.shape}", flush=True)


if __name__ == "__main__":
    main()
