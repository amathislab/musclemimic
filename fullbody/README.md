# Full-body experiment directory

`fullbody.experiment` is the training module. `fullbody.eval` evaluates,
records, or visualizes a saved checkpoint. Root-level `conf_fullbody*.yaml`
files are runnable Hydra presets or shared parents for runnable presets.

The reusable inheritance path is:

```text
conf_fullbody
  -> conf_fullbody_gmr
    -> conf_fullbody_gmr_resnet
      -> conf_fullbody_gmr_resnet_large
      -> conf_fullbody_gmr_resnet_4gpu
    -> conf_fullbody_flashsac
```

Directories below the root contain Hydra configuration groups. They encode one
experiment axis and are selected as command-line overrides:

```text
fullbody/
  observation/           tracking default or legacy checkpoint compatibility
  flashsac/termination/  FlashSAC termination profiles
```

Run commands from the repository root. Resolve a configuration before allocating
compute:

```bash
uv run python -m fullbody.experiment \
  --config-name=conf_fullbody_gmr_resnet \
  --cfg job --resolve
```

The complete experiment catalog, data requirements, launch commands, checkpoint
behavior, evaluation commands, and maintenance rules are in
[`docs/fullbody_experiments.md`](../docs/fullbody_experiments.md).

Muscle activation and GRF export, sampling-time conventions, and MM-EvalKit
commands are documented in
[`docs/physiology_export.md`](../docs/physiology_export.md).

Hydra writes per-launch logs to `outputs/`. Auto-resume checkpoints use
`checkpoints/<run_id-or-config-hash>/` unless a config sets `checkpoint_root`.
Both paths are ignored by Git. Generated caches and experiment outputs do not
belong in this directory.
