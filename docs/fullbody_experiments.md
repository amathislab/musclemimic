# Full-body experiment guide

This document defines the current `fullbody/` experiment layout and the common
reproduction workflow. Run every command from the repository root.

## Setup and data

The project uses Python 3.11 and `uv`. A full GPU and GMR development environment
can be installed with:

```bash
uv sync --extra dev --extra cuda --extra gmr
```

CUDA training requires Linux and an NVIDIA GPU. Evaluation with CPU MuJoCo also
works on macOS. The repository-wide setup and gated demo-cache instructions are
in [`README.md`](../README.md).

Set an explicit cache directory when the default user cache is unsuitable:

```bash
uv run musclemimic-set-all-caches --path /path/to/converted_datasets
```

Download an existing GMR cache by dataset group:

```bash
uv run musclemimic-download-gmr-caches \
  --dataset-group KIT_KINESIS_TRAINING_MOTIONS
```

Configs with `write_materialized_cache: true` can construct a materialized cache
during startup. Timed comparisons must materialize every selected train and
validation set before the measured run. This keeps cache construction outside
the training wall-clock comparison.

## Entry points and output locations

`fullbody/experiment.py` composes a Hydra config and calls the shared experiment
runner. Select a root preset with `--config-name` and select configuration-group
axes with `group=name`.

`fullbody/eval.py` loads a local or Hugging Face checkpoint. It supports MJX and
MuJoCo playback, validation metrics, video recording, and trajectory export.

Hydra stores per-launch logs and the resolved config under
`outputs/YYYY-MM-DD/HH-MM-SS/`. With the default `experiment.auto_resume=true`,
the runner stores checkpoints under
`checkpoints/<experiment.run_id-or-config-hash>/`. A config can set
`experiment.checkpoint_root` to choose a stable family-specific root. The
checkpoint directory contains `manifest.json` and `checkpoint_<step>` entries.

Inspect the fully composed config before a training allocation:

```bash
uv run python fullbody/experiment.py \
  --config-name=conf_fullbody_gmr_resnet \
  --cfg job --resolve
```

This command resolves inherited values and interpolation without starting
training.

## Common launch patterns

Run the default full-body PPO config on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python fullbody/experiment.py
```

Run the GMR residual baseline and disable W&B logging:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python fullbody/experiment.py \
  --config-name=conf_fullbody_gmr_resnet \
  wandb.mode=disabled
```

Run a preset on four visible GPUs. `experiment.env_params.num_envs` is the global
environment count and must divide evenly by the device count.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python fullbody/experiment.py \
  --config-name=conf_fullbody_gmr_resnet \
  experiment.distributed.num_devices=4 \
  experiment.trajectory.sharding=static
```

Use an explicit `experiment.run_id` for scheduler retries that must share an
auto-resume checkpoint directory. A changed training config changes the default
config hash and therefore selects a new directory.

```bash
uv run python fullbody/experiment.py \
  --config-name=conf_fullbody_gmr_resnet \
  experiment.run_id=my-reproducible-run
```

Set `experiment.auto_resume=false` for an isolated launch. Set
`experiment.resume_from=/path/to/checkpoint_<step>` or an `hf://` checkpoint for
an explicit initialization source. Resume changes require checkpoint-compatible
network and optimizer settings.

## Configuration structure

The root preset names are existing CLI interfaces. Scripts, tests, checkpoint
manifests, and external experiment records can refer to these names. Keep them
at the root until a compatibility and migration plan covers those consumers.

The main inheritance chain is:

```text
conf_fullbody
  -> conf_fullbody_gmr
    -> conf_fullbody_gmr_resnet
      -> conf_fullbody_gmr_resnet_large
      -> conf_fullbody_gmr_resnet_4gpu
    -> conf_fullbody_flashsac
```

`observation/` and `flashsac/termination/` are Hydra configuration groups. Each
file in those directories represents one independent option, which keeps shared
behavior in parent configs.

## Base and reusable presets

- `conf_fullbody` is the default PPO configuration and shared schema. It uses KIT
  Kinesis train and test groups with egocentric `GoalFullBodyTracking`
  observations.
- `conf_fullbody_gmr` selects GMR retargeted train and validation data.
- `conf_fullbody_gmr_resnet` adds the gated residual policy and critic used by
  current PPO baselines.
- `conf_fullbody_gmr_resnet_large` provides the checkpoint-compatible large
  residual network.
- `conf_fullbody_demo` uses the small downloadable demo cache and 2,048
  environments.
- `conf_fullbody_moe` enables the soft mixture-of-experts network.

The base presets are useful starting points. Record every command-line override
and save the resolved config for a reproducible comparison.

## Multi-GPU presets

- `conf_fullbody_gmr_resnet_4gpu` is the shared four-GPU PPO parent. It sets four
  devices, static trajectory sharding, and the common GPU memory allocation.
- `conf_fullbody_gmr_resnet_kit_kinesis_4gpu` is the four-GPU KIT Kinesis PPO
  preset with static trajectory sharding.

The standard observation lives in `conf_fullbody`. Add
`experiment.env_params.goal_params.include_support_intent=true` for a tracking
run with support intent.

## FlashSAC experiments

The FlashSAC inheritance chain contains these presets:

- `conf_fullbody_flashsac` is the reusable single-device FlashSAC parent.
- `conf_fullbody_flashsac_4gpu` enables four-device data parallelism.
- `conf_fullbody_flashsac_large` increases the single-device environment and
  replay workload.

## Checkpoint evaluation

List trajectories stored by a checkpoint:

```bash
uv run python fullbody/eval.py \
  --path checkpoints/<run-id>/checkpoint_<step> \
  --list_trajs
```

Run exact MJX metrics over a complete motion group:

```bash
uv run python fullbody/eval.py \
  --path checkpoints/<run-id>/checkpoint_<step> \
  --evaluate_all \
  --motion_group KIT_KINESIS_TESTING_MOTIONS \
  --eval_seed 0
```

`--evaluate_all` starts each trajectory at frame zero, reports metrics, and
exits after one complete sweep. It ignores `--n_steps` and `--metrics_steps`.
Early termination ends the current episode and advances to the next trajectory.
Actions are deterministic by default; add `--stochastic` for sampled actions.
Add `--use_mujoco` for CPU evaluation. `bimanual.eval` uses the same options.

Run CPU MuJoCo playback in the viewer:

```bash
uv run python fullbody/eval.py \
  --path checkpoints/<run-id>/checkpoint_<step> \
  --use_mujoco --mujoco_viewer
```

macOS viewer launches use `uv run mjpython fullbody/eval.py`. The command-line
help and module docstring list recording, trajectory export, stochastic action,
termination, and viewer options.

## Reproduction record

Store these facts for every result intended for comparison or publication:

- repository commit and dirty-worktree status;
- resolved Hydra config from `--cfg job --resolve`;
- data source, split manifest, retargeting namespace, and cache provenance;
- random seed flow, total environment-step budget, and resume lineage;
- GPU model and count, CUDA and driver versions, and dependency lockfile;
- W&B entity, project, group, run ID, and run name;
- checkpoint manifest and final checkpoint step;
- validation motion group, action mode, evaluation seed, termination settings,
  and analysis command.

Save generated configs, logs, checkpoints, videos, and analysis outputs in their
ignored output locations. Commit protocol configs, split manifests, launchers,
analysis code, tests, and concise documentation.

## Directory maintenance

Generated paths such as `__pycache__/`, `outputs/`, `checkpoints/`, and `wandb/`
are ignored and can be regenerated or downloaded. Remove local copies when their results have been archived.

Root presets represent existing CLI and experiment-record interfaces. Removal
requires all of these checks:

- search source, tests, scripts, docs, examples, W&B records, and checkpoint
  manifests for the exact config name;
- identify the replacement config and compare fully resolved values;
- preserve a migration note for commands and checkpoint lineage;
- remove or update the corresponding tests, launchers, and documentation in the
  same change.

New matrix experiments should use one root protocol config and one configuration
group per independent axis. Add a dedicated document when the experiment needs
a fixed endpoint or analysis rule. Put scheduler submission in `scripts/` and
composition tests in `tests/unit/`.
