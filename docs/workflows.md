# Detailed Workflows

This guide keeps the longer setup and experiment commands out of the project
front page while preserving the exact workflows used by MuscleMimic users.

## System Requirements

- **Training**: Linux with an NVIDIA GPU.
- **Inference and evaluation**: Linux and macOS.

## Demo Cache

The demo cache provides pre-retargeted motions for both **MyoBimanualArm** and
**MyoFullBody**. It is hosted as a gated Hugging Face dataset.

1. Request access to
   [amathislab/demo_dataset](https://huggingface.co/datasets/amathislab/demo_dataset).
2. Create an access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Log in from the terminal:

```bash
uv run hf auth login
```

Download the demo motions:

```bash
uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_bimanual; setup_demo_for_bimanual()"
uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_myo_fullbody; setup_demo_for_myo_fullbody()"
```

Run short training jobs. Logging is handled by Weights & Biases:

```bash
# For MyoBimanualArm
uv run bimanual/experiment.py --config-name=conf_bimanual_demo
# For MyoFullBody
uv run fullbody/experiment.py --config-name=conf_fullbody_demo
```

## Evaluate a Checkpoint

Examples below assume you have downloaded the MyoFullBody demo cache:

```bash
uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_myo_fullbody; setup_demo_for_myo_fullbody()"
```

Use `mjpython` on macOS for viewer-based MuJoCo commands. On Linux, a regular
`python` entrypoint is sufficient.

```bash
uv run mjpython fullbody/eval.py \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco \
  --stochastic \
  --eval_seed 0 \
  --n_steps 1000 \
  --mujoco_viewer
```

```bash
uv run mjpython fullbody/eval.py \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/348/turn_right03_poses \
  --use_mujoco \
  --stochastic \
  --eval_seed 0 \
  --n_steps 1000 \
  --mujoco_viewer
```

```bash
uv run mjpython fullbody/eval.py \
  --path hf://amathislab/mm-10m-2 \
  --motion_path KIT/4/WalkInCounterClockwiseCircle04_poses \
  --use_mujoco \
  --stochastic \
  --eval_seed 0 \
  --n_steps 1000 \
  --mujoco_viewer
```

## Retargeting with GMR-Fit

MuscleMimic retargets to MyoFullBody and MyoBimanualArm with
[General Motion Retargeting (GMR) Fit](https://github.com/amathislab/gmr_plus),
which incorporates SMPL fitting rather than relying on manually defined AMASS
joint configurations.

### Hugging Face Resources

- MyoBimanualArm
  - Checkpoints: [amathislab/mm-bimanual-v0](https://huggingface.co/amathislab/mm-bimanual-v0)
  - Dataset: [amathislab/musclemimic-bimanual-retargeted](https://huggingface.co/datasets/amathislab/musclemimic-bimanual-retargeted)
- MyoFullBody
  - Checkpoints: [amathislab/mm-fullbody-base](https://huggingface.co/amathislab/mm-fullbody-base)
  - Dataset: [amathislab/musclemimic-retargeted](https://huggingface.co/datasets/amathislab/musclemimic-retargeted)

Set a local cache root when you want to control where converted datasets live:

```bash
uv run musclemimic-set-all-caches --path /path/to/converted_datasets
```

Access pre-retargeted GMR caches in any of these ways:

1. Set `retargeting_method: gmr` in your config and let MuscleMimic download
   the required caches automatically.
2. Download manually from the CLI:

```bash
uv run musclemimic-download-gmr-caches --dataset-group KIT_KINESIS_TRAINING_MOTIONS
uv run musclemimic-download-gmr-caches --dataset-group AMASS_BIMANUAL_TRAIN_MOTIONS --env-name MyoBimanualArm
```

3. Download from Python:

```python
from musclemimic.utils import download_gmr_dataset_group

download_gmr_dataset_group("KIT_KINESIS_TRAINING_MOTIONS")
download_gmr_dataset_group(dataset_group="AMASS_BIMANUAL_TRAIN_MOTIONS", env_name="MyoBimanualArm")
```

## Full Retargeting with AMASS

Use this path when you want to retarget your own AMASS dataset in batch.

### Download AMASS

Register and download AMASS from
[amass.is.tue.mpg.de](https://amass.is.tue.mpg.de/). Place all datasets in one
directory:

```text
/path/to/amass/
|-- ACCAD/
|-- KIT/
|   |-- 1/
|   |   |-- LeftTurn03_poses.npz
|   |   `-- ...
|   `-- ...
`-- ...
```

Install the SMPL dependencies:

```bash
uv sync --extra smpl --extra gmr
```

### Download SMPL-H and MANO

Download the required assets from the
[MANO website](https://mano.is.tue.mpg.de/download.php):

- Extended SMPL+H model, which includes the SMPL-H model without hands.
- Models & Code, which includes the hand models.

Extract them into a directory like:

```text
/path/to/smpl/
|-- mano_v1_2/
`-- smplh/
```

### Set Paths

```bash
uv run musclemimic-set-amass-path --path /path/to/amass
uv run musclemimic-set-smpl-model-path --path /path/to/smpl
uv run musclemimic-set-all-caches --path /path/to/converted_datasets
```

These commands write user-specific settings to
`~/.musclemimic/MUSCLEMIMIC_VARIABLES.yaml` by default. Set
`MUSCLEMIMIC_CONFIG_PATH` to use a different config file.

### Convert SMPL-H and MANO

Generate the `SMPLH_neutral.pkl` file needed for retargeting:

```bash
cd loco_mujoco/smpl
bash install_smplh.sh
```

### Run Retargeting

```bash
uv run scripts/retarget_dataset.py --model MyoFullBody --retargeting-method gmr --dataset KIT_KINESIS_TRAINING_MOTIONS --workers 8
uv run scripts/retarget_dataset.py --model MyoBimanualArm --retargeting-method gmr --dataset AMASS_BIMANUAL_MARGINAL_MOTIONS --workers 8
```

## C3D Workflows

The browser-based C3D pipeline can view raw markers, fit C3D markers to
SMPL-X/SMPL-H, export fitted motion, retarget to MuscleMimic models, and browse
retargeted trajectories.

See [the C3D web viewer guide](../musclemimic/web_viewer/README.md).

## Training and Finetuning from a Checkpoint

For targeted finetuning, reset the policy standard deviation to `3` to
encourage exploration on the new motion:

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_gmr_resnet \
  experiment.resume_from="hf://amathislab/mm-fullbody-base" \
  experiment.reset_std_on_resume=3 \
  experiment.task_factory.params.amass_dataset_conf.dataset_group=null \
  experiment.task_factory.params.amass_dataset_conf.rel_dataset_path='["KIT/200/Handstand01_poses"]'
```

To continue training on a broader motion distribution, resume from the same
checkpoint and switch to the transition-augmented training set:

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_gmr_resnet \
  experiment.resume_from="hf://amathislab/mm-fullbody-base" \
  experiment.task_factory.params.amass_dataset_conf.dataset_group="KIT_KINESIS_TRANSITION_TRAINING_MOTIONS"
```

## Visualization with Viser

[Viser](https://github.com/nerfstudio-project/viser) provides real-time policy
visualization with muscle tendons.

```bash
uv run bimanual/eval.py \
  --path outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/XXXXXX/checkpoint_XXX \
  --use_mujoco --viser_viewer
```

```bash
uv run fullbody/eval.py \
  --path outputs/2025-10-12/09-32-55/checkpoints/2510120733/checkpoint_400 \
  --use_mujoco --viser_viewer
```

Viser requires `--use_mujoco` and does not run with MJX.
