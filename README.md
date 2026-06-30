<p align="center">
  <img src="./assets/banner.jpg" alt="Banner" width="100%">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-76B900.svg" alt="License"></a>
  <a href="https://cnai.epfl.ch/mm-blog/"><img src="https://img.shields.io/badge/MuscleMimic-Blog-blue" alt="MuscleMimic Blog"></a>
  <a href="https://arxiv.org/abs/2603.25544"><img src="https://img.shields.io/badge/Preprint-arXiv-b31b1b" alt="Preprint"></a>
  <a href="https://huggingface.co/spaces/amathislab/musclemimic_space">
    <img src="https://img.shields.io/badge/Hugging%20Face-Space-2D2D2D?style=flat-square&logo=huggingface&logoColor=FFD21E&labelColor=2D2D2D" alt="Hugging Face Space">
  </a>
</p>

# MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale

**MuscleMimic** is a JAX-based motion imitation learning benchmark for
biomechanically accurate, muscle-actuated models. It targets full-body
locomotion and manipulation with GPU-parallel training, MuJoCo/MJWarp dynamics,
and retargeted motion datasets.

<div align="center">
  <img src="assets/teaser.gif" width="1280" alt="MuscleMimic teaser">
</div>

## News

- **2026-06**: Added C3D browser tooling for fitting motion-capture markers,
  exporting AMASS/SMPL-H-compatible motion, and retargeting to MuscleMimic
  trajectories. See the [web viewer guide](musclemimic/web_viewer/README.md).
- **2026-03**: MuscleMimic preprint released on
  [arXiv](https://arxiv.org/abs/2603.25544).
- **2026-02**: MuscleMimic blog post released at
  [https://cnai.epfl.ch/mm-blog/](https://cnai.epfl.ch/mm-blog/).

## Highlights

- **Muscle-actuated dynamics**: Hill-type muscle models with physiological
  activation dynamics.
- **Accelerated training**: JAX JIT compilation with the MuJoCo Warp backend,
  supporting thousands of parallel environments.
- **Generalist imitation policies**: DeepMimic-style rewards and validation
  metrics for diverse motion datasets.
- **Retargeting tools**: GMR-Fit support for AMASS/SMPL motions and C3D marker
  data.

## Available Models

| Model | Type | Joints | Muscles | DoFs | Focus |
| --- | --- | ---: | ---: | ---: | --- |
| MyoBimanualArm | Fixed-base | 76 (36*) | 126 (64*) | 54 (14*) | Upper-body manipulation |
| MyoFullBody | Free-root | 123 (83*) | 416 (354*) | 72 (32*) | Locomotion and manipulation |

<sub>*Configurations with finger muscles disabled.</sub>

## Quick Start

Training requires Linux with an NVIDIA GPU. Inference and evaluation are
supported on Linux and macOS.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/amathislab/musclemimic
cd musclemimic
uv sync --extra cuda
```


### Run a Demo

The first-time path uses pre-retargeted demo motions, so no AMASS download is
needed. The demo dataset is gated on Hugging Face:
[amathislab/demo_dataset](https://huggingface.co/datasets/amathislab/demo_dataset).

```bash
uv run hf auth login
uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_bimanual; setup_demo_for_bimanual()"
uv run python -c "from musclemimic.utils.demo_cache import setup_demo_for_myo_fullbody; setup_demo_for_myo_fullbody()"
```

Start a short demo training run. These configs log to Weights & Biases with
`wandb.mode=online` by default:

```bash
uv run bimanual/experiment.py --config-name=conf_bimanual_demo
uv run fullbody/experiment.py --config-name=conf_fullbody_demo
```

Evaluate a released MyoFullBody checkpoint with the MuJoCo viewer:

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

On Linux, a regular `python` entrypoint is sufficient for viewer-based MuJoCo
commands. On macOS, use `mjpython`.

## Data and Checkpoints

| Model | Checkpoints | Retargeted dataset |
| --- | --- | --- |
| MyoBimanualArm | [amathislab/mm-bimanual-v0](https://huggingface.co/amathislab/mm-bimanual-v0) | [amathislab/musclemimic-bimanual-retargeted](https://huggingface.co/datasets/amathislab/musclemimic-bimanual-retargeted) |
| MyoFullBody | [amathislab/mm-fullbody-base](https://huggingface.co/amathislab/mm-fullbody-base) | [amathislab/musclemimic-retargeted](https://huggingface.co/datasets/amathislab/musclemimic-retargeted) |

Pre-retargeted GMR caches can be downloaded directly:

```bash
uv run musclemimic-download-gmr-caches --dataset-group KIT_KINESIS_TRAINING_MOTIONS
uv run musclemimic-download-gmr-caches --dataset-group AMASS_BIMANUAL_TRAIN_MOTIONS --env-name MyoBimanualArm
```

For full AMASS setup, C3D conversion, finetuning from checkpoints, and viewer
workflows, see [Detailed Workflows](docs/workflows.md).

<div align="center">
  <img src="assets/retargeting.gif" width="1280" alt="Retargeting example">
</div>

## Guides

- [Detailed workflows](docs/workflows.md): demo cache, evaluation variants, GMR
  caches, full AMASS retargeting, finetuning, and Viser visualization.
- [C3D web viewer](musclemimic/web_viewer/README.md): C3D marker viewing,
  SMPL-X/SMPL-H fitting, cache layout, and browser retargeting.
- [Contributing](CONTRIBUTING.md): local development, testing, and review
  guidelines.

## Development

```bash
make install-dev
make precommit-install
make ci
```

`pre-commit` currently targets a curated subset of files while the repository is
being migrated toward broader coverage. `make lint` and `make format` follow
that same scoped set rather than reformatting the whole repository.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Li2026MuscleMimic,
  title={Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale},
  author={Li, Chengkun and Wang, Cheryl and Ziliotto, Bianca and Simos, Merkourios and Kovecses, Jozsef and Durandau, Guillaume and Mathis, Alexander},
  journal={arXiv preprint arXiv:2603.25544},
  year={2026}
}
```

## License

This project is licensed under the [Apache License](LICENSE). Model checkpoints,
datasets, SMPL-family assets, and other third-party software may be licensed
separately; review each provider's terms before use.

## Acknowledgments

Inspired by and built on
[MyoSuite](https://github.com/MyoHub/myosuite),
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
[Kinesis](https://github.com/amathislab/Kinesis),
[LocoMuJoCo](https://github.com/robfiras/loco-mujoco),
[SMPL-X](https://github.com/vchoutas/smplx),
[PureJaxRL](https://github.com/luchris429/purejaxrl), and
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).
