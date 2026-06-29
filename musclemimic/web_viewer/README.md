# Web Viewer

Browser-based tools for:

- viewing raw C3D markers
- fitting C3D markers to SMPL-X/SMPL-H surfaces
- exporting fitted motion to AMASS/SMPL-H format for GMR
- retargeting the fitted motion to MuscleMimic models
- browsing existing retargeted trajectories


## Install

Raw C3D viewer only:

```bash
uv sync --extra c3d
```

C3D fitting and default GMR retargeting:

```bash
uv sync --extra c3d --extra smpl --extra gmr
```

CUDA-accelerated default JAX Stage I:

```bash
uv sync --extra c3d --extra smpl --extra gmr --extra cuda
```

## Assets

SMPL/SMPL-X model files are licensed separately and are not downloaded by this
repo. Download them from the official SMPL-X/SMPL websites after accepting the
corresponding license terms:

- SMPL-X: https://smpl-x.is.tue.mpg.de/
- SMPL/SMPL-H family loaders: see the SMPL-X project instructions at
  https://github.com/vchoutas/smplx

Required for C3D fitting:

- A SMPL-X or SMPL-H body model path passed with `--c3d-model-path`, or configured with:

```bash
musclemimic-set-c3d-model-path --path /path/to/c3d-fitting/smplx-or-smplh
```

- The current validated default is `--c3d-surface-model smplx --c3d-gender male`.
- For SMPL-X, the path can be a directory containing `male/model.pkl`,
  `SMPLX_MALE.pkl`, or `SMPLX_MALE.npz`.
- If using a MoSh++/SOMA-style SMPL-X `model.pkl`, provide
  `pose_hand_prior.npz` next to the model, one directory above it, or through
  the configured MoSh++ assets path.

Recommended optional assets for MoSh++-parity fitting:

- `pose_body_prior.pkl` for the GMM body-pose prior.
- `ssm_head_marker_corr.npz` for Stage-I head marker correlation.

Configure a directory containing those files with:

```bash
musclemimic-set-moshpp-assets-path --path /path/to/moshpp/assets
```

The directory key is `MUSCLEMIMIC_MOSHPP_ASSETS_PATH`. Split layouts can instead
set `MUSCLEMIMIC_MOSHPP_POSE_HAND_PRIOR_PATH`,
`MUSCLEMIMIC_MOSHPP_POSE_BODY_PRIOR_PATH`, and
`MUSCLEMIMIC_MOSHPP_HEAD_MARKER_CORR_PATH`.

If you use official SMPL-X `.npz` files instead of the MoSh++/SOMA-style
`model.pkl`, put them under a model directory such as:

```text
/path/to/smplx/
  SMPLX_MALE.npz
  SMPLX_FEMALE.npz
  SMPLX_NEUTRAL.npz
```

and pass `--c3d-model-path /path/to/smplx`. In that case the extra
`pose_hand_prior.npz` file is not needed by our loader.

Required only for direct `--retargeting-method smpl`:

- A retargeting-compatible SMPL/SMPL-H model path passed with
  `--retarget-smpl-model-path`, or configured with:

```bash
musclemimic-set-smpl-model-path --path /path/to/retarget/smplh/models
```

The browser pipeline keeps fitting and retargeting model paths separate:
`--c3d-model-path` / `MUSCLEMIMIC_C3D_MODEL_PATH` is used only for C3D marker
fitting, while `--retarget-smpl-model-path` / `MUSCLEMIMIC_SMPL_MODEL_PATH` is
used only by the direct SMPL retargeting backend. GMR retargeting consumes the
exported AMASS/SMPL-H-compatible motion file and does not need a separate
retarget model path.

## Common Commands

View raw C3D markers:

```bash
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer path/to/file.c3d --trail 10
```

Fit C3D, retarget it with default GMR, and view it in the browser:

```bash
uv run --extra c3d --extra smpl --extra gmr python -m musclemimic.web_viewer.run \
  --c3d-file path/to/file.c3d \
  --c3d-model-path /path/to/smplx/models \
  --c3d-dataset-name MyStudy/Subject01/Trial01
```

Directly visualize or record the retargeted C3D motion in MuJoCo:

```bash
uv run --extra c3d --extra smpl --extra gmr examples/retargeting/retarget_visualize.py \
  --c3d-file /path/to/file.c3d \
  --c3d-model-path /path/to/smplx_model \
  --retargeting-method gmr \
  --record
```

The default Stage-I solver is `joint_dogleg_jax`. Add the CUDA extra when you
want JAX to run on GPU:

```bash
uv run --extra c3d --extra smpl --extra gmr --extra cuda python -m musclemimic.web_viewer.run \
  --c3d-file path/to/file.c3d \
  --c3d-model-path /path/to/smplx/models
```

Example with an explicit researcher-facing dataset name:

```bash
uv run --extra c3d --extra smpl --extra gmr --extra cuda python -m musclemimic.web_viewer.run \
  --c3d-file path/to/walk_trial_01.c3d \
  --c3d-model-path /path/to/c3d-fitting/smplx \
  --c3d-dataset-name ExampleStudy/Subject01/Walking/Trial01/WalkTrial01
```

Use the direct SMPL retargeting backend only when both model paths are explicit:

```bash
uv run --extra c3d --extra smpl python -m musclemimic.web_viewer.run \
  --c3d-file path/to/file.c3d \
  --retargeting-method smpl \
  --c3d-model-path /path/to/smplx/models \
  --retarget-smpl-model-path /path/to/retarget/smplh/models
```

View existing retargeted motions:

```bash
uv run python -m musclemimic.web_viewer.run \
  --motion "KIT/6/WalkInCounterClockwiseCircle06_1_poses"
```

Or a dataset group:

```bash
uv run python -m musclemimic.web_viewer.run \
  --dataset-group AMASS_LOCOMOTION_DATASETS
```

## C3D -> SMPL Design

The fitting entry point lives in [c3d_to_smpl.py](./c3d_to_smpl.py).
Implementation helpers are split by responsibility:

- `c3d/markers.py`: C3D loading, label canonicalization, marker layouts, frame picking
- `c3d/smpl_models.py`: SMPL-H/SMPL-X model loading and the MoSh++ `model.pkl` adapter
- `c3d/surface_markers.py`: surface marker reconstruction and mesh-distance geometry
- `c3d/pose_prior.py`: MoSh++ GMM body-pose prior
- `c3d/optim.py`: dense Powell dogleg optimizer shared by torch/JAX Stage I
- `stagei_jax_solver.py`: CUDA-capable JAX Stage-I solver

It is MoSh++-inspired. Stage II is implemented in torch; Stage I has both the
torch dogleg solver and a JAX dogleg solver with the same residual blocks.
The current validated default surface is SMPL-X male, while the exported file
for GMR is still AMASS/SMPL-H-compatible.

Core ideas:

1. Canonicalize incoming marker labels to MoSh++ body-marker names.
2. Attach each marker to a fixed SMPL-X or SMPL-H surface location using the same marker vertex IDs as MoSh++.
3. Initialize a canonical-space `markers_latent` position for each marker from the mesh surface normal.
4. Stage I optimizes shared `betas`, per-reference-frame rigid alignment, and `markers_latent`.
5. Stage II keeps shape and latent markers fixed, then fits pose and translation frame by frame.

A few practical details:

- body markers use a `9.5 mm` surface offset
- wrist markers on sticks use a `39 mm` offset
- C3D fitting runs at the source mocap rate unless an explicit diagnostic `target_fps` is passed
- missing markers are handled with NaNs / zero detection and per-frame weighting
- intermediate SMPL fits and GMR input files are cached in `.smpl_cache/`
- final training-ready C3D trajectories are saved under the converted C3D cache root

## Converted C3D Cache

The final output of the C3D pipeline is an extended MuscleMimic `Trajectory`
`.npz`, not an AMASS file. It is saved under:

```text
${MUSCLEMIMIC_CONVERTED_C3D_PATH:-~/.musclemimic/caches/C3D}/<model>/<method>/<dataset-name>.npz
```

Set the cache location with:

```bash
musclemimic-set-conv-c3d-path --path /path/to/converted/C3D
```

Use `--c3d-dataset-name` to choose the researcher-facing relative name under
that cache root. The name is sanitized and must stay relative. If omitted, the
pipeline uses the C3D path relative to the current working directory when
possible, plus a short hash of the fitting/retargeting settings to avoid
collisions. Analysis data is saved next to the trajectory as
`<dataset-name>_analysis.npz`.

When the converted trajectory already exists, rerunning the same C3D command
loads this training-ready cache directly before SMPL fitting or retargeting.
Pass `--clear-c3d-cache` to recompute the converted trajectory and the
intermediate SMPL fit.

Training configs can load these trajectories with `c3d_dataset_conf`:

```bash
uv run fullbody/experiment.py --config-name=conf_fullbody_gmr \
  experiment.task_factory.params.amass_dataset_conf=null \
  +experiment.task_factory.params.c3d_dataset_conf.rel_dataset_path='["ExampleStudy/Subject01/Walking/Trial01/WalkTrial01"]'
```

## Pipeline

```text
C3D
 -> fit_smpl_to_c3d_cached()
 -> {pose_aa, trans, betas, fps}
 -> fit_smpl_motion() / fit_gmr_motion()
 -> MuscleMimic trajectory
 -> extend_motion()
 -> converted C3D trajectory cache
 -> web viewer or MuJoCo viewer
```

## Notes

- `python -m musclemimic.web_viewer.run` accepts exactly one of:
  `--motion`, `--dataset-group`, or `--c3d-file`
- `--model` supports `MyoFullBody` and `MyoBimanualArm`
- `--retargeting-method` supports `smpl` and `gmr`
