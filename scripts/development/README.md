# Development Scripts

These scripts are for local validation and debugging during development. They are
not part of the production C3D viewer or retargeting pipeline.

## Stage-I MoSh++ Parity

Compare the current C3D Stage-I fit against a local MoSh++ `*_stagei.pkl`:

```bash
uv run --extra c3d --extra smpl scripts/development/validate_stagei_against_moshpp.py \
  --stagei-pkl /path/to/male_stagei.pkl
```

Set `MUSCLEMIMIC_MOSHPP_ASSETS_PATH` with
`musclemimic-set-moshpp-assets-path --path /path/to/moshpp/assets` when the
MoSh++ auxiliary files are not colocated with the C3D fitting model.
Set `MUSCLEMIMIC_C3D_MODEL_PATH` with
`musclemimic-set-c3d-model-path --path /path/to/c3d-fitting/models` when you do
not pass `--c3d-model-path` directly.
