# Submitting Jobs with Run:AI
An example of submitting with Run:AI

```bash
runai submit \
  --name mm-$(date +%m%d-%H%M%S) \
  --image <your-musclemimic-image> \
  --run-as-uid <your_epfl_uid> \
  --run-as-gid <your_epfl_gid> \
  --gpu 1 --node-pools h100 \
  --existing-pvc claimname=<your_claim_name>,path=/users \
  --environment UV_PROJECT_ENVIRONMENT=/tmp/venv \
  --environment UV_CACHE_DIR=/users/<your_username>/.cache/uv \
  --environment UV_PYTHON_INSTALL_DIR=/users/<your_username>/.uv-pythons \
  --environment HOME=/users/<your_username> \
  --environment WANDB_MODE=disabled \
  --backoff-limit 0 \
  --command -- /bin/bash -c "cd /users/<your_username>/musclemimic; uv sync --extra smpl --extra gmr --extra cuda; uv run fullbody/experiment.py --config-name=<some_config>"
```
