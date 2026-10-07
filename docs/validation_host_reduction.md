# Multi-device validation reduction

PPO and FlashSAC run validation independently on each device. The compiled
rollout returns shard-local metrics with a leading device axis. The host waits
for all local results, copies their addressable shards, and reduces the metrics
with NumPy before logging or starting the next training chunk.

This boundary prevents validation runtime differences from starting an XLA
collective while another device is still simulating. A four-GPU FlashSAC run
aborted at its first exact validation because only two of four devices
reached the collective within its 40-second rendezvous timeout. The host
completion boundary establishes execution ordering before reduction starts.

Exact validation returns additive counts, return sums, length sums, covered
frames, and valid-step metric sums. The host computes episode-weighted returns
and lengths, frame-weighted coverage, and valid-step-weighted diagnostics.
Sampled validation preserves the existing count, maximum, RMSE, and weighted
episode aggregation semantics. Padding and post-terminal diagnostics do not
contribute to exact-validation step metrics.

Validation disables the reward-normalization wrapper because that wrapper
synchronizes running statistics inside the rollout. Validation reports raw
environment rewards. Training reward normalization and gradient synchronization
remain unchanged. Validation keeps its independent RNG stream, including traced
PPO skip branches. Checkpoint formats and validation logging field names remain
unchanged.

The common `evaluate_exact_trajectory_set` function retains its globally reduced
summary API. Scheduled PPO and FlashSAC validation use
`evaluate_exact_trajectory_set_local` and return sufficient statistics instead.
Direct callers of these scheduled algorithm validation functions must finalize
their shard-local results through `runner.validation_reduction.finish_validation`
outside JAX transformations.

Standalone `--metrics` evaluation also honors a checkpoint's
`validation.evaluate_all=true` for PPO and FlashSAC. It traverses the complete
validation library in fixed-size `--metrics_envs` batches, supplies trajectory
ids, lengths, and a padding mask, and reduces the completed additive statistics
on host before flattening metrics. The last batch excludes padding. Returns and
lengths use episode counts, coverage uses frame counts, and step diagnostics use
valid-step counts. `--metrics_steps` retains the configured horizon semantics;
unfinished trajectories contribute horizon timeouts. Each batch advances the
evaluation RNG. Reproducibility requires the same seed and batch size. This
standalone path uses one JAX process and leaves sampled evaluation unchanged.

PPO and FlashSAC retain their existing buffer donation. The historical
[MuJoCo issue #2980](https://github.com/google-deepmind/mujoco/issues/2980)
reported a hang with older `pmap`, MJX 3.4.0, and Warp 1.10.1. The reporter
later closed it after no longer observing the failure. The current runtime
uses `shard_map`, MJX 3.7.0, and Warp 1.12.0; its active Warp FFI graph cache
key includes the device ordinal. These facts do not prove all donation paths
safe, but the observed validation timeout does not establish a donation fault.
No donation configuration or multi-device prohibition is added here.

This change supports the PR's single-process multi-device topology. The optional
host-exchange callback is a validation integration boundary; it does not add
multi-host training support. Multiple processes require an explicit metric
exchange, and missing exchanges raise an error instead of reporting local
metrics as global values.

## Verification

The regression tests lower and execute PPO and FlashSAC exact rollouts on four
simulated CPU devices, use unequal per-device trajectory counts, and assert
that the compiled rollout contains no device collectives. The same collective
assertions fail on PR head `2169301`. Additional tests cover sampled reduction,
host wait/copy/reduce ordering, reward-normalization exclusion, traced PPO skip
behavior, and retained single-device and multi-device donation.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
make test PYTEST_ARGS='tests/unit/test_validation_rollout_collectives.py tests/unit/test_validation_reduction.py tests/unit/test_ppo_evaluation.py tests/unit/test_ppo_train.py tests/unit/test_flashsac_train.py tests/unit/test_run_training_dispatch.py tests/unit/test_flashsac_runtime.py -q'
```

Real GPU smoke testing complements these structural checks. A short horizon
checks training/validation transitions; it does not reproduce the runtime of a
23,000-step full validation or establish learned-policy quality.

When sharing an editable virtual environment across worktrees, `PYTHONPATH=.`
ensures that pytest imports this checkout. CPU-only runs hide CUDA devices so
Warp's hardware detection skips GPU-only tests.

A four-GPU Warp integration test compares donation enabled and disabled using
identical seeds and a minimal hinge model, one environment per GPU. Each path
runs four 64-step chunks, interleaves host-reduced validation, clears graph
caches at completed boundaries, and continues after a host state round trip.
The test verifies executable input/output aliasing and matching final position,
velocity and validation returns at `rtol=atol=1e-6`, and identical RNG. This checks the current
FFI and host lifecycle; it is not a long-duration PPO stress test or an Orbax
checkpoint test.

```bash
PYTHONPATH=. JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
make test PYTEST_ARGS='tests/integration/test_warp_shard_map_donation.py -q -s'
```
