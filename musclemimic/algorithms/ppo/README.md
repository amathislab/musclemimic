# PPO blocked training

The PPO host runner always uses chunked training. One compiled `jax.lax.scan`
executes several PPO updates per host dispatch. Each PPO update collects
`num_steps` steps from every environment, computes advantages over the complete
rollout batch, and runs `update_epochs * num_minibatches` optimizer steps.

```text
transitions_per_update = num_steps * global_num_envs
```

The configured `num_envs` gives the global total across all devices. The
training builder assigns an equal share to each device. Chunk size counts
globally synchronized PPO updates.

PPO derives its static scan window from the validation schedule. It currently
exposes chunk size through `validation.num`.

```text
num_updates = total_timesteps // num_steps // global_num_envs
validation_interval = max(1, num_updates // validation.num)
max_chunk_size = validation_interval
```

Each host dispatch stops at the nearest validation, positive periodic
checkpoint, or end-of-training boundary. Here,
`distance_to_next_multiple(position, interval)` returns the positive number of
units required to reach the next multiple of `interval`.

```text
to_validation = distance_to_next_multiple(counter, validation_interval)
to_checkpoint = distance_to_next_multiple(counter, checkpoint_interval)
chunk_size = min(remaining_updates, to_validation, to_checkpoint)
```

The runner includes `to_checkpoint` when `checkpoint_interval` has a positive
value. Resume reconstructs `counter` from the optimizer step, which preserves
alignment with the validation and checkpoint schedules.

The compiled scan length remains `max_chunk_size`. A shorter dispatch executes
the active prefix. Padded positions preserve state and emit zero metrics. This
fixed shape prevents recompilation when a lifecycle boundary shortens a chunk.

Validation, positive periodic checkpoints, and training completion create hard
boundaries. `online_logging_interval` selects entries from the returned metric
history after dispatch, so logging preserves chunk size. Four-device MJX-Warp
training waits for live outputs and clears the Warp graph cache after each
chunk.

[`PPOTraining.update_chunk`](train.py) implements the compiled scan.
[`run_training`](../../runner/training_runtime.py) computes dispatch sizes and
handles lifecycle boundaries.

The validated four-device configuration lives in
[`fullbody/conf_fullbody_gmr_resnet_kit_kinesis_4gpu.yaml`](../../../fullbody/conf_fullbody_gmr_resnet_kit_kinesis_4gpu.yaml).

```text
global_num_envs       = 8192
local_num_envs        = 2048 per GPU
num_steps             = 20
update_epochs         = 1
num_minibatches       = 128
local_minibatch_size  = 320 samples per GPU
num_updates           = 75,000
validation.num        = 1000
validation_interval  = 75
max_chunk_size        = 75 PPO updates
transitions_per_update = 163,840
transitions_per_chunk  = 12,288,000
optimizer_steps_per_chunk = 9,600
```

This configuration saves checkpoints on validation boundaries. Every regular
chunk contains 75 PPO updates, and 75 divides the total update count exactly.
Increasing `validation.num` shortens chunks and raises validation frequency.
Smaller chunks increase host dispatches and Warp graph cache clears. Larger
chunks reduce that overhead and enlarge the static scan and returned metric
history.
