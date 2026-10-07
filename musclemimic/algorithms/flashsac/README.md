# FlashSAC

This package provides a native JAX implementation of
[FlashSAC](https://github.com/Holiday-Robot/FlashSAC) for MuscleMimic. It
implements the tanh-normal actor, persistent truncated-zeta exploration,
categorical double critic, learned entropy temperature, Polyak target, reward
scaling, and parameter norm constraints.

Reward normalization includes each terminal reward in the current episode's
discounted return, then clears the carry before the next auto-reset episode.
FlashSACJax v2 introduced these episodic statistics. V5 adds corrected replay
terminal labels. V2 through V4 learner-only checkpoints preserve reward
statistics, while V1 resets them. V2 through V4 checkpoints containing replay
cannot resume into V5 because the old replay does not contain enough information
to reconstruct the corrected terminal labels.

`networks.py`, `losses.py`, and `update.py` implement the learner.
`replay_buffer.py` and `state.py` define dynamic JAX state. `train.py` contains
pure initialization, interaction, and validation kernels. `algorithm.py`
provides the MuscleMimic algorithm interface. The runner owns logging,
checkpoint scheduling, and W&B calls.

Run either task configuration from the repository root.

```bash
python fullbody/experiment.py -cn conf_fullbody_flashsac
python bimanual/experiment.py -cn conf_bimanual_flashsac
```

The fullbody command selects the production configuration used for the PHC
FlashSAC stability experiments. Its principal defaults are 8,192 environments,
49,152 interactions, a 2,097,152-transition replay buffer, a 1,048,576-transition
warmup, batches of 16,384, two critic updates per interaction, a 768-wide actor,
an actor period of four, and an initial entropy temperature of `1e-4`.
It also uses the balanced PHC tracking reward weights from the stability run.

The default `loose` termination profile uses relative mean site, root position,
and root orientation thresholds of `1.0`. The stricter `phc` profile applies a
maximum absolute site-error threshold of 0.25 m from the second transition
onward and excludes the ankle and toe sites. Profiles remain selectable without
copying the experiment configuration.

```bash
python fullbody/experiment.py -cn conf_fullbody_flashsac flashsac/termination=phc
python fullbody/experiment.py -cn conf_fullbody_flashsac flashsac/termination=no_tracking
```

These defaults change algorithm, collection, termination, checkpoint, and
validation settings. The motion dataset remains inherited from
`conf_fullbody_gmr.yaml`; this configuration does not introduce a new PHC data
split or retargeting pipeline.

FlashSAC supports data-parallel training across multiple devices and one seed
per process. Lightweight resume restores the learner and creates fresh
environment, exploration, and replay state. The fullbody configuration persists
replay by default for faithful preemption recovery. Set
`experiment.checkpoint_include_replay_buffer=false` for smaller learner-only
checkpoints. A learner-only resume collects uniform-random transitions and
pauses learner updates until replay reaches `min_buffer_size`. The bimanual
configuration continues to use learner-only checkpoints by default.

The FlashSAC host runner always uses chunked training. One compiled
`jax.lax.scan` executes several vectorized environment interactions per host
dispatch. Each interaction collects one transition from every environment and
writes the data to each device replay shard. Once replay reaches
`min_buffer_size`, the interaction also runs `updates_per_interaction` learner
updates.

Each interaction averages actor and critic diagnostics over their respective
valid updates. Actor loss, entropy, and Q diagnostics remain available when the
interaction ends with a critic-only update. Temperature reports its final value
for the interaction. Replay warmup emits no learner measurements.

Chunk size counts vectorized interactions. One interaction represents
`global_num_envs` global transitions. `updates_per_interaction` determines the
learner update count inside each interaction.

`flashsac_config.update_chunk_size` limits the static scan window. Its default
value is 10.

```text
num_interactions = total_timesteps // global_num_envs
validation_interval = max(1, num_interactions // validation.num)
max_chunk_size = min(update_chunk_size, validation_interval, checkpoint_interval)
```

The runner includes `validation_interval` when validation is active. It includes
`checkpoint_interval` when checkpointing is active and the interval has a
positive value. Each dispatch also stops at the nearest active lifecycle
boundary. Here, `distance_to_next_multiple(position, interval)` returns the
positive number of units required to reach the next multiple of `interval`.

```text
to_validation = distance_to_next_multiple(interaction, validation_interval)
to_checkpoint = distance_to_next_multiple(interaction, checkpoint_interval)
chunk_size = min(remaining_interactions, max_chunk_size, to_validation, to_checkpoint)
```

The compiled scan length remains `max_chunk_size`. A shorter dispatch executes
the active prefix. Padded positions preserve state and emit zero metrics. This
fixed shape prevents recompilation when a lifecycle boundary shortens a chunk.

Validation, enabled periodic checkpoints, and training completion create hard
boundaries. `online_logging_interval` selects interactions from the returned
metric history after dispatch, so logging preserves chunk size. Four-device
MJX-Warp training waits for live outputs and clears the Warp graph cache after
each chunk.

[`FlashSACTraining.update_chunk`](train.py) implements the compiled scan.
[`run_flashsac_training`](../../runner/flashsac_runtime.py) computes dispatch
sizes and handles lifecycle boundaries.

The four-device configuration lives in
[`fullbody/conf_fullbody_flashsac_4gpu.yaml`](../../../fullbody/conf_fullbody_flashsac_4gpu.yaml).

```text
global_num_envs           = 2048
local_num_envs            = 512 per GPU
global_buffer_capacity    = 4,194,304
local_buffer_capacity     = 1,048,576 per GPU
global_min_buffer_size    = 524,288
local_min_buffer_size     = 131,072 per GPU
global_sample_batch_size  = 8192
local_sample_batch_size   = 2048 per GPU
updates_per_interaction   = 2
update_chunk_size         = 1
total_timesteps           = 1,024,000,000 global transitions
num_interactions          = 500,000
validation_interval       = 10,000 interactions
checkpoint_interval      = 2048 interactions
max_chunk_size            = 1 interaction
transitions_per_chunk     = 2048
```

Every chunk in this configuration contains one vectorized interaction. An empty
replay reaches `min_buffer_size` after 256 interactions. FlashSAC collects data
during this interval. Each later chunk normally contains two learner updates.

Increasing `update_chunk_size` reduces host dispatches, lengthens the compiled
scan and metric history, and reduces Warp graph cache clear frequency. Changes
to this value require renewed memory and four-device stability checks. The
fullbody single-device configuration uses a chunk size of one. The bimanual
single-device configuration retains the generic default of 10, with shorter
chunks at validation and checkpoint boundaries.
