# FlashSAC episode boundaries

FlashSAC stores the observation immediately after the action and before autoreset
as the replay transition's next observation. The observation after autoreset is
used only for collecting the next transition.

The Bellman target stops bootstrapping on absorbing task failures, finite
trajectory ends, and numerical failures. The finite trajectory handler clamps
the reference at its final frame, so it does not define a continuation beyond
that frame. Administrative episode horizons retain bootstrap from the pre-reset
observation. A future cyclic trajectory implementation must define its actual
continuation before changing this terminal convention.

Both CPU MuJoCo and MJX detect nonfinite actions, observations, and rewards.
They record `done_numerical`, mark the transition absorbing, and sanitize the
returned observation and reward. Actions are checked before control sanitization,
so a numerical policy output still triggers autoreset. FlashSAC also sanitizes
replay values to prevent masked target computations from propagating NaN or Inf.
The environment's actual `done` controls episodic return reset and the clearing
of temporally repeated exploration noise.

The collection log separates tracking failures, numerical failures, trajectory
ends, horizons, and site-position failures. These reason counts can overlap when
multiple conditions hold on the same transition. `episodes/early_termination_rate`
includes absorbing failures; `episodes/tracking_termination_rate` excludes
numerical failures.

Training logs aggregate every interaction since the previous emitted record,
including interactions between logging intervals and across compiled chunks.
Episode and termination-reason counts are sums; episode return and length are
weighted by the number of completed episodes. Mean step rewards and their
components average over interactions. Actor and critic diagnostics average
over interactions whose corresponding update-validity flag is true.
Temperature, reward EMA, replay size, and learning rate describe the latest
state at the logging boundary. Validation events and the final interaction flush
the current window. `flashsac/log_window_interactions` records its width.

Windows with zero completed episodes emit counts but omit episode means and
termination rates. These quantities have no samples in that window. Earlier
logs sampled only the interaction at each logging boundary; their counts and
rates do not represent complete windows. Historical records cannot recover
unlogged episode outcomes. Each process starts an empty logging window; the
host accumulator is not checkpointed. This changes emitted diagnostics.
Checkpoint payloads retain their existing schema.

`MeanSiteDeviationTerminalStateHandler` accepts `reduction: mean` or `max`,
`exclude_sites`, and `minimum_episode_steps`. It compares absolute world-space
site positions after aligning the reference to the episode's root XY origin.
The threshold remains in `mean_site_deviation_threshold` for configuration
compatibility. Episode step numbering begins at 1; setting
`minimum_episode_steps: 2` enables the gate on the second transition.

Checkpoint algorithm version `FlashSACJax_v5` identifies the new replay terminal
semantics. Resuming a prior-version checkpoint containing replay fails explicitly
because those records lack the information needed to reconstruct terminal labels.
Start a fresh run or deliberately use a learner-only checkpoint to initialize
empty replay. Earlier experimental branches already used V3 and V4 for runtime
resume and logging changes with the old terminal labels. V2 through V4
learner-only checkpoints preserve reward statistics; V1 learner-only checkpoints
retain the existing reward-statistics migration.
