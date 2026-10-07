from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
from omegaconf import OmegaConf

from musclemimic.algorithms.common.checkpoint_utils import TrainingConfig, compute_resume_state
from musclemimic.algorithms.common.dataclasses import TrainState
from musclemimic.algorithms.ppo import train as ppo_train
from musclemimic.algorithms.ppo.learning import initialize_train_state


def _collect_schedule_counts(opt_state) -> list[int]:
    counts: list[int] = []

    def _visit(node):
        if isinstance(node, optax.ScaleByScheduleState):
            counts.append(int(node.count))
        return node

    jax.tree_util.tree_map(
        _visit,
        opt_state,
        is_leaf=lambda x: isinstance(x, optax.ScaleByScheduleState),
    )
    return counts


def test_compute_resume_state_same_config_additional_timesteps():
    """Resume with unchanged config should use straightforward remaining updates."""
    cfg = TrainingConfig(num_envs=2048, num_steps=8, num_minibatches=8, update_epochs=1)
    steps_per_update = cfg.num_envs * cfg.num_steps
    completed_updates = 200
    base_global_ts = completed_updates * steps_per_update

    total_timesteps = base_global_ts + 57 * steps_per_update
    done, remaining, changed = compute_resume_state(
        checkpoint_metadata={
            "update_number": completed_updates,
            "global_timestep": base_global_ts,
            "num_envs": cfg.num_envs,
            "num_steps": cfg.num_steps,
        },
        current_config=cfg,
        total_timesteps=total_timesteps,
    )

    assert done == completed_updates
    assert remaining == 57
    assert changed is False


def test_compute_resume_state_changed_config_uses_current_steps_per_update():
    """When num_envs/num_steps change, remaining updates must use current config."""
    ckpt_num_envs = 2048
    ckpt_num_steps = 8
    ckpt_steps_per_update = ckpt_num_envs * ckpt_num_steps
    completed_updates = 120
    base_global_ts = completed_updates * ckpt_steps_per_update

    current_cfg = TrainingConfig(num_envs=1024, num_steps=20, num_minibatches=8, update_epochs=1)
    current_steps_per_update = current_cfg.num_envs * current_cfg.num_steps

    # Choose a target that is not divisible by current_steps_per_update to verify
    # ceiling division path for remaining updates.
    total_timesteps = base_global_ts + 7 * current_steps_per_update + (current_steps_per_update // 3)

    done, remaining, changed = compute_resume_state(
        checkpoint_metadata={
            "update_number": completed_updates,
            "global_timestep": base_global_ts,
            "num_envs": ckpt_num_envs,
            "num_steps": ckpt_num_steps,
        },
        current_config=current_cfg,
        total_timesteps=total_timesteps,
    )

    assert done == completed_updates
    assert changed is True
    assert remaining == 8  # 7 full + ceil(extra/current_steps_per_update)


def test_training_config_uses_global_num_envs_after_data_sharding():
    exp = SimpleNamespace(
        num_envs=256,
        global_num_envs=1024,
        num_steps=20,
        num_minibatches=8,
        update_epochs=2,
    )
    cfg = TrainingConfig.from_experiment_config(exp)
    assert cfg.num_envs == 1024
    assert cfg.num_steps == 20


def test_init_train_state_resume_with_lr_reset_and_std_reset():
    """Resume should reset schedule counter/step and optionally reset action std."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    # Includes ScaleByScheduleState so reset_lr_schedule_count has real effect.
    tx = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lambda count: jnp.asarray(1.0, dtype=jnp.float32)),
        optax.scale(-1.0),
    )

    params = {
        "w": jnp.array([1.0, -1.0], dtype=jnp.float32),
        "log_std": jnp.zeros((3,), dtype=jnp.float32),
    }
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for _ in range(3):
        _updates, opt_state = tx.update(grads, opt_state, params)

    loaded_counts = _collect_schedule_counts(opt_state)
    assert loaded_counts and all(c > 0 for c in loaded_counts)

    loaded_ts = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(123, dtype=jnp.int32),
        run_stats={},
    )

    resumed = initialize_train_state(  # pylint: disable=protected-access
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        optimizer=tx,
        agent_state=SimpleNamespace(train_state=loaded_ts),
        config=OmegaConf.create(
            {
                "reset_lr_schedule_on_resume": True,
                "reset_std_on_resume": 0.5,
            }
        ),
    )

    assert int(resumed.step) == 0
    assert jnp.allclose(resumed.params["log_std"], jnp.log(0.5), atol=1e-6)

    resumed_counts = _collect_schedule_counts(resumed.opt_state)
    assert resumed_counts, "expected ScaleByScheduleState in optimizer state"
    assert all(c == 0 for c in resumed_counts)


def test_init_train_state_resume_without_resets_preserves_step_and_std():
    """Resume without reset flags should preserve optimizer step and action std."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    tx = optax.adam(learning_rate=3e-4)
    params = {
        "w": jnp.array([0.1], dtype=jnp.float32),
        "log_std": jnp.array([0.2, -0.3], dtype=jnp.float32),
    }
    loaded_ts = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        step=jnp.asarray(77, dtype=jnp.int32),
        run_stats={},
    )

    resumed = initialize_train_state(  # pylint: disable=protected-access
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        optimizer=tx,
        agent_state=SimpleNamespace(train_state=loaded_ts),
        config=OmegaConf.create({"reset_lr_schedule_on_resume": False}),
    )

    assert int(resumed.step) == 77
    assert jnp.allclose(resumed.params["log_std"], params["log_std"])


def test_init_train_state_resume_does_not_initialize_optimizer_state():
    """Resume should use the restored optimizer state without allocating a new one."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    class NoInitTx:
        def init(self, _params):
            raise AssertionError("resume path must not initialize optimizer state")

    params = {"w": jnp.array([0.1], dtype=jnp.float32)}
    opt_state = {"raw": jnp.array([0.0], dtype=jnp.float32)}
    loaded_ts = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=NoInitTx(),
        opt_state=opt_state,
        step=jnp.asarray(77, dtype=jnp.int32),
        run_stats={},
    )

    resumed = initialize_train_state(  # pylint: disable=protected-access
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        optimizer=NoInitTx(),
        agent_state=SimpleNamespace(train_state=loaded_ts),
        config=OmegaConf.create({"reset_lr_schedule_on_resume": False}),
    )

    assert resumed.opt_state is opt_state
    assert int(resumed.step) == 77


def test_init_train_state_resume_reset_std_only_preserves_step():
    """With reset_std_on_resume only, step stays unchanged while log_std resets."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    tx = optax.adam(learning_rate=3e-4)
    params = {
        "w": jnp.array([0.3, -0.7], dtype=jnp.float32),
        "log_std": jnp.array([0.2, -0.3, 0.5], dtype=jnp.float32),
    }
    loaded_ts = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        step=jnp.asarray(314, dtype=jnp.int32),
        run_stats={},
    )

    resumed = initialize_train_state(  # pylint: disable=protected-access
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        optimizer=tx,
        agent_state=SimpleNamespace(train_state=loaded_ts),
        config=OmegaConf.create(
            {
                "reset_lr_schedule_on_resume": False,
                "reset_std_on_resume": 0.25,
            }
        ),
    )

    assert int(resumed.step) == 314
    assert jnp.allclose(resumed.params["log_std"], jnp.log(0.25), atol=1e-6)
    assert jnp.allclose(resumed.params["w"], params["w"])


def test_init_train_state_local_auto_resume_skips_one_shot_resets():
    """Local auto-resume should preserve schedule count, step, and std."""

    class DummyNetwork:
        def apply(self, *args, **kwargs):
            raise NotImplementedError

    tx = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(lambda count: jnp.asarray(1.0, dtype=jnp.float32)),
        optax.scale(-1.0),
    )

    params = {
        "w": jnp.array([0.4, -0.2], dtype=jnp.float32),
        "log_std": jnp.array([0.1, -0.4], dtype=jnp.float32),
    }
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    for _ in range(4):
        _updates, opt_state = tx.update(grads, opt_state, params)

    loaded_counts = _collect_schedule_counts(opt_state)
    assert loaded_counts and all(c > 0 for c in loaded_counts)

    loaded_ts = TrainState(
        apply_fn=DummyNetwork().apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        step=jnp.asarray(271, dtype=jnp.int32),
        run_stats={},
    )

    resumed = initialize_train_state(  # pylint: disable=protected-access
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        env=SimpleNamespace(info=SimpleNamespace(observation_space=SimpleNamespace(shape=(4,)))),
        network=DummyNetwork(),
        optimizer=tx,
        agent_state=SimpleNamespace(train_state=loaded_ts),
        config=OmegaConf.create(
            {
                "reset_lr_schedule_on_resume": True,
                "reset_std_on_resume": 0.25,
            }
        ),
        apply_resume_resets=False,
    )

    assert int(resumed.step) == 271
    assert jnp.allclose(resumed.params["log_std"], params["log_std"])
    assert _collect_schedule_counts(resumed.opt_state) == loaded_counts


def test_compute_resume_info_stored_target_used_on_auto_resume():
    """Auto-resume should use stored target_global_timestep from checkpoint."""

    config = OmegaConf.create(
        {
            "num_updates": 10,
            "total_timesteps": 320,
            "num_envs": 4,
            "num_steps": 8,
            "num_minibatches": 1,
            "update_epochs": 1,
        }
    )
    train_state = SimpleNamespace(step=jnp.asarray(3, dtype=jnp.int32))
    agent_state = SimpleNamespace(train_state=train_state)
    # Checkpoint saved during a finetune run that resolved target to 416
    resume_info = {
        "update_number": 3,
        "global_timestep": 96,
        "target_global_timestep": 416,
        "optimizer_step": 3,
        "num_envs": 4,
        "num_steps": 8,
        "num_minibatches": 1,
        "update_epochs": 1,
    }

    completed, remaining, base_ts, base_step, target = ppo_train._compute_resume_info(
        config,
        agent_state,
        resume_info,
        train_state,
        preserve_checkpoint_target=True,
    )

    assert completed == 3
    assert remaining == 10  # (416 - 96) / 32 = 10
    assert base_ts == 96
    assert base_step == 3
    assert target == 416


def test_compute_resume_info_first_explicit_resume_uses_additional():
    """First explicit resume (no stored target) should add total_timesteps on top."""

    config = OmegaConf.create(
        {
            "num_updates": 10,
            "total_timesteps": 320,
            "num_envs": 4,
            "num_steps": 8,
            "num_minibatches": 1,
            "update_epochs": 1,
        }
    )
    train_state = SimpleNamespace(step=jnp.asarray(3, dtype=jnp.int32))
    agent_state = SimpleNamespace(train_state=train_state)
    # External checkpoint — no target_global_timestep stored
    resume_info = {
        "update_number": 3,
        "global_timestep": 96,
        "target_global_timestep": 0,
        "optimizer_step": 3,
        "num_envs": 4,
        "num_steps": 8,
        "num_minibatches": 1,
        "update_epochs": 1,
    }

    completed, remaining, base_ts, base_step, target = ppo_train._compute_resume_info(
        config,
        agent_state,
        resume_info,
        train_state,
    )

    assert completed == 3
    assert remaining == 10  # (320 + 96 - 96) / 32 = 10
    assert base_ts == 96
    assert base_step == 3
    assert target == 416  # 320 + 96


def test_compute_resume_info_explicit_resume_ignores_stored_target():
    """Explicit resume should use the new relative budget unless absolute mode is enabled."""

    config = OmegaConf.create(
        {
            "num_updates": 10,
            "total_timesteps": 160,
            "num_envs": 4,
            "num_steps": 8,
            "num_minibatches": 1,
            "update_epochs": 1,
        }
    )
    train_state = SimpleNamespace(step=jnp.asarray(3, dtype=jnp.int32))
    agent_state = SimpleNamespace(train_state=train_state)
    resume_info = {
        "update_number": 3,
        "global_timestep": 96,
        "target_global_timestep": 416,
        "optimizer_step": 3,
        "num_envs": 4,
        "num_steps": 8,
        "num_minibatches": 1,
        "update_epochs": 1,
    }

    completed, remaining, base_ts, base_step, target = ppo_train._compute_resume_info(
        config,
        agent_state,
        resume_info,
        train_state,
        preserve_checkpoint_target=False,
    )

    assert completed == 3
    assert remaining == 5  # (160 + 96 - 96) / 32 = 5
    assert base_ts == 96
    assert base_step == 3
    assert target == 256


def test_compute_resume_info_absolute_budget_overrides_stored_target():
    """Absolute mode should ignore any checkpoint-stored target budget."""

    config = OmegaConf.create(
        {
            "num_updates": 10,
            "total_timesteps": 320,
            "total_timesteps_is_absolute": True,
            "num_envs": 4,
            "num_steps": 8,
            "num_minibatches": 1,
            "update_epochs": 1,
        }
    )
    train_state = SimpleNamespace(step=jnp.asarray(3, dtype=jnp.int32))
    agent_state = SimpleNamespace(train_state=train_state)
    resume_info = {
        "update_number": 3,
        "global_timestep": 96,
        "target_global_timestep": 416,
        "optimizer_step": 3,
        "num_envs": 4,
        "num_steps": 8,
        "num_minibatches": 1,
        "update_epochs": 1,
    }

    completed, remaining, base_ts, base_step, target = ppo_train._compute_resume_info(
        config,
        agent_state,
        resume_info,
        train_state,
        preserve_checkpoint_target=True,
    )

    assert completed == 3
    assert remaining == 7  # (320 - 96) / 32 = 7
    assert base_ts == 96
    assert base_step == 3
    assert target == 320


def test_compute_resume_info_finetune_preempt_preserves_target():
    """Scenario D: finetune preempted mid-run, auto-resume reads stored target."""

    config = OmegaConf.create(
        {
            "num_updates": 15,  # config says 15 but stored target overrides
            "total_timesteps": 500,
            "num_envs": 4,
            "num_steps": 8,
            "num_minibatches": 1,
            "update_epochs": 1,
        }
    )
    train_state = SimpleNamespace(step=jnp.asarray(10, dtype=jnp.int32))
    agent_state = SimpleNamespace(train_state=train_state)
    # Finetune started at global_ts=1000, target was 1000+500=1500.
    # Preempted at global_ts=1200.
    resume_info = {
        "update_number": 10,
        "global_timestep": 1200,
        "target_global_timestep": 1500,
        "optimizer_step": 10,
        "num_envs": 4,
        "num_steps": 8,
        "num_minibatches": 1,
        "update_epochs": 1,
    }

    _completed, remaining, base_ts, _base_step, target = ppo_train._compute_resume_info(
        config,
        agent_state,
        resume_info,
        train_state,
        preserve_checkpoint_target=True,
    )

    assert target == 1500
    assert remaining == 10  # (1500 - 1200) / 32 = 9.375 → ceil = 10
    assert base_ts == 1200


def test_resumed_training_preserves_resume_policy(monkeypatch):
    from musclemimic.algorithms.ppo import ppo as ppo_mod
    from musclemimic.algorithms.ppo.ppo import PPOJax

    loaded_state = SimpleNamespace(train_state="train-state", runtime_state={"step": 3})
    resume_info = {"global_timestep": 123}
    monkeypatch.setattr(ppo_mod, "load_checkpoint_for_resume", lambda *a, **k: (loaded_state, resume_info))
    captured = {}

    def fake_build_training(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "training"

    monkeypatch.setattr(ppo_mod, "build_training", fake_build_training)

    training, resume_state = PPOJax.build_resumed_training(
        env="env",
        agent_conf="agent-conf",
        checkpoint_path="/tmp/ckpt",
        val_env="val-env",
        apply_resume_resets=False,
        preserve_checkpoint_target=True,
        runtime_state_target={"step": 0},
    )
    assert training == "training"
    assert resume_state == {"train_state": "train-state", "runtime_state": {"step": 3}}
    kwargs = captured["kwargs"]
    assert kwargs["resume_info"] is resume_info
    assert kwargs["val_env"] == "val-env"
    assert kwargs["apply_resume_resets"] is False
    assert kwargs["preserve_checkpoint_target"] is True
