"""
Consolidated PPO tests.

Covers loss functions, optimizer schedules, MoE helpers, and env wrapper order.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from loco_mujoco.core.utils.env import Box
from musclemimic.algorithms.common.env_utils import wrap_env
from musclemimic.algorithms.common.optimizer import (
    build_optimizer,
    get_optimizer,
    linear_lr_schedule,
    warmup_cosine_lr_schedule,
)
from musclemimic.algorithms.ppo.loss import (
    PPOLossOutput,
    approx_kl,
    normalize_advantages,
    ppo_actor_loss,
    ppo_loss,
    ppo_value_loss,
)
from musclemimic.algorithms.ppo.moe import aggregate_moe_metrics, zero_moe_metrics
from musclemimic.algorithms.ppo.ppo import PPOJax
from musclemimic.core.wrappers.mjx import AutoResetWrapper, LogWrapper, NStepWrapper, VecEnv


class _DummyEnv:
    def __init__(self, mjx_env: bool):
        self.mjx_env = mjx_env
        obs_space = Box(low=-np.ones(2), high=np.ones(2))
        self.info = types.SimpleNamespace(observation_space=obs_space)

    def reset(self, _rng_key):
        obs = jnp.zeros((2,), dtype=jnp.float32)
        return obs, obs

    def reset_to(self, rng_key, _traj_idx):
        return self.reset(rng_key)

    def reset_to_step(self, rng_key, _traj_idx, _subtraj_step):
        return self.reset(rng_key)

    def step(self, state, _action):
        obs = jnp.zeros((2,), dtype=jnp.float32)
        reward = jnp.array(0.0, dtype=jnp.float32)
        absorbing = jnp.array(False)
        done = jnp.array(False)
        info = {}
        return obs, reward, absorbing, done, info, state


class _DummyTrainEnv:
    def __init__(self, obs_dim: int = 4, act_dim: int = 2):
        obs_space = Box(low=-np.ones(obs_dim), high=np.ones(obs_dim))
        act_space = Box(low=-np.ones(act_dim), high=np.ones(act_dim))
        self.info = types.SimpleNamespace(action_space=act_space, observation_space=obs_space)
        self.mdp_info = types.SimpleNamespace(observation_space=obs_space)


def _make_optimizer_config(
    *,
    anneal_lr: bool = False,
    schedule: str = "fixed",
    lr_schedule_type: str = "linear",
    warmup_steps: int | None = None,
    min_lr_ratio: float = 0.0,
):
    return OmegaConf.create(
        {
            "experiment": {
                "lr": 3e-4,
                "num_minibatches": 4,
                "update_epochs": 2,
                "num_updates": 10,
                "anneal_lr": anneal_lr,
                "lr_schedule_type": lr_schedule_type,
                "warmup_steps": warmup_steps,
                "min_lr_ratio": min_lr_ratio,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "schedule": schedule,
                "optimizer_type": "adamw",
            }
        }
    )


class TestPPOValueLoss:
    def test_no_clipping_within_range(self):
        value = jnp.array([1.1, 0.9])
        value_old = jnp.array([1.0, 1.0])
        targets = jnp.array([1.0, 1.0])
        clip_eps = 0.2

        loss = ppo_value_loss(value, value_old, targets, clip_eps)
        expected = 0.5 * jnp.mean(jnp.square(value - targets))

        assert jnp.allclose(loss, expected, atol=1e-6)

    def test_clipping_outside_range(self):
        value = jnp.array([2.0])
        value_old = jnp.array([1.0])
        targets = jnp.array([1.5])
        clip_eps = 0.2

        loss = ppo_value_loss(value, value_old, targets, clip_eps)
        expected = 0.5 * max((2.0 - 1.5) ** 2, (1.2 - 1.5) ** 2)

        assert jnp.allclose(loss, expected, atol=1e-6)

    def test_formula_hand_computed(self):
        value = jnp.array([1.1, 2.5])
        value_old = jnp.array([1.0, 2.0])
        targets = jnp.array([1.0, 2.0])
        clip_eps = 0.2

        loss = ppo_value_loss(value, value_old, targets, clip_eps)
        expected = 0.5 * jnp.mean(
            jnp.array(
                [
                    (1.1 - 1.0) ** 2,
                    max((2.5 - 2.0) ** 2, (2.2 - 2.0) ** 2),
                ]
            )
        )

        assert jnp.allclose(loss, expected, atol=1e-6)


class TestPPOActorLoss:
    def test_positive_advantage_encourages_action(self):
        log_prob = jnp.array([0.0])
        log_prob_old = jnp.array([0.0])
        advantages = jnp.array([1.0])
        clip_eps = 0.2

        loss = ppo_actor_loss(log_prob, log_prob_old, advantages, clip_eps)

        assert loss < 0.0

    def test_clipping_limits_ratio(self):
        log_prob = jnp.array([2.0])
        log_prob_old = jnp.array([0.0])
        advantages = jnp.array([1.0])
        clip_eps = 0.2

        loss = ppo_actor_loss(log_prob, log_prob_old, advantages, clip_eps)

        assert jnp.allclose(loss, jnp.array(-1.2), atol=1e-5)

    def test_formula_hand_computed(self):
        log_prob = jnp.array([0.0, 0.1])
        log_prob_old = jnp.array([0.0, 0.0])
        advantages = jnp.array([1.0, -1.0])
        clip_eps = 0.2

        loss = ppo_actor_loss(log_prob, log_prob_old, advantages, clip_eps)
        ratio = jnp.exp(log_prob - log_prob_old)
        loss_actor1 = ratio * advantages
        loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        expected = -jnp.minimum(loss_actor1, loss_actor2).mean()

        assert jnp.allclose(loss, expected, atol=1e-6)

    def test_zero_advantage_gives_zero_loss(self):
        """
        Zero advantages should produce zero actor loss,
        regardless of log-probabilities.
        """
        log_prob = jnp.array([1.0, -1.0])
        log_prob_old = jnp.array([0.0, 0.0])
        advantages = jnp.zeros(2)
        clip_eps = 0.2

        loss = ppo_actor_loss(log_prob, log_prob_old, advantages, clip_eps)

        assert jnp.allclose(loss, 0.0, atol=1e-6)

    def test_negative_advantage_discourages_action(self):
        """
        Negative advantages should increase loss
        (discourage the action).
        """
        log_prob = jnp.array([0.2])
        log_prob_old = jnp.array([0.0])
        advantages = jnp.array([-1.0])
        clip_eps = 0.2

        loss = ppo_actor_loss(log_prob, log_prob_old, advantages, clip_eps)

        assert loss > 0.0


class TestKLDivergence:
    def test_approx_kl_formula(self):
        log_prob_old = jnp.array([-0.1, -0.2])
        log_prob = jnp.array([-0.2, -0.2])

        expected = jnp.mean(log_prob_old - log_prob)
        assert jnp.allclose(approx_kl(log_prob_old, log_prob), expected, atol=1e-6)


class TestNormalizeAdvantages:
    def test_zero_mean_unit_std(self):
        advantages = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_advantages(advantages)

        assert jnp.allclose(normalized.mean(), 0.0, atol=1e-6)
        assert jnp.allclose(normalized.std(), 1.0, atol=1e-5)

    def test_constant_input_stability(self):
        advantages = jnp.ones(8)
        normalized = normalize_advantages(advantages)

        assert jnp.all(jnp.isfinite(normalized))

    def test_normalization_preserves_order(self):
        """
        Normalization should preserve relative ordering.
        """
        advantages = jnp.array([1.0, 2.0, 10.0])
        normalized = normalize_advantages(advantages)

        assert jnp.argsort(advantages).tolist() == jnp.argsort(normalized).tolist()


class TestPPOLoss:
    def test_combined_formula(self):
        log_prob = jnp.array([0.0, 0.1])
        log_prob_old = jnp.array([-0.1, 0.0])
        value = jnp.array([1.0, 2.0])
        value_old = jnp.array([0.8, 2.2])
        targets = jnp.array([1.5, 1.0])
        advantages = jnp.array([1.0, 2.0])
        entropy = jnp.array(0.5)
        clip_eps = 0.2
        vf_coef = 0.5
        ent_coef = 0.01

        normalized_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = jnp.exp(log_prob - log_prob_old)
        loss_actor1 = ratio * normalized_adv
        loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * normalized_adv
        expected_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

        value_pred_clipped = value_old + (value - value_old).clip(-clip_eps, clip_eps)
        v_loss = (
            0.5
            * jnp.maximum(
                jnp.square(value - targets),
                jnp.square(value_pred_clipped - targets),
            ).mean()
        )

        expected_total = expected_actor + vf_coef * v_loss - ent_coef * entropy
        expected_kl = jnp.mean(log_prob_old - log_prob)

        output = ppo_loss(
            log_prob=log_prob,
            log_prob_old=log_prob_old,
            value=value,
            value_old=value_old,
            targets=targets,
            advantages=advantages,
            entropy=entropy,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
        )

        assert jnp.allclose(output.actor_loss, expected_actor, atol=1e-6)
        assert jnp.allclose(output.value_loss, v_loss, atol=1e-6)
        assert jnp.allclose(output.total_loss, expected_total, atol=1e-6)
        assert jnp.allclose(output.kl_mean, expected_kl, atol=1e-6)

    def test_clip_eps_vf_override(self):
        log_prob = jnp.array([0.1, -0.1])
        log_prob_old = jnp.array([0.0, 0.0])
        value = jnp.array([2.0, 0.0])
        value_old = jnp.array([1.0, 0.5])
        targets = jnp.array([1.5, 0.25])
        advantages = jnp.array([0.5, -0.5])
        entropy = jnp.array(0.0)
        clip_eps = 0.2
        clip_eps_vf = 0.05

        output = ppo_loss(
            log_prob=log_prob,
            log_prob_old=log_prob_old,
            value=value,
            value_old=value_old,
            targets=targets,
            advantages=advantages,
            entropy=entropy,
            clip_eps=clip_eps,
            vf_coef=1.0,
            ent_coef=0.0,
            clip_eps_vf=clip_eps_vf,
        )

        value_pred_clipped = value_old + (value - value_old).clip(-clip_eps_vf, clip_eps_vf)
        expected_value_loss = (
            0.5
            * jnp.maximum(
                jnp.square(value - targets),
                jnp.square(value_pred_clipped - targets),
            ).mean()
        )

        assert jnp.allclose(output.value_loss, expected_value_loss, atol=1e-6)

    def test_gradient_flow(self):
        log_prob = jnp.array([0.1, -0.2, 0.3])
        log_prob_old = jnp.array([0.0, 0.0, 0.0])
        value = jnp.array([1.0, 0.5, -0.5])
        value_old = jnp.array([0.8, 0.6, -0.4])
        targets = jnp.array([1.2, 0.4, -0.2])
        advantages = jnp.array([0.5, -0.1, 1.0])
        entropy = jnp.array(0.2)

        def loss_fn(lp):
            return ppo_loss(
                log_prob=lp,
                log_prob_old=log_prob_old,
                value=value,
                value_old=value_old,
                targets=targets,
                advantages=advantages,
                entropy=entropy,
                clip_eps=0.2,
                vf_coef=0.5,
                ent_coef=0.01,
            ).total_loss

        grad = jax.grad(loss_fn)(log_prob)

        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(jnp.abs(grad) > 1e-6)

    def test_jit_compatible(self):
        log_prob = jnp.array([0.0, 0.1])
        log_prob_old = jnp.array([0.0, 0.0])
        value = jnp.array([1.0, 2.0])
        value_old = jnp.array([1.0, 1.8])
        targets = jnp.array([1.5, 1.0])
        advantages = jnp.array([1.0, -1.0])
        entropy = jnp.array(0.1)

        jit_fn = jax.jit(ppo_loss)
        output = jit_fn(
            log_prob,
            log_prob_old,
            value,
            value_old,
            targets,
            advantages,
            entropy,
            0.2,
            0.5,
            0.01,
        )

        assert jnp.isfinite(output.total_loss)

    def test_zero_entropy_coef_removes_entropy_term(self):
        """
        ent_coef=0 must remove entropy contribution entirely.
        """
        log_prob = jnp.array([0.1])
        log_prob_old = jnp.array([0.0])
        value = jnp.array([1.0])
        value_old = jnp.array([1.0])
        targets = jnp.array([1.0])
        advantages = jnp.array([1.0])
        entropy = jnp.array(10.0)

        out = ppo_loss(
            log_prob,
            log_prob_old,
            value,
            value_old,
            targets,
            advantages,
            entropy,
            clip_eps=0.2,
            vf_coef=1.0,
            ent_coef=0.0,
        )

        assert jnp.allclose(out.total_loss, out.actor_loss + out.value_loss)

    def test_loss_invariant_to_batch_size(self):
        """
        PPO loss should be mean-based, not sum-based.
        """
        log_prob = jnp.array([0.1])
        log_prob_old = jnp.array([0.0])
        value = jnp.array([1.0])
        value_old = jnp.array([1.0])
        targets = jnp.array([1.0])
        advantages = jnp.array([1.0])
        entropy = jnp.array(0.1)

        out1 = ppo_loss(log_prob, log_prob_old, value, value_old, targets, advantages, entropy, 0.2, 1.0, 0.0)

        out2 = ppo_loss(
            jnp.repeat(log_prob, 10),
            jnp.repeat(log_prob_old, 10),
            jnp.repeat(value, 10),
            jnp.repeat(value_old, 10),
            jnp.repeat(targets, 10),
            jnp.repeat(advantages, 10),
            entropy,
            0.2,
            1.0,
            0.0,
        )

        assert jnp.allclose(out1.total_loss, out2.total_loss, atol=1e-6)


class TestPPOLossOutput:
    def test_pytree_roundtrip(self):
        output = PPOLossOutput(
            total_loss=jnp.array(1.0),
            value_loss=jnp.array(0.1),
            actor_loss=jnp.array(0.2),
            entropy=jnp.array(0.3),
            kl_mean=jnp.array(0.01),
        )

        flat, treedef = jax.tree_util.tree_flatten(output)
        restored = jax.tree_util.tree_unflatten(treedef, flat)

        assert jnp.allclose(restored.total_loss, output.total_loss)
        assert jnp.allclose(restored.value_loss, output.value_loss)
        assert jnp.allclose(restored.actor_loss, output.actor_loss)
        assert jnp.allclose(restored.entropy, output.entropy)
        assert jnp.allclose(restored.kl_mean, output.kl_mean)

    def test_vmap_compatible(self):
        log_prob = jnp.array([0.0, 0.1])
        log_prob_old = jnp.array([0.0, 0.0])
        value = jnp.array([1.0, 2.0])
        value_old = jnp.array([1.0, 1.8])
        targets = jnp.array([1.5, 1.0])
        entropy = jnp.array(0.1)

        stacked_adv = jnp.stack([jnp.array([1.0, -1.0]), jnp.array([0.5, 0.5])])

        def compute(adv):
            return ppo_loss(
                log_prob=log_prob,
                log_prob_old=log_prob_old,
                value=value,
                value_old=value_old,
                targets=targets,
                advantages=adv,
                entropy=entropy,
                clip_eps=0.2,
                vf_coef=0.5,
                ent_coef=0.01,
            )

        outputs = jax.vmap(compute)(stacked_adv)
        assert outputs.total_loss.shape == (2,)
        assert outputs.value_loss.shape == (2,)


class TestLRSchedules:
    def test_linear_decay_endpoints(self):
        lr = 1e-3
        num_minibatches = 2
        update_epochs = 3
        num_updates = 4
        steps_per_update = num_minibatches * update_epochs

        start = linear_lr_schedule(0, num_minibatches, update_epochs, lr, num_updates)
        end = linear_lr_schedule(steps_per_update * num_updates, num_minibatches, update_epochs, lr, num_updates)

        assert jnp.allclose(start, lr, atol=1e-6)
        assert jnp.allclose(end, 0.0, atol=1e-6)

    def test_warmup_cosine_phases(self):
        lr = 1e-3
        num_minibatches = 1
        update_epochs = 1
        num_updates = 6
        warmup_steps = 2

        lr0 = warmup_cosine_lr_schedule(0, num_minibatches, update_epochs, lr, num_updates, warmup_steps, 0.0)
        lr1 = warmup_cosine_lr_schedule(1, num_minibatches, update_epochs, lr, num_updates, warmup_steps, 0.0)
        lr2 = warmup_cosine_lr_schedule(2, num_minibatches, update_epochs, lr, num_updates, warmup_steps, 0.0)
        lr5 = warmup_cosine_lr_schedule(5, num_minibatches, update_epochs, lr, num_updates, warmup_steps, 0.0)

        assert jnp.allclose(lr0, 0.0, atol=1e-6)
        assert lr1 < lr2
        assert lr2 <= lr
        assert lr5 < lr2

    def test_min_lr_respected(self):
        lr = 1e-3
        num_minibatches = 1
        update_epochs = 1
        num_updates = 5
        warmup_steps = 1
        min_ratio = 0.1

        lr_end = warmup_cosine_lr_schedule(
            num_updates, num_minibatches, update_epochs, lr, num_updates, warmup_steps, min_ratio
        )

        assert jnp.allclose(lr_end, lr * min_ratio, atol=1e-6)

    def test_warmup_cosine_handles_warmup_beyond_num_updates(self):
        lr = 1e-3
        num_minibatches = 1
        update_epochs = 1
        num_updates = 5
        warmup_steps = 10
        min_ratio = 0.1

        lr5 = warmup_cosine_lr_schedule(5, num_minibatches, update_epochs, lr, num_updates, warmup_steps, min_ratio)
        lr10 = warmup_cosine_lr_schedule(10, num_minibatches, update_epochs, lr, num_updates, warmup_steps, min_ratio)
        lr11 = warmup_cosine_lr_schedule(11, num_minibatches, update_epochs, lr, num_updates, warmup_steps, min_ratio)

        assert jnp.isfinite(lr5)
        assert jnp.isfinite(lr10)
        assert jnp.isfinite(lr11)
        assert jnp.allclose(lr5, lr * 0.5, atol=1e-6)
        assert jnp.allclose(lr10, lr, atol=1e-6)
        assert jnp.allclose(lr11, lr * min_ratio, atol=1e-6)

    def test_linear_lr_monotonic(self):
        """
        Linear schedule must be non-increasing.
        """
        lr = 1e-3
        vals = [linear_lr_schedule(i, 1, 1, lr, 10) for i in range(10)]

        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


class TestOptimizer:
    def test_get_optimizer_fixed_vs_annealed(self, monkeypatch):
        import musclemimic.algorithms.common.optimizer as optimizer

        captured = []
        original = optimizer.build_optimizer

        def spy_build_optimizer(*args, **kwargs):
            captured.append(kwargs["learning_rate"])
            return original(*args, **kwargs)

        monkeypatch.setattr(optimizer, "build_optimizer", spy_build_optimizer)

        config_fixed = _make_optimizer_config(anneal_lr=False, schedule="fixed")
        optimizer.get_optimizer(config_fixed.experiment)
        assert isinstance(captured[-1], float)

        config_anneal = _make_optimizer_config(anneal_lr=True, schedule="fixed")
        optimizer.get_optimizer(config_anneal.experiment)
        assert callable(captured[-1])

        config_cosine = _make_optimizer_config(
            anneal_lr=True,
            schedule="fixed",
            lr_schedule_type="warmup_cosine",
            warmup_steps=12,
            min_lr_ratio=0.1,
        )
        optimizer.get_optimizer(config_cosine.experiment)
        schedule_fn = captured[-1]

        assert callable(schedule_fn)
        assert jnp.isfinite(schedule_fn(0))
        assert jnp.isfinite(schedule_fn(12 * 8))
        assert jnp.isfinite(schedule_fn(13 * 8))

    def test_build_optimizer_rejects_unknown(self):
        with pytest.raises(ValueError):
            build_optimizer(
                optimizer_type="unknown",
                learning_rate=1e-3,
                weight_decay=0.0,
                max_grad_norm=1.0,
            )

    def test_gradient_clipping_applied(self):
        max_grad_norm = 1.0
        tx = build_optimizer(
            optimizer_type="adamw",
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=max_grad_norm,
        )

        # Large gradient that should be clipped (norm ~37.4)
        params = {"w": jnp.array([1.0, 2.0, 3.0])}
        grads = {"w": jnp.array([10.0, 20.0, 30.0])}
        opt_state = tx.init(params)

        updates, _ = tx.update(grads, opt_state, params)

        # After clipping + optimizer scaling, updates should be much smaller than original grads
        original_norm = jnp.sqrt(jnp.sum(jnp.array([10.0, 20.0, 30.0]) ** 2))
        update_norm = jnp.sqrt(jnp.sum(updates["w"] ** 2))
        # Clipping reduces norm from ~37.4 to 1.0, then adamw applies lr scaling
        assert update_norm < original_norm * 0.1


class TestMoEMetrics:
    def test_aggregate_moe_metrics(self):
        moe_metrics = {
            "total_moe_loss": 2.0,
            "actor": {
                "layer1_metrics": {
                    "gate_entropy": 1.0,
                    "expert_utilization_var": 2.0,
                    "top2_expert_usage": 3.0,
                    "gate_weights_mean": 4.0,
                    "gate_weights_std": 5.0,
                }
            },
            "critic": {
                "layer2_metrics": {
                    "gate_entropy": 3.0,
                    "expert_utilization_var": 4.0,
                    "top2_expert_usage": 5.0,
                    "gate_weights_mean": 6.0,
                    "gate_weights_std": 7.0,
                }
            },
        }

        out = aggregate_moe_metrics(moe_metrics)

        expected = (
            2.0,
            (1.0 + 3.0) / 2.0,
            (2.0 + 4.0) / 2.0,
            (3.0 + 5.0) / 2.0,
            (4.0 + 6.0) / 2.0,
            (5.0 + 7.0) / 2.0,
        )

        assert out == expected

    def test_zero_moe_metrics(self):
        assert zero_moe_metrics() == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class TestEnvUtils:
    def test_wrapper_order(self):
        env = _DummyEnv(mjx_env=True)
        config = OmegaConf.create({"normalize_env": False, "gamma": 0.99, "len_obs_history": 1})

        wrapped = wrap_env(env, config)

        assert isinstance(wrapped, AutoResetWrapper)
        assert isinstance(wrapped.env, LogWrapper)
        assert isinstance(wrapped.env.env, VecEnv)
        assert wrapped.env.env.env is env

    def test_len_obs_history_path(self):
        env = _DummyEnv(mjx_env=True)
        config = OmegaConf.create({"normalize_env": False, "gamma": 0.99, "len_obs_history": 4})

        wrapped = wrap_env(env, config)

        assert isinstance(wrapped, AutoResetWrapper)
        assert isinstance(wrapped.env.env.env, NStepWrapper)


class TestPPOInitAgentConf:
    def test_sets_num_updates_in_struct_config(self):
        env = _DummyTrainEnv()
        config = OmegaConf.create(
            {
                "experiment": {
                    "total_timesteps": 1000,
                    "num_envs": 10,
                    "ppo_config": {
                        "num_steps": 5,
                        "update_epochs": 2,
                        "num_minibatches": 2,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_eps": 0.2,
                        "clip_eps_vf": 0.2,
                        "init_std": 1.0,
                        "learnable_std": True,
                        "ent_coef": 0.0,
                        "vf_coef": 0.5,
                    },
                    "validation": {"num": 4},
                    "actor_hidden_layers": [32],
                    "critic_hidden_layers": [32],
                    "activation": "tanh",
                    "use_layernorm": False,
                    "layernorm_eps": 1e-5,
                    "lr": 3e-4,
                    "anneal_lr": False,
                    "lr_schedule_type": "linear",
                    "warmup_steps": None,
                    "min_lr_ratio": 0.0,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "optimizer_type": "adamw",
                }
            }
        )
        OmegaConf.set_struct(config, True)

        PPOJax.init_agent_conf(env, config)

        assert config.experiment.num_updates == 20
        assert config.experiment.minibatch_size == 25
        assert config.experiment.validation_interval == 5

    def test_validation_frequency_is_capped_at_once_per_update(self):
        env = _DummyTrainEnv()
        config = OmegaConf.create(
            {
                "experiment": {
                    "total_timesteps": 50,
                    "num_envs": 10,
                    "ppo_config": {
                        "num_steps": 5,
                        "update_epochs": 1,
                        "num_minibatches": 1,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_eps": 0.2,
                        "clip_eps_vf": 0.2,
                        "init_std": 1.0,
                        "learnable_std": True,
                        "ent_coef": 0.0,
                        "vf_coef": 0.5,
                    },
                    "validation": {"num": 100},
                    "actor_hidden_layers": [32],
                    "critic_hidden_layers": [32],
                    "activation": "tanh",
                    "use_layernorm": False,
                    "layernorm_eps": 1e-5,
                    "lr": 3e-4,
                    "anneal_lr": False,
                    "lr_schedule_type": "linear",
                    "warmup_steps": None,
                    "min_lr_ratio": 0.0,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "optimizer_type": "adamw",
                }
            }
        )

        PPOJax.init_agent_conf(env, config)

        assert config.experiment.num_updates == 1
        assert config.experiment.validation_interval == 1
        assert config.experiment.validation.num == 1
