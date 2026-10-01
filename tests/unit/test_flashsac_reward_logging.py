"""Regression tests for task-specific FlashSAC reward diagnostics."""

import jax
import jax.numpy as jnp
import pytest

from musclemimic.algorithms.flashsac.train import mean_reward_components


def test_mean_reward_components_omits_metrics_absent_from_task_info():
    info = {
        "reward_total": jnp.asarray([1.0, 3.0]),
        "err_root_yaw": jnp.asarray([0.2, 0.4]),
        "activation_energy_raw": jnp.asarray([0.1, 0.3]),
    }

    metrics = jax.jit(mean_reward_components)(info)

    assert set(metrics) == {"reward/total", "err/root_yaw"}
    assert float(metrics["reward/total"]) == 2.0
    assert float(metrics["err/root_yaw"]) == pytest.approx(0.3)
