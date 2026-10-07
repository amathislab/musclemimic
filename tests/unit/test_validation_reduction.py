"""Tests for host-side validation metric reduction."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from musclemimic.algorithms.common.trajectory_evaluation import (
    ExactTrajectorySetStatistics,
)
from musclemimic.runner import validation_reduction
from musclemimic.utils.metrics import ValidationSummary


class _ValidationResult(NamedTuple):
    metrics: object
    rng: jnp.ndarray


def test_exact_statistics_are_summed_on_host() -> None:
    first = ExactTrajectorySetStatistics(
        episode_count=np.asarray([1.0, 2.0]),
        episode_return_sum=np.asarray([4.0, 10.0]),
        episode_length_sum=np.asarray([3.0, 7.0]),
        early_termination_count=np.asarray([1.0, 0.0]),
        horizon_timeout_count=np.asarray([0.0, 1.0]),
        covered_frame_sum=np.asarray([3.0, 6.0]),
        total_frame=np.asarray([4.0, 8.0]),
        step_count=np.asarray([3.0, 7.0]),
        step_sums={"reward_total": np.asarray([6.0, 21.0])},
    )
    second = ExactTrajectorySetStatistics(
        episode_count=np.asarray([3.0]),
        episode_return_sum=np.asarray([18.0]),
        episode_length_sum=np.asarray([11.0]),
        early_termination_count=np.asarray([1.0]),
        horizon_timeout_count=np.asarray([0.0]),
        covered_frame_sum=np.asarray([9.0]),
        total_frame=np.asarray([12.0]),
        step_count=np.asarray([11.0]),
        step_sums={"reward_total": np.asarray([44.0])},
    )

    result = validation_reduction.reduce_validation_metrics_host(
        (first, second),
        evaluate_all=True,
    )

    assert result.episode_count == pytest.approx(6.0)
    assert result.mean_episode_return == pytest.approx(32.0 / 6.0)
    assert result.mean_episode_length == pytest.approx(21.0 / 6.0)
    assert result.early_termination_count == pytest.approx(2.0)
    assert result.horizon_timeout_count == pytest.approx(1.0)
    assert result.frame_coverage == pytest.approx(18.0 / 24.0)
    assert result.reward_total == pytest.approx(71.0 / 21.0)


def test_sampled_summaries_preserve_cross_device_semantics() -> None:
    first = ValidationSummary(
        episode_count=1.0,
        mean_episode_return=0.0,
        mean_episode_length=10.0,
        early_termination_count=1.0,
        early_termination_rate=1.0,
        err_root_xyz=1.0,
        max_timestep=7,
    )
    second = ValidationSummary(
        episode_count=9.0,
        mean_episode_return=100.0,
        mean_episode_length=20.0,
        early_termination_count=3.0,
        early_termination_rate=1.0 / 3.0,
        err_root_xyz=3.0,
        max_timestep=8,
    )
    stacked = jax.tree.map(
        lambda left, right: np.stack((left, right)),
        first,
        second,
    )

    result = validation_reduction.reduce_validation_metrics_host(
        (stacked,),
        evaluate_all=False,
    )

    assert result.episode_count == pytest.approx(10.0)
    assert result.mean_episode_return == pytest.approx(90.0)
    assert result.mean_episode_length == pytest.approx(19.0)
    assert result.early_termination_count == pytest.approx(4.0)
    assert result.early_termination_rate == pytest.approx(0.4)
    assert result.err_root_xyz == pytest.approx(np.sqrt(5.0))
    assert result.max_timestep == 8


def test_finish_validation_waits_then_uses_only_host_reduction(monkeypatch) -> None:
    events: list[str] = []
    local_result = _ValidationResult(
        metrics=ValidationSummary(),
        rng=jnp.asarray(7, dtype=jnp.int32),
    )
    local_metrics = ValidationSummary(episode_count=np.asarray([1.0]))
    remote_metrics = ValidationSummary(episode_count=np.asarray([2.0]))

    def block_until_ready(result):
        assert result is local_result
        events.append("local-ready")
        return result

    def addressable_metrics(metrics):
        assert metrics is local_result.metrics
        events.append("copy-addressable-shards")
        return local_metrics

    def process_exchange(event_name: str, metrics):
        assert event_name == "validation_00000001"
        assert metrics is local_metrics
        events.append("host-exchange")
        return local_metrics, remote_metrics

    def reduce_host(process_metrics, *, evaluate_all):
        assert process_metrics == (local_metrics, remote_metrics)
        assert evaluate_all is False
        events.append("host-reduce")
        return ValidationSummary(episode_count=np.asarray(3.0))

    monkeypatch.setattr(validation_reduction.jax, "block_until_ready", block_until_ready)
    monkeypatch.setattr(
        validation_reduction,
        "addressable_validation_metrics",
        addressable_metrics,
    )
    monkeypatch.setattr(
        validation_reduction,
        "reduce_validation_metrics_host",
        reduce_host,
    )

    result = validation_reduction.finish_validation(
        local_result,
        process_exchange,
        evaluate_all=False,
        event_name="validation_00000001",
    )

    assert events == [
        "local-ready",
        "copy-addressable-shards",
        "host-exchange",
        "host-reduce",
    ]
    assert float(result.metrics.episode_count) == 3.0
    assert int(result.rng) == 7
