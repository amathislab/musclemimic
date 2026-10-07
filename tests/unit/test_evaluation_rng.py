"""Tests for validation RNG streams shared by metrics and videos."""

import jax
import numpy as np

from musclemimic.algorithms.common.evaluation_rng import (
    evaluation_rollout_keys,
    evaluation_stream_keys,
    evaluation_video_seed,
)


def test_evaluation_streams_are_reproducible_and_validation_specific():
    first_run = evaluation_rollout_keys(0, validation_number=3, num_devices=2)
    second_run = evaluation_rollout_keys(0, validation_number=3, num_devices=2)
    next_validation = evaluation_rollout_keys(0, validation_number=4, num_devices=2)

    np.testing.assert_array_equal(
        jax.random.key_data(first_run),
        jax.random.key_data(second_run),
    )
    assert not np.array_equal(
        jax.random.key_data(first_run),
        jax.random.key_data(next_validation),
    )
    assert not np.array_equal(
        jax.random.key_data(first_run[0]),
        jax.random.key_data(first_run[1]),
    )


def test_video_seed_uses_independent_split_of_same_evaluation_key():
    rollout_key, video_key = evaluation_stream_keys(11, validation_number=7)

    assert evaluation_video_seed(11, validation_number=7) == evaluation_video_seed(
        11,
        validation_number=7,
    )
    assert evaluation_video_seed(11, validation_number=7) != evaluation_video_seed(
        11,
        validation_number=8,
    )
    assert not np.array_equal(
        jax.random.key_data(rollout_key),
        jax.random.key_data(video_key),
    )
