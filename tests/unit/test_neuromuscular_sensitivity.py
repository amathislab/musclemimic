import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analyses"))

from neuromuscular import NeuroMuscularAnalysis


def _muscle_profile(offset=0.0):
    time = np.linspace(0, 2 * np.pi, 100)
    return np.stack(
        [
            np.sin(time) + offset,
            np.cos(time) + offset,
            np.sin(2 * time) + 0.3 * np.cos(time) + offset,
        ]
    )


def _analysis_with_synthetic_profiles():
    analysis = NeuroMuscularAnalysis.__new__(NeuroMuscularAnalysis)
    analysis.subject_based = True
    analysis._current_gait_position = 0
    analysis.subj_list = [51, 52, 53]
    analysis.selected_joint_names = ["hip", "knee"]
    analysis.selected_muscle_names = ["m1", "m2", "m3"]

    time = np.linspace(0, 1, 100)
    joint_51 = np.stack([time, time**2])
    joint_52 = np.stack([2 * time, 2 * time**2])
    muscle_51 = _muscle_profile()
    muscle_52 = _muscle_profile(offset=0.2)

    analysis.policies_data = [
        {
            "config": {"label": "test policy"},
            "per_subject_avg_joint_gaits": {51: joint_51, 52: joint_52},
            "per_subject_avg_muscle_gaits": {51: muscle_51, 52: muscle_52},
            "per_subject_avg_muscle_rms": {
                51: analysis.compute_rsm(muscle_51),
                52: analysis.compute_rsm(muscle_52),
            },
        }
    ]
    analysis._human_joint_by_step_train = {
        51: {0: joint_51},
        52: {0: joint_52 + 1.0},
        53: {0: joint_51},
    }
    analysis._human_muscle_by_step_train = {
        51: {0: 2 * muscle_51 + 0.5},
        52: {0: 2 * muscle_52 - 0.5},
        53: {0: muscle_51},
    }
    return analysis


def test_policy_to_earlier_human_sensitivity_is_subject_matched():
    analysis = _analysis_with_synthetic_profiles()

    analysis.compute_policy_to_earlier_human_sensitivity()

    result = analysis.policies_data[0]["earlier_human_sensitivity"]
    assert result["n"] == 2
    assert result["subjects"] == [51, 52]
    assert result["joint_rmse"]["all_joints"] == pytest.approx([0.0, 1.0])
    assert result["muscle_corr"]["all_muscles"] == pytest.approx([1.0, 1.0])
    assert result["rsa_spearman"] == pytest.approx([1.0, 1.0])
    assert result["rsa_pearson"] == pytest.approx([1.0, 1.0])


def test_policy_to_earlier_human_sensitivity_skips_missing_human_gait():
    analysis = _analysis_with_synthetic_profiles()
    analysis._human_joint_by_step_train[52] = {}
    analysis._human_muscle_by_step_train[52] = {}

    analysis.compute_policy_to_earlier_human_sensitivity()

    result = analysis.policies_data[0]["earlier_human_sensitivity"]
    assert result["n"] == 1
    assert result["subjects"] == [51]
