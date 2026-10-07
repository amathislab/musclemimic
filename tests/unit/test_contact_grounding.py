"""Tests for source-contact-aware root-z correction."""

from types import SimpleNamespace

import numpy as np
import pytest

import loco_mujoco.smpl.contact_grounding as contact_grounding
from loco_mujoco.smpl.contact_grounding import (
    ContactDetectionConfig,
    RootZCorrectionConfig,
    SupportDefinition,
    contact_frame_clearance,
    infer_support_contacts,
    solve_contact_aware_root_z_shift,
)


def _foot_definitions() -> dict[str, SupportDefinition]:
    return {
        "left_foot": SupportDefinition(
            source_frame_names=("left",),
            robot_body_names=frozenset(("left",)),
            enter_height_m=0.06,
            exit_height_m=0.08,
        ),
        "right_foot": SupportDefinition(
            source_frame_names=("right",),
            robot_body_names=frozenset(("right",)),
            enter_height_m=0.06,
            exit_height_m=0.08,
        ),
    }


def test_infer_support_contacts_uses_height_speed_and_minimum_duration() -> None:
    target_positions_m = np.zeros((8, 2, 3), dtype=np.float64)

    contacts = infer_support_contacts(
        target_positions_m,
        ("left", "right"),
        frequency_hz=30.0,
        definitions=_foot_definitions(),
        config=ContactDetectionConfig(),
    )

    np.testing.assert_array_equal(contacts["left_foot"], [False, True, True, True, True, True, True, False])
    np.testing.assert_array_equal(contacts["right_foot"], contacts["left_foot"])


def test_contact_aware_root_z_shift_is_robust_to_terminal_penetration() -> None:
    global_clearance_m = np.array([0.03, 0.04, 0.03, 0.04, -0.135])
    contacts = {"left_foot": np.array([True, True, True, True, False])}
    support_clearance_m = {"left_foot": global_clearance_m.copy()}

    correction_m, _diagnostics = solve_contact_aware_root_z_shift(
        global_clearance_m,
        support_clearance_m,
        contacts,
        RootZCorrectionConfig(),
    )

    corrected_clearance_m = global_clearance_m + correction_m
    global_minimum_shift_m = -float(np.min(global_clearance_m))
    assert np.min(corrected_clearance_m) >= -RootZCorrectionConfig().penetration_tolerance_m - 1e-12
    assert np.mean(np.abs(corrected_clearance_m[:4])) < np.mean(global_clearance_m[:4] + global_minimum_shift_m)
    assert np.ptp(correction_m) > 0.10


@pytest.mark.parametrize("optimality", (9.65e-8, 1.01e-7))
def test_contact_aware_root_z_shift_guards_iteration_limit_solution(
    monkeypatch: pytest.MonkeyPatch,
    optimality: float,
) -> None:
    result = SimpleNamespace(
        success=False,
        status=0,
        nit=2000,
        optimality=optimality,
        cost=0.03,
        message="The maximum number of iterations is exceeded.",
        x=np.zeros(2),
    )
    monkeypatch.setattr(contact_grounding, "lsq_linear", lambda *_args, **_kwargs: result)
    inputs = (
        np.zeros(2),
        {"left_foot": np.zeros(2)},
        {"left_foot": np.ones(2, dtype=bool)},
        RootZCorrectionConfig(),
    )

    if optimality <= RootZCorrectionConfig().solver_iteration_limit_optimality_tolerance:
        correction_m, diagnostics = solve_contact_aware_root_z_shift(*inputs)
        np.testing.assert_array_equal(correction_m, np.zeros(2))
        assert diagnostics["root_z_solver_reached_iteration_limit"] == 1
    else:
        with pytest.raises(RuntimeError, match="maximum number of iterations"):
            solve_contact_aware_root_z_shift(*inputs)


def test_contact_frame_clearance_ignores_intended_flight() -> None:
    contacts = {
        "left_foot": np.array([True, False, False]),
        "right_foot": np.array([True, True, False]),
    }
    support_clearance_m = {
        "left_foot": np.array([0.01, 0.30, 0.40]),
        "right_foot": np.array([0.02, 0.03, 0.50]),
    }

    clearance_m = contact_frame_clearance(support_clearance_m, contacts)

    np.testing.assert_allclose(clearance_m[:2], [0.02, 0.03])
    assert np.isnan(clearance_m[2])
