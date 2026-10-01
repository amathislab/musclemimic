"""Public integration boundary for external retargeting repositories."""

import mujoco
import numpy as np
import pytest

from loco_mujoco.smpl.retargeting import (
    _scalar_joint_velocity_limits,
)
from loco_mujoco.smpl.retargeting import (
    fit_gmr_motion as internal_fit_gmr_motion,
)
from loco_mujoco.smpl.retargeting import (
    fit_smpl_motion as internal_fit_smpl_motion,
)
from loco_mujoco.smpl.retargeting import (
    retargeting_cache_dir as internal_retargeting_cache_dir,
)
from musclemimic.retargeting import fit_gmr_motion, fit_smpl_motion, retargeting_cache_dir


def test_public_api_exposes_shared_baseline_implementations():
    assert fit_smpl_motion is internal_fit_smpl_motion
    assert fit_gmr_motion is internal_fit_gmr_motion
    assert retargeting_cache_dir is internal_retargeting_cache_dir


def test_gmr_velocity_limits_use_scalar_joint_names_not_muscle_actuators():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <freejoint name="root"/>
              <geom type="sphere" size="0.1"/>
              <body>
                <joint name="hip_flexion" type="hinge"/>
                <joint name="pelvis_slide" type="slide"/>
                <geom type="sphere" size="0.1"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor name="Ps_L1_VB_r" joint="hip_flexion"/>
          </actuator>
        </mujoco>
        """
    )

    limits = _scalar_joint_velocity_limits(model, 3 * np.pi)

    assert limits == {
        "hip_flexion": pytest.approx(3 * np.pi),
        "pelvis_slide": pytest.approx(3 * np.pi),
    }
    assert "root" not in limits
    assert "Ps_L1_VB_r" not in limits


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_gmr_velocity_limits_reject_invalid_values(value):
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body><joint name="hinge"/>
          <geom type="sphere" size="0.1"/>
        </body></worldbody></mujoco>
        """
    )

    with pytest.raises(ValueError, match="positive and finite"):
        _scalar_joint_velocity_limits(model, value)
