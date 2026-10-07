from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from loco_mujoco.core.terminal_state_handler.base import TerminalStateHandler
from musclemimic.core.terminal_state_handler.enhanced_fullbody import (
    EnhancedFullBodyTerminalStateHandler,
    MeanRelativeSiteDeviationTerminalStateHandler,
    MeanRelativeSiteDeviationWithRootTerminalStateHandler,
    MeanSiteDeviationTerminalStateHandler,
)

# Site ordering taken from the real MyoFullBody configuration and trajectory output.
ENV_SITE_ORDER = [
    "pelvis_mimic",
    "upper_body_mimic",
    "head_mimic",
    "left_shoulder_mimic",
    "left_elbow_mimic",
    "left_hand_mimic",
    "right_shoulder_mimic",
    "right_elbow_mimic",
    "right_hand_mimic",
    "left_hip_mimic",
    "left_knee_mimic",
    "left_ankle_mimic",
    "left_toes_mimic",
    "right_hip_mimic",
    "right_knee_mimic",
    "right_ankle_mimic",
    "right_toes_mimic",
]

TRAJ_SITE_ORDER = [
    "upper_body_mimic",
    "head_mimic",
    "right_shoulder_mimic",
    "right_elbow_mimic",
    "right_hand_mimic",
    "left_shoulder_mimic",
    "left_elbow_mimic",
    "left_hand_mimic",
    "pelvis_mimic",
    "right_hip_mimic",
    "right_knee_mimic",
    "right_ankle_mimic",
    "right_toes_mimic",
    "left_hip_mimic",
    "left_knee_mimic",
    "left_ankle_mimic",
    "left_toes_mimic",
]

SITE_NAME_TO_ID = {name: idx for idx, name in enumerate(ENV_SITE_ORDER)}


class FakeSiteMapper:
    """Mimics MyoFullBodySiteMapper using real trajectory ordering."""

    def __init__(self):
        self.requires_mapping = True
        self._traj_index = {SITE_NAME_TO_ID[name]: idx for idx, name in enumerate(TRAJ_SITE_ORDER)}

    def model_ids_to_traj_indices(self, model_site_ids):
        ids = np.asarray(model_site_ids, dtype=int)
        missing = [mid for mid in ids if mid not in self._traj_index]
        if missing:
            raise ValueError(f"Missing ids in mapper: {missing}")
        return np.array([self._traj_index[int(mid)] for mid in ids], dtype=int)


class FakeGoal:
    def __init__(self, site_mapper: FakeSiteMapper):
        self._rel_site_ids = np.array([SITE_NAME_TO_ID[name] for name in ENV_SITE_ORDER])
        self._site_bodyid = np.arange(len(SITE_NAME_TO_ID))
        self._site_mapper = site_mapper
        self._info_props = {"sites_for_mimic": ENV_SITE_ORDER}


class FakeTrajectoryHandler:
    def __init__(self, ref_data):
        self._ref_data = ref_data

    def get_current_traj_data(self, carry, backend):
        return self._ref_data


class FakeTrajectoryHandlerWithInit(FakeTrajectoryHandler):
    def __init__(self, ref_data, init_data):
        super().__init__(ref_data)
        self._init_data = init_data

    def get_init_traj_data(self, carry, backend):
        return self._init_data


class FakeModel:
    def __init__(self, site_name_to_id: dict[str, int]):
        self._site_name_to_id = site_name_to_id
        self.body_rootid = np.zeros(len(site_name_to_id), dtype=int)
        self.site_bodyid = np.arange(len(site_name_to_id))

    def site(self, name: str):
        return SimpleNamespace(id=self._site_name_to_id[name])


class FakeEnv:
    def __init__(self, ref_data, site_mapper: FakeSiteMapper):
        self._goal = FakeGoal(site_mapper)
        self.th = FakeTrajectoryHandler(ref_data)
        self.model = FakeModel(SITE_NAME_TO_ID)
        self._model = SimpleNamespace()  # Height handler expects this attribute
        self._get_all_info_properties = lambda: {
            "root_height_healthy_range": [0.5, 2.0],
            "root_free_joint_xml_name": "root",
        }


class EnvWithoutTrajectory:
    """Minimal env that triggers early-return guards."""

    def __init__(self, site_mapper: FakeSiteMapper):
        self._goal = FakeGoal(site_mapper)
        self.th = None
        self.model = FakeModel(SITE_NAME_TO_ID)
        self._model = SimpleNamespace()
        self._get_all_info_properties = lambda: {
            "root_height_healthy_range": [0.5, 2.0],
            "root_free_joint_xml_name": "root",
        }


def _positions_in_model_order(traj_positions: np.ndarray) -> np.ndarray:
    positions_by_name = {name: traj_positions[idx] for idx, name in enumerate(TRAJ_SITE_ORDER)}
    return np.array([positions_by_name[name] for name in ENV_SITE_ORDER])


def make_ref_data():
    traj_positions = np.array([[idx, 0.0, 0.0] for idx in range(len(TRAJ_SITE_ORDER))], dtype=float)
    n_sites = len(TRAJ_SITE_ORDER)
    return SimpleNamespace(
        site_xpos=traj_positions,
        site_xmat=np.tile(np.eye(3).reshape(1, 9), (n_sites, 1)),
        cvel=np.zeros((len(ENV_SITE_ORDER), 6)),
        xpos=np.zeros((len(ENV_SITE_ORDER), 3)),
        subtree_com=np.zeros((len(ENV_SITE_ORDER), 3)),
    )


def make_state_from_traj(traj_positions, *, translation=None, backend=np, height=1.0):
    positions = backend.asarray(_positions_in_model_order(np.asarray(traj_positions)))
    if translation is not None:
        positions = positions + backend.asarray(translation)
    n_sites = len(ENV_SITE_ORDER)
    eye = backend.eye(3).reshape(1, 9)
    return SimpleNamespace(
        site_xpos=backend.asarray(positions),
        site_xmat=backend.tile(eye, (n_sites, 1)),
        cvel=backend.zeros((len(ENV_SITE_ORDER), 6)),
        xpos=backend.zeros((len(ENV_SITE_ORDER), 3)),
        subtree_com=backend.zeros((len(ENV_SITE_ORDER), 3)),
        qpos=backend.array([0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0]),
    )


def offset_site(traj_positions: np.ndarray, site_name: str, delta: np.ndarray) -> np.ndarray:
    updated = np.array(traj_positions, copy=True)
    idx = TRAJ_SITE_ORDER.index(site_name)
    updated[idx] = updated[idx] + delta
    return updated


def set_threshold(carry, value: float) -> None:
    carry.termination_threshold = jnp.asarray(value, dtype=jnp.float32)


@pytest.fixture
def site_mapper():
    return FakeSiteMapper()


@pytest.fixture
def ref_data():
    return make_ref_data()


@pytest.fixture
def env(ref_data, site_mapper):
    return FakeEnv(ref_data, site_mapper)


@pytest.fixture
def carry():
    return SimpleNamespace(termination_threshold=jnp.asarray(0.3, dtype=jnp.float32))


def test_handlers_registered():
    assert "EnhancedFullBodyTerminalStateHandler" in TerminalStateHandler.registered
    assert "MeanSiteDeviationTerminalStateHandler" in TerminalStateHandler.registered
    assert "MeanRelativeSiteDeviationTerminalStateHandler" in TerminalStateHandler.registered
    assert "MeanRelativeSiteDeviationWithRootTerminalStateHandler" in TerminalStateHandler.registered


def test_mean_site_deviation_uses_myo_mapping(env, ref_data, carry):
    handler = MeanSiteDeviationTerminalStateHandler(env, mean_site_deviation_threshold=0.2, enable_site_check=True)
    set_threshold(carry, 0.2)
    state = make_state_from_traj(ref_data.site_xpos)

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal)


@patch("musclemimic.core.terminal_state_handler.enhanced_fullbody.mj_jntname2qposid", return_value=np.arange(7))
def test_mean_site_deviation_xy_offset_alignment(mock_jnt, site_mapper, carry):
    base_ref = make_ref_data()
    offset = np.array([5.0, 3.0, 0.0])
    ref_data = SimpleNamespace(
        site_xpos=base_ref.site_xpos + offset,
        site_xmat=base_ref.site_xmat,
        cvel=base_ref.cvel,
        xpos=base_ref.xpos,
        subtree_com=base_ref.subtree_com,
    )
    init_data = SimpleNamespace(qpos=np.array([5.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

    env = FakeEnv(ref_data, site_mapper)
    env.th = FakeTrajectoryHandlerWithInit(ref_data, init_data)
    handler = MeanSiteDeviationTerminalStateHandler(env, mean_site_deviation_threshold=0.1, enable_site_check=True)
    set_threshold(carry, 0.1)
    state = make_state_from_traj(base_ref.site_xpos)

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal)


def test_mean_site_deviation_flags_translation(env, ref_data, carry):
    handler = MeanSiteDeviationTerminalStateHandler(env, mean_site_deviation_threshold=0.25, enable_site_check=True)
    set_threshold(carry, 0.25)
    state = make_state_from_traj(ref_data.site_xpos, translation=np.array([0.4, 0.0, 0.0]))

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert bool(is_terminal)


def test_mean_site_deviation_reduction_controls_outlier_sensitivity(env, ref_data, carry):
    moved = offset_site(ref_data.site_xpos, "right_hand_mimic", np.array([0.4, 0.0, 0.0]))
    state = make_state_from_traj(moved)
    set_threshold(carry, 0.1)
    mean_handler = MeanSiteDeviationTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.1,
        reduction="mean",
    )
    max_handler = MeanSiteDeviationTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.1,
        reduction="max",
    )
    mean_info = {}
    max_info = {}

    mean_terminal, _ = mean_handler.is_absorbing(env, np.array([]), mean_info, state, carry)
    max_terminal, _ = max_handler.is_absorbing(env, np.array([]), max_info, state, carry)

    assert not bool(mean_terminal)
    assert bool(max_terminal)
    assert not bool(mean_info["termination_site_position"])
    assert bool(max_info["termination_site_position"])

    excluded_handler = MeanSiteDeviationTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.1,
        reduction="max",
        exclude_sites=["right_hand_mimic"],
    )
    excluded_terminal, _ = excluded_handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(excluded_terminal)


def test_mean_site_deviation_minimum_steps_has_cpu_jax_and_jit_parity(env, ref_data):
    handler = MeanSiteDeviationTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.1,
        minimum_episode_steps=3,
    )
    state_np = make_state_from_traj(
        ref_data.site_xpos,
        translation=np.array([0.4, 0.0, 0.0]),
    )

    cpu_results = []
    for episode_step in (2, 3):
        info = {}
        terminal, _ = handler.is_absorbing(
            env,
            np.array([]),
            info,
            state_np,
            SimpleNamespace(
                termination_threshold=np.asarray(0.1, dtype=np.float32),
                cur_step_in_episode=np.asarray(episode_step, dtype=np.int32),
            ),
        )
        cpu_results.append((bool(terminal), bool(info["termination_site_position"])))

    state_jax = make_state_from_traj(
        ref_data.site_xpos,
        translation=jnp.array([0.4, 0.0, 0.0]),
        backend=jnp,
    )

    def evaluate_jax(episode_step):
        info = {}
        terminal, _ = handler.mjx_is_absorbing(
            env,
            jnp.array([]),
            info,
            state_jax,
            SimpleNamespace(
                termination_threshold=jnp.asarray(0.1, dtype=jnp.float32),
                cur_step_in_episode=episode_step,
            ),
        )
        return terminal, info["termination_site_position"]

    eager_results = [tuple(bool(value) for value in evaluate_jax(step)) for step in (2, 3)]
    compiled_evaluate = jax.jit(evaluate_jax)
    jit_results = [tuple(bool(value) for value in compiled_evaluate(step)) for step in (2, 3)]

    assert cpu_results == eager_results == jit_results == [(False, False), (True, True)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reduction": "median"}, "reduction must be one of"),
        ({"minimum_episode_steps": -1}, "minimum_episode_steps must be a nonnegative integer"),
        ({"minimum_episode_steps": 1.5}, "minimum_episode_steps must be a nonnegative integer"),
    ],
)
def test_mean_site_deviation_rejects_invalid_gate_configuration(env, kwargs, message):
    with pytest.raises(ValueError, match=message):
        MeanSiteDeviationTerminalStateHandler(env, **kwargs)


def test_mean_site_deviation_uses_carry_threshold(env, ref_data, carry):
    """Carry threshold should override handler default."""
    handler = MeanSiteDeviationTerminalStateHandler(env, mean_site_deviation_threshold=1.0, enable_site_check=True)
    set_threshold(carry, 0.05)
    state = make_state_from_traj(ref_data.site_xpos, translation=np.array([0.4, 0.0, 0.0]))

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert bool(is_terminal)


def test_mean_site_deviation_can_be_disabled(env, ref_data, carry):
    handler = MeanSiteDeviationTerminalStateHandler(env, mean_site_deviation_threshold=0.05, enable_site_check=False)
    set_threshold(carry, 0.05)
    state = make_state_from_traj(ref_data.site_xpos, translation=np.array([1.0, 0.0, 0.0]))

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal)


def test_mean_site_deviation_exclude_sites(env, ref_data, carry):
    """Excluding sites should remove them from mean calculation."""
    moved = offset_site(ref_data.site_xpos, "right_hand_mimic", np.array([1.0, 0.0, 0.0]))
    state = make_state_from_traj(moved)

    handler_no_excl = MeanSiteDeviationTerminalStateHandler(
        env, mean_site_deviation_threshold=0.05, enable_site_check=True
    )
    set_threshold(carry, 0.05)
    is_terminal, _ = handler_no_excl.is_absorbing(env, np.array([]), {}, state, carry)
    assert bool(is_terminal)

    handler_excl = MeanSiteDeviationTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.05,
        enable_site_check=True,
        exclude_sites=["right_hand_mimic"],
    )
    set_threshold(carry, 0.05)
    is_terminal, _ = handler_excl.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal)


def test_mean_site_deviation_exclude_multiple_sites(env, ref_data, carry):
    """Multiple excluded sites should all be removed from calculation."""
    handler = MeanSiteDeviationTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.1,
        enable_site_check=True,
        exclude_sites=["left_ankle_mimic", "right_ankle_mimic", "left_toes_mimic", "right_toes_mimic"],
    )
    assert handler._n_included == len(ENV_SITE_ORDER) - 4


def test_relative_handler_translation_invariant(env, ref_data, carry):
    handler = MeanRelativeSiteDeviationTerminalStateHandler(
        env, mean_site_deviation_threshold=0.2, enable_site_check=True
    )
    set_threshold(carry, 0.2)
    state = make_state_from_traj(ref_data.site_xpos, translation=np.array([0.5, 0.1, 0.0]))

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal)


def test_relative_handler_detects_configuration_change(env, ref_data, carry):
    # 17 sites total, but relative handler excludes main site (pelvis) -> 16 sites
    # Mean deviation = 0.5 / 16 ≈ 0.031, so threshold must be strictly below that
    handler = MeanRelativeSiteDeviationTerminalStateHandler(
        env, mean_site_deviation_threshold=0.03, enable_site_check=True
    )
    set_threshold(carry, 0.03)
    moved = offset_site(ref_data.site_xpos, "right_hand_mimic", np.array([0.5, 0.0, 0.0]))
    state = make_state_from_traj(moved)

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert bool(is_terminal)


def test_relative_handler_uses_carry_threshold(env, ref_data, carry):
    """Carry threshold should override handler default for relative handler."""
    handler = MeanRelativeSiteDeviationTerminalStateHandler(
        env, mean_site_deviation_threshold=1.0, enable_site_check=True
    )
    set_threshold(carry, 0.01)
    moved = offset_site(ref_data.site_xpos, "right_hand_mimic", np.array([0.5, 0.0, 0.0]))
    state = make_state_from_traj(moved)

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert bool(is_terminal)


def test_relative_with_root_uses_carry_threshold(env, ref_data, carry):
    """Carry threshold should override handler default for relative+root handler."""
    handler = MeanRelativeSiteDeviationWithRootTerminalStateHandler(
        env,
        mean_site_deviation_threshold=1.0,
        root_deviation_threshold=1.0,
        root_site="pelvis_mimic",
        enable_site_check=True,
    )
    set_threshold(carry, 0.01)
    moved = offset_site(ref_data.site_xpos, "right_hand_mimic", np.array([0.5, 0.0, 0.0]))
    state = make_state_from_traj(moved)

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert bool(is_terminal)


def test_relative_handler_exclude_sites(env, ref_data, carry):
    """Relative handler should support exclude_sites with shifted indices."""
    moved = offset_site(ref_data.site_xpos, "head_mimic", np.array([0.5, 0.0, 0.0]))
    state = make_state_from_traj(moved)

    handler_no_excl = MeanRelativeSiteDeviationTerminalStateHandler(
        env, mean_site_deviation_threshold=0.01, enable_site_check=True
    )
    set_threshold(carry, 0.01)
    is_terminal, _ = handler_no_excl.is_absorbing(env, np.array([]), {}, state, carry)
    assert bool(is_terminal)

    handler_excl = MeanRelativeSiteDeviationTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.01,
        enable_site_check=True,
        exclude_sites=["head_mimic"],
    )
    set_threshold(carry, 0.01)
    is_terminal, _ = handler_excl.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal)


def test_relative_handler_pelvis_exclude_is_noop(env, ref_data, carry):
    """Excluding pelvis_mimic in relative handler should warn but not error."""
    handler = MeanRelativeSiteDeviationTerminalStateHandler(
        env, mean_site_deviation_threshold=0.2, enable_site_check=True, exclude_sites=["pelvis_mimic"]
    )
    assert handler._n_included == len(ENV_SITE_ORDER) - 1


def test_relative_handler_jax_path(env, ref_data, carry):
    handler = MeanRelativeSiteDeviationTerminalStateHandler(
        env, mean_site_deviation_threshold=0.2, enable_site_check=True
    )
    set_threshold(carry, 0.2)
    state = make_state_from_traj(ref_data.site_xpos, translation=jnp.array([0.25, 0.0, 0.0]), backend=jnp)

    is_terminal, _ = handler.mjx_is_absorbing(env, jnp.array([]), {}, state, carry)
    assert not bool(is_terminal)


def test_exclude_sites_jax_backend(env, ref_data, carry):
    """Exclusion should work with JAX backend."""
    moved = offset_site(ref_data.site_xpos, "right_hand_mimic", np.array([1.0, 0.0, 0.0]))
    state = make_state_from_traj(moved, backend=jnp)

    handler = MeanSiteDeviationTerminalStateHandler(
        env, mean_site_deviation_threshold=0.05, enable_site_check=True, exclude_sites=["right_hand_mimic"]
    )
    set_threshold(carry, 0.05)
    is_terminal, _ = handler.mjx_is_absorbing(env, jnp.array([]), {}, state, carry)
    assert not bool(is_terminal)


def test_exclude_all_sites_returns_false(env, ref_data, carry):
    """Excluding all sites should return False (never terminate), not NaN."""
    handler = MeanSiteDeviationTerminalStateHandler(
        env, mean_site_deviation_threshold=0.01, enable_site_check=True, exclude_sites=ENV_SITE_ORDER
    )
    set_threshold(carry, 0.01)
    assert handler._n_included == 0

    state = make_state_from_traj(ref_data.site_xpos, translation=np.array([10.0, 0.0, 0.0]))
    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal)


@patch("loco_mujoco.core.terminal_state_handler.height.mj_jntname2qposid", return_value=np.arange(7))
def test_enhanced_handler_tracks_root_and_ankles(mock_jnt, env, ref_data, carry):
    handler = EnhancedFullBodyTerminalStateHandler(
        env,
        ankle_deviation=0.2,
        root_deviation=0.1,
        ankle_sites=["left_ankle_mimic", "right_ankle_mimic"],
        root_site="pelvis_mimic",
        site_deviation_mode="max",
        enable_site_check=True,
    )
    shifted = offset_site(ref_data.site_xpos, "pelvis_mimic", np.array([0.4, 0.0, 0.0]))
    state = make_state_from_traj(shifted, backend=jnp, height=1.0)

    is_terminal, _ = handler.mjx_is_absorbing(env, jnp.array([]), {}, state, carry)
    assert bool(is_terminal)


@patch("loco_mujoco.core.terminal_state_handler.height.mj_jntname2qposid", return_value=np.arange(7))
def test_enhanced_handler_height_violation(mock_jnt, env, ref_data, carry):
    handler = EnhancedFullBodyTerminalStateHandler(env, root_height_healthy_range=[0.5, 1.5], enable_site_check=False)
    state = make_state_from_traj(ref_data.site_xpos, backend=jnp, height=0.1)

    is_terminal, _ = handler.mjx_is_absorbing(env, jnp.array([]), {}, state, carry)
    assert bool(is_terminal)


@patch("loco_mujoco.core.terminal_state_handler.height.mj_jntname2qposid", return_value=np.arange(7))
def test_enhanced_handler_invalid_mode_raises(mock_jnt, env):
    with pytest.raises(ValueError, match="site_deviation_mode"):
        EnhancedFullBodyTerminalStateHandler(env, site_deviation_mode="invalid")


def test_handlers_return_false_without_trajectory(site_mapper, ref_data, carry):
    env = EnvWithoutTrajectory(site_mapper)
    handler = MeanSiteDeviationTerminalStateHandler(env, mean_site_deviation_threshold=0.1, enable_site_check=True)
    set_threshold(carry, 0.1)
    state = make_state_from_traj(ref_data.site_xpos, backend=jnp)

    is_terminal, _ = handler.mjx_is_absorbing(env, jnp.array([]), {}, state, carry)
    assert not bool(is_terminal)


# ---- MeanRelativeSiteDeviationWithRootTerminalStateHandler tests ----


def test_relative_with_root_perfect_tracking(env, ref_data, carry):
    """Perfect tracking should not terminate when root + relative checks enabled."""
    handler = MeanRelativeSiteDeviationWithRootTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.3,
        root_deviation_threshold=0.3,
        root_site="pelvis_mimic",
        enable_site_check=True,
    )
    set_threshold(carry, 0.3)
    state = make_state_from_traj(ref_data.site_xpos)

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal), "Perfect tracking should not trigger termination"


def test_relative_with_root_uses_mapper_updated_after_construction(env, ref_data, carry):
    """Root lookup should use the sparse trajectory order attached by the goal."""
    handler = MeanRelativeSiteDeviationWithRootTerminalStateHandler(
        env,
        mean_site_deviation_threshold=1.0,
        root_deviation_threshold=0.3,
        root_site="pelvis_mimic",
        enable_site_check=True,
    )
    model_order_ref = SimpleNamespace(**vars(ref_data))
    model_order_ref.site_xpos = _positions_in_model_order(ref_data.site_xpos)
    env.th._ref_data = model_order_ref
    env._goal._site_mapper._traj_index = {idx: idx for idx in range(len(ENV_SITE_ORDER))}
    state = make_state_from_traj(ref_data.site_xpos)

    root_violation = handler._check_root_deviation(env, state, carry, np)

    assert not bool(root_violation)


def test_relative_with_root_translation_triggers_termination(env, ref_data, carry):
    """Pure translation should terminate via root check while pose is preserved."""
    handler = MeanRelativeSiteDeviationWithRootTerminalStateHandler(
        env,
        mean_site_deviation_threshold=1.0,  # High to isolate root check
        root_deviation_threshold=0.2,
        root_site="pelvis_mimic",
        enable_site_check=True,
    )
    set_threshold(carry, 1.0)
    state = make_state_from_traj(ref_data.site_xpos, translation=np.array([0.4, 0.3, 0.0]))

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert bool(is_terminal), "Root deviation should trigger termination even when pose is unchanged"


@patch("musclemimic.core.terminal_state_handler.enhanced_fullbody.mj_jntname2qposid", return_value=np.arange(7))
def test_relative_with_root_xy_offset_alignment(mock_jnt, site_mapper, carry):
    """Root deviation should not trigger when reference is offset by init XY."""
    base_ref = make_ref_data()
    offset = np.array([5.0, 3.0, 0.0])
    ref_data = SimpleNamespace(
        site_xpos=base_ref.site_xpos + offset,
        site_xmat=base_ref.site_xmat,
        cvel=base_ref.cvel,
        xpos=base_ref.xpos,
        subtree_com=base_ref.subtree_com,
    )
    init_data = SimpleNamespace(qpos=np.array([5.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

    env = FakeEnv(ref_data, site_mapper)
    env.th = FakeTrajectoryHandlerWithInit(ref_data, init_data)
    handler = MeanRelativeSiteDeviationWithRootTerminalStateHandler(
        env,
        mean_site_deviation_threshold=1.0,
        root_deviation_threshold=0.1,
        root_site="pelvis_mimic",
        enable_site_check=True,
    )
    set_threshold(carry, 1.0)
    state = make_state_from_traj(base_ref.site_xpos)

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal), "Root deviation should respect init XY offset"


def test_relative_with_root_configuration_change(env, ref_data, carry):
    """Body configuration changes should trigger relative check independent of root position."""
    handler = MeanRelativeSiteDeviationWithRootTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.03,  # Strict threshold for config change
        root_deviation_threshold=1.0,  # Relax root to isolate relative check
        root_site="pelvis_mimic",
        enable_site_check=True,
    )
    set_threshold(carry, 0.03)
    moved = offset_site(ref_data.site_xpos, "right_hand_mimic", np.array([0.5, 0.0, 0.0]))
    state = make_state_from_traj(moved)

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert bool(is_terminal), "Relative pose change should trigger termination even with aligned root"


def test_relative_with_root_disabled_site_check(env, ref_data, carry):
    """When site checks are disabled, handler should not terminate on pose or root changes."""
    handler = MeanRelativeSiteDeviationWithRootTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.2,
        root_deviation_threshold=0.2,
        root_site="pelvis_mimic",
        enable_site_check=False,
    )
    set_threshold(carry, 0.2)
    state = make_state_from_traj(ref_data.site_xpos, translation=np.array([0.5, 0.0, 0.0]))

    is_terminal, _ = handler.is_absorbing(env, np.array([]), {}, state, carry)
    assert not bool(is_terminal), "Disabling site checks should bypass both root and relative checks"


def test_relative_with_root_jax_path(env, ref_data, carry):
    """MJX path: perfect tracking should not terminate with root + relative checks."""
    handler = MeanRelativeSiteDeviationWithRootTerminalStateHandler(
        env,
        mean_site_deviation_threshold=0.3,
        root_deviation_threshold=0.3,
        root_site="pelvis_mimic",
        enable_site_check=True,
    )
    set_threshold(carry, 0.3)
    state = make_state_from_traj(ref_data.site_xpos, backend=jnp)

    is_terminal, _ = handler.mjx_is_absorbing(env, jnp.array([]), {}, state, carry)
    assert not bool(is_terminal), "Perfect tracking on MJX backend should not terminate"


def test_relative_with_root_jax_translation_triggers(env, ref_data, carry):
    """MJX path: translation should trigger root deviation."""
    handler = MeanRelativeSiteDeviationWithRootTerminalStateHandler(
        env,
        mean_site_deviation_threshold=1.0,
        root_deviation_threshold=0.1,
        root_site="pelvis_mimic",
        enable_site_check=True,
    )
    set_threshold(carry, 1.0)
    state = make_state_from_traj(ref_data.site_xpos, translation=jnp.array([0.3, 0.2, 0.0]), backend=jnp)

    is_terminal, _ = handler.mjx_is_absorbing(env, jnp.array([]), {}, state, carry)
    assert bool(is_terminal), "Root translation should trigger termination on MJX backend"
