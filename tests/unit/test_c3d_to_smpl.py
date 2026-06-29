from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from musclemimic.web_viewer import c3d_to_smpl
from musclemimic.web_viewer.c3d import markers as c3d_markers
from musclemimic.web_viewer.c3d.asset_paths import (
    C3D_MODEL_PATH_KEY,
    HEAD_MARKER_CORR_PATH_KEY,
    MOSHPP_ASSETS_PATH_KEY,
    POSE_BODY_PRIOR_PATH_KEY,
    POSE_HAND_PRIOR_PATH_KEY,
    resolve_c3d_model_path,
    resolve_head_marker_corr_path,
    resolve_pose_body_prior_path,
    resolve_pose_hand_prior_path,
)


@pytest.fixture(autouse=True)
def isolate_musclemimic_path_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MUSCLEMIMIC_CONFIG_PATH", str(tmp_path / "missing_variables.yaml"))
    for key in (
        MOSHPP_ASSETS_PATH_KEY,
        C3D_MODEL_PATH_KEY,
        POSE_BODY_PRIOR_PATH_KEY,
        POSE_HAND_PRIOR_PATH_KEY,
        HEAD_MARKER_CORR_PATH_KEY,
    ):
        monkeypatch.delenv(key, raising=False)


def test_build_marker_layout_matches_moshpp_aliases() -> None:
    layout = c3d_to_smpl.build_marker_layout(
        ["LASI", "RASI", "LPSI", "RPSI", "LWRA", "LWRB", "C7"],
        wrist_markers_on_stick=True,
        surface_model_type="smplh",
    )

    assert layout.labels == ["C7", "LBWT", "LFWT", "LIWR", "LOWR", "RBWT", "RFWT"]
    assert layout.marker_vids["LFWT"] == 857
    assert layout.marker_vids["RBWT"] == 6544
    assert layout.marker_vids["LIWR"] == 2112
    assert layout.marker_type["LIWR"] == "wrist"
    assert layout.marker_type["LOWR"] == "wrist"
    assert layout.marker_type["LFWT"] == "body"
    assert layout.m2b_distance == {"body": 0.0095, "wrist": 0.039}


def test_build_marker_layout_supports_moshpp_smplx_vids() -> None:
    layout = c3d_to_smpl.build_marker_layout(
        ["C7", "CLAV", "LANK", "RANK", "LTOE", "RTOE"],
        surface_model_type="smplx",
    )

    assert layout.surface_model_type == "smplx"
    assert layout.marker_vids == {
        "C7": 3832,
        "CLAV": 5533,
        "LANK": 5882,
        "LTOE": 5787,
        "RANK": 8576,
        "RTOE": 8481,
    }


def test_build_marker_layout_includes_smplx_body_markers() -> None:
    layout = c3d_to_smpl.build_marker_layout(
        ["RBAC", "LSHN", "RSHN", "LMT5", "RMT5", "LTOE1", "RTOE1"],
        surface_model_type="smplx",
    )

    assert layout.marker_vids == {
        "LMT5": 5901,
        "LSHN": 3712,
        "LTOE": 5787,
        "RBAK": 6127,
        "RMT5": 8595,
        "RSHN": 6473,
        "RTOE": 8481,
    }


def test_prepare_marker_observations_collapses_duplicate_aliases() -> None:
    positions = np.array(
        [
            [
                [1.0, 0.0, 0.0],  # LASI -> LFWT
                [3.0, 0.0, 0.0],  # duplicate LFWT
                [10.0, 0.0, 0.0],  # unknown
                [np.nan, np.nan, np.nan],  # LWRB -> LOWR
                [5.0, 1.0, 1.0],  # duplicate LOWR
            ]
        ],
        dtype=np.float32,
    )

    prepared = c3d_to_smpl.prepare_marker_observations(
        positions,
        ["LASI", "LFWT", "UNKNOWN", "LWRB", "LOWR"],
    )

    assert prepared.labels == ["LFWT", "LOWR"]
    np.testing.assert_allclose(prepared.positions[0, 0], np.array([2.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(prepared.positions[0, 1], np.array([5.0, 1.0, 1.0], dtype=np.float32))
    assert prepared.unknown_labels == ["UNKNOWN"]


def test_pick_stagei_frames_random_strict_filters_by_availability() -> None:
    markers = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[np.nan, np.nan, np.nan], [2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    picks = c3d_to_smpl.pick_stagei_frames(markers, num_frames=2, seed=123, least_avail_markers=1.0, strict=True)
    assert picks.tolist() == [1, 3]

    with pytest.raises(ValueError, match="Not enough frames"):
        c3d_to_smpl.pick_stagei_frames(markers, num_frames=3, seed=123, least_avail_markers=1.0, strict=True)


def test_pick_stagei_frames_matches_moshpp_random_strict_trial02_order() -> None:
    markers = np.ones((200, 31, 3), dtype=np.float32)
    picks = c3d_to_smpl.pick_stagei_frames(markers, num_frames=12, seed=100, least_avail_markers=1.0, strict=True)
    assert picks.tolist() == [111, 52, 69, 96, 116, 99, 92, 104, 124, 164, 126, 167]


def test_pick_stagei_frames_does_not_reset_global_rng() -> None:
    markers = np.ones((20, 4, 3), dtype=np.float32)
    np.random.seed(12345)
    before = np.random.get_state()

    c3d_to_smpl.pick_stagei_frames(markers, num_frames=5, seed=100, least_avail_markers=1.0, strict=True)

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_stage_weight_formulas_match_moshpp_defaults() -> None:
    stage1 = c3d_to_smpl.compute_stagei_weights(num_markers=23, marker_types=["body", "wrist"], anneal_factor=0.5)
    assert stage1.data == pytest.approx((75.0 / 0.5) * (46.0 / 23.0))
    assert stage1.pose_body == pytest.approx(3.0 * 0.5)
    assert stage1.betas == pytest.approx(10.0 * 0.5)
    assert stage1.init_by_type["body"] == pytest.approx(300.0 * 0.5)
    assert stage1.surf == pytest.approx(10000.0)

    stage2 = c3d_to_smpl.compute_stageii_weights(num_observed_markers=18, num_total_markers=24)
    assert stage2.anneal_factor == pytest.approx(1.0 + ((24 - 18) / 24.0) * 2.5)
    assert stage2.data == pytest.approx(400.0 * (46.0 / 18.0))
    assert stage2.pose_body == pytest.approx(1.6 * stage2.anneal_factor)
    assert stage2.velo == pytest.approx(2.5)


def test_pose_freeze_hook_matches_moshpp_default_free_vars() -> None:
    torch = pytest.importorskip("torch")

    grad = torch.ones(1, 156)
    frozen = c3d_to_smpl._make_pose_freeze_hook(66, optimize_toes=False)(grad)

    assert torch.all(frozen[:, :30] == 1.0)
    assert torch.all(frozen[:, 30:36] == 0.0)
    assert torch.all(frozen[:, 36:66] == 1.0)
    assert torch.all(frozen[:, 66:] == 0.0)

    toes_enabled = c3d_to_smpl._make_pose_freeze_hook(66, optimize_toes=True)(grad)
    assert torch.all(toes_enabled[:, 30:36] == 1.0)
    assert torch.all(toes_enabled[:, 66:] == 0.0)


def test_load_stagei_head_marker_correlation_matches_moshpp_activation(tmp_path) -> None:
    corr_path = tmp_path / "ssm_head_marker_corr.npz"
    np.savez(
        corr_path,
        mrk_labels=np.asarray(["LFHD", "RFHD"]),
        corr=np.asarray([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32),
    )

    corr = c3d_to_smpl.load_stagei_head_marker_correlation(corr_path, ["C7", "LFHD", "RFHD"])

    assert corr is not None
    assert corr.labels == ("LFHD", "RFHD")
    assert corr.marker_indices.tolist() == [1, 2]
    np.testing.assert_allclose(corr.corr, np.asarray([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32))

    inactive = c3d_to_smpl.load_stagei_head_marker_correlation(corr_path, ["C7", "LFHD"])
    assert inactive is None


def test_moshpp_asset_paths_resolve_from_configured_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MUSCLEMIMIC_CONFIG_PATH", str(tmp_path / "missing_variables.yaml"))
    for key in (
        MOSHPP_ASSETS_PATH_KEY,
        C3D_MODEL_PATH_KEY,
        POSE_BODY_PRIOR_PATH_KEY,
        POSE_HAND_PRIOR_PATH_KEY,
        HEAD_MARKER_CORR_PATH_KEY,
    ):
        monkeypatch.delenv(key, raising=False)

    root = tmp_path / "moshpp_assets"
    c3d_model_root = tmp_path / "c3d_smplx"
    root.mkdir()
    c3d_model_root.mkdir()
    pose_body_prior = root / "pose_body_prior.pkl"
    pose_hand_prior = root / "pose_hand_prior.npz"
    head_marker_corr = root / "ssm_head_marker_corr.npz"
    pose_body_prior.write_bytes(b"prior")
    np.savez(pose_hand_prior, componentsl=np.zeros((1, 1)), componentsr=np.zeros((1, 1)))
    np.savez(head_marker_corr, mrk_labels=np.asarray(["LFHD"]), corr=np.eye(1, dtype=np.float32))

    monkeypatch.setenv(MOSHPP_ASSETS_PATH_KEY, str(root))
    monkeypatch.setenv(C3D_MODEL_PATH_KEY, str(c3d_model_root))

    assert resolve_c3d_model_path() == c3d_model_root
    assert resolve_pose_body_prior_path() == pose_body_prior
    assert resolve_pose_hand_prior_path() == pose_hand_prior
    assert resolve_head_marker_corr_path() == head_marker_corr
    assert c3d_to_smpl.resolve_stagei_head_marker_corr_path(
        None, smpl_model_path="unused", surface_model_type="smplx"
    ) == str(head_marker_corr)


def test_surface_marker_model_transforms_canonical_latent_markers() -> None:
    torch = pytest.importorskip("torch")

    marker_model = c3d_to_smpl.SurfaceMarkerModel(
        vids=np.array([0], dtype=np.int64),
        frame_vids=np.array([[0, 1, 2]], dtype=np.int64),
        initial_coeffs=np.array([[0.2, 0.3, 0.4]], dtype=np.float32),
        initial_latents=np.array([[0.2, 0.3, 0.4]], dtype=np.float32),
        desired_distances=np.array([0.4], dtype=np.float32),
        marker_types=("body",),
    )
    canonical_verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    posed_verts = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    markers_latent = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32)

    coeffs = marker_model.latents_to_coeffs(canonical_verts, markers_latent)
    reconstructed = marker_model.reconstruct(
        posed_verts,
        markers_latent=markers_latent,
        canonical_verts=canonical_verts,
    )

    np.testing.assert_allclose(coeffs.numpy(), np.array([[0.2, 0.3, 0.4]], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(reconstructed.numpy(), np.array([[-0.3, 0.2, 0.4]], dtype=np.float32), atol=1e-6)


def test_dynamic_frame_vids_use_nearest_non_collinear_vertices() -> None:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [0.0, 0.0, 0.03],
        ],
        dtype=np.float32,
    )
    marker = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    raw = c3d_to_smpl.SurfaceMarkerModel.compute_dynamic_frame_vids(verts, marker)

    assert raw.tolist() == [[0, 1, 2]]


def test_dynamic_frame_vids_exclude_smplx_eyeballs() -> None:
    eyeballs_path = Path("references/moshpp/support_data/smplx_eyeballs.npz")
    if not eyeballs_path.exists():
        pytest.skip("MoSh++ SMPL-X eyeball support data is not available")

    eyeball_vid = int(np.load(eyeballs_path)["eyeballs"][0])
    verts = np.zeros((10475, 3), dtype=np.float32)
    verts[:, 0] = np.arange(10475, dtype=np.float32)
    verts[eyeball_vid] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    verts[0] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    verts[1] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    marker = verts[eyeball_vid : eyeball_vid + 1].copy()

    frame_vids = c3d_to_smpl.SurfaceMarkerModel.compute_dynamic_frame_vids(verts, marker)

    assert eyeball_vid not in frame_vids[0].tolist()


def test_load_c3d_markers_handles_units_and_missing_points_without_axis_remap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_c3d = {
        "parameters": {
            "POINT": {
                "LABELS": {"value": ["A", "B"]},
                "UNITS": {"value": ["mm"]},
            }
        },
        "header": {"points": {"frame_rate": 120}},
        "data": {
            "points": np.array(
                [
                    [[1000.0], [0.0]],
                    [[2000.0], [0.0]],
                    [[3000.0], [0.0]],
                    [[0.0], [0.0]],
                ],
                dtype=np.float32,
            ),
            "meta_points": {
                "residuals": np.array([[[1.0], [-1.0]]], dtype=np.float32),
            },
        },
    }

    class FakeEzc3d:
        @staticmethod
        def c3d(_: str):
            return fake_c3d

    monkeypatch.setattr(c3d_markers, "ezc3d", FakeEzc3d)
    positions, labels, fps = c3d_to_smpl.load_c3d_markers("dummy.c3d")

    assert labels == ["A", "B"]
    assert fps == 120.0
    np.testing.assert_allclose(positions[0, 0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.isnan(positions[0, 1]).all()


def test_load_c3d_markers_uses_point_residual_row_when_meta_residuals_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_c3d = {
        "parameters": {
            "POINT": {
                "LABELS": {"value": ["A", "B"]},
                "UNITS": {"value": ["cm"]},
            }
        },
        "header": {"points": {"frame_rate": 100}},
        "data": {
            "points": np.array(
                [
                    [[100.0], [200.0]],
                    [[0.0], [300.0]],
                    [[0.0], [400.0]],
                    [[1.0], [-1.0]],
                ],
                dtype=np.float32,
            ),
        },
    }

    class FakeEzc3d:
        @staticmethod
        def c3d(_: str):
            return fake_c3d

    monkeypatch.setattr(c3d_markers, "ezc3d", FakeEzc3d)
    positions, labels, fps = c3d_to_smpl.load_c3d_markers("dummy.c3d")

    assert labels == ["A", "B"]
    assert fps == 100.0
    np.testing.assert_allclose(positions[0, 0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.isnan(positions[0, 1]).all()


def test_load_c3d_markers_reads_split_point_label_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_c3d = {
        "parameters": {
            "POINT": {
                "LABELS": {"value": ["A", "B"]},
                "LABELS2": {"value": ["C", "D"]},
                "UNITS": {"value": ["mm"]},
            }
        },
        "header": {"points": {"frame_rate": 120}},
        "data": {
            "points": np.array(
                [
                    [[1000.0], [2000.0], [3000.0], [4000.0]],
                    [[100.0], [200.0], [300.0], [400.0]],
                    [[10.0], [20.0], [30.0], [40.0]],
                    [[1.0], [1.0], [1.0], [1.0]],
                ],
                dtype=np.float32,
            ),
        },
    }

    class FakeEzc3d:
        @staticmethod
        def c3d(_: str):
            return fake_c3d

    monkeypatch.setattr(c3d_markers, "ezc3d", FakeEzc3d)
    positions, labels, fps = c3d_to_smpl.load_c3d_markers("dummy.c3d")

    assert labels == ["A", "B", "C", "D"]
    assert len(labels) == positions.shape[1]
    assert fps == 120.0


def test_load_c3d_markers_pads_missing_point_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_c3d = {
        "parameters": {
            "POINT": {
                "LABELS": {"value": ["A", "B"]},
                "UNITS": {"value": ["mm"]},
            }
        },
        "header": {"points": {"frame_rate": 120}},
        "data": {
            "points": np.array(
                [
                    [[1000.0], [2000.0], [3000.0]],
                    [[100.0], [200.0], [300.0]],
                    [[10.0], [20.0], [30.0]],
                    [[1.0], [1.0], [1.0]],
                ],
                dtype=np.float32,
            ),
        },
    }

    class FakeEzc3d:
        @staticmethod
        def c3d(_: str):
            return fake_c3d

    monkeypatch.setattr(c3d_markers, "ezc3d", FakeEzc3d)
    positions, labels, _fps = c3d_to_smpl.load_c3d_markers("dummy.c3d")

    assert labels == ["A", "B", "UNLABELED_3"]
    assert len(labels) == positions.shape[1]


def test_fit_smpl_to_c3d_cached_uses_existing_cache(tmp_path) -> None:
    cache_dir = tmp_path / ".smpl_cache"
    cache_dir.mkdir()
    suffix = c3d_to_smpl._cache_suffix_for_fit("smpl_dir", {})
    cache_path = cache_dir / f"sample_smpl_v{c3d_to_smpl.C3D_TO_SMPL_CACHE_VERSION}{suffix}.npz"
    np.savez(
        cache_path,
        pose_aa=np.ones((2, 72), dtype=np.float32),
        fullpose=np.ones((2, 156), dtype=np.float32),
        trans=np.ones((2, 3), dtype=np.float32),
        betas=np.arange(10, dtype=np.float32),
        fps=np.array(120.0, dtype=np.float32),
        gender=np.array("neutral"),
    )

    result = c3d_to_smpl.fit_smpl_to_c3d_cached("sample.c3d", "smpl_dir", cache_dir=str(cache_dir))

    assert result["pose_aa"].shape == (2, 72)
    assert result["fullpose"].shape == (2, 156)
    assert result["trans"].shape == (2, 3)
    np.testing.assert_array_equal(result["betas"], np.arange(10, dtype=np.float32))
    assert result["fps"] == 120.0
    assert result["gender"] == "neutral"


def test_fit_smpl_to_c3d_cached_clear_cache_refits_existing_cache(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / ".smpl_cache"
    cache_dir.mkdir()
    suffix = c3d_to_smpl._cache_suffix_for_fit("smpl_dir", {})
    cache_path = cache_dir / f"sample_smpl_v{c3d_to_smpl.C3D_TO_SMPL_CACHE_VERSION}{suffix}.npz"
    np.savez(
        cache_path,
        pose_aa=np.ones((2, 72), dtype=np.float32),
        trans=np.ones((2, 3), dtype=np.float32),
        betas=np.ones(10, dtype=np.float32),
        fps=np.array(120.0, dtype=np.float32),
        gender=np.array("neutral"),
    )

    calls: dict = {}

    def fit_smpl_to_c3d(c3d_path, smpl_model_path, **kwargs):
        calls["c3d_path"] = c3d_path
        calls["smpl_model_path"] = smpl_model_path
        calls["kwargs"] = kwargs
        return {
            "pose_aa": np.zeros((1, 72), dtype=np.float32),
            "trans": np.zeros((1, 3), dtype=np.float32),
            "betas": np.zeros(10, dtype=np.float32),
            "fps": 30.0,
            "gender": "male",
        }

    monkeypatch.setattr(c3d_to_smpl, "fit_smpl_to_c3d", fit_smpl_to_c3d)

    result = c3d_to_smpl.fit_smpl_to_c3d_cached(
        "sample.c3d",
        "smpl_dir",
        cache_dir=str(cache_dir),
        clear_cache=True,
    )

    assert calls == {"c3d_path": "sample.c3d", "smpl_model_path": "smpl_dir", "kwargs": {}}
    np.testing.assert_array_equal(result["pose_aa"], np.zeros((1, 72), dtype=np.float32))
    written = np.load(cache_path, allow_pickle=False)
    np.testing.assert_array_equal(written["pose_aa"], np.zeros((1, 72), dtype=np.float32))
    assert float(written["fps"]) == 30.0
    assert str(written["gender"]) == "male"


def test_mosh_gmm_pose_prior_matches_single_gaussian_cost(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    import pickle

    prior_path = tmp_path / "pose_body_prior.pkl"
    with prior_path.open("wb") as handle:
        pickle.dump(
            {
                "covars": np.eye(63, dtype=np.float64)[None],
                "means": np.zeros((1, 63), dtype=np.float64),
                "weights": np.ones(1, dtype=np.float64),
            },
            handle,
        )

    prior = c3d_to_smpl.MoshGmmPosePrior(prior_path)
    body_pose = torch.zeros(2, 63, dtype=torch.float32)

    # With one identity Gaussian and zero pose, the only remaining term is
    # -log(weight / const), once per frame.
    expected_per_frame = 0.5 * 63 * np.log(2.0 * np.pi)
    assert float(prior(body_pose)) == pytest.approx(2.0 * expected_per_frame, rel=1e-5)


def test_mosh_gmm_pose_prior_supports_batches_and_mixtures(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    import pickle

    prior_path = tmp_path / "pose_body_prior.pkl"
    with prior_path.open("wb") as handle:
        pickle.dump(
            {
                "covars": np.stack([np.eye(63, dtype=np.float64), np.eye(63, dtype=np.float64)]),
                "means": np.stack([np.zeros(63, dtype=np.float64), np.full(63, 10.0, dtype=np.float64)]),
                "weights": np.ones(2, dtype=np.float64),
            },
            handle,
        )

    prior = c3d_to_smpl.MoshGmmPosePrior(prior_path)
    body_pose = torch.zeros(3, 63, dtype=torch.float32)

    expected_per_frame = 0.5 * 63 * np.log(2.0 * np.pi)
    assert float(prior(body_pose)) == pytest.approx(3.0 * expected_per_frame, rel=1e-5)


def test_resolve_smplx_model_file_prefers_moshpp_gender_model_pkl(tmp_path) -> None:
    model_dir = tmp_path / "male"
    model_dir.mkdir()
    model_file = model_dir / "model.pkl"
    model_file.write_bytes(b"placeholder")

    resolved, ext = c3d_to_smpl._resolve_smplx_model_file(tmp_path, "male")

    assert resolved == model_file
    assert ext == "pkl"


def test_resolve_smplx_model_file_checks_smplx_subdirectory(tmp_path) -> None:
    model_dir = tmp_path / "smplx"
    model_dir.mkdir()
    model_file = model_dir / "SMPLX_MALE.npz"
    model_file.write_bytes(b"placeholder")

    resolved, ext = c3d_to_smpl._resolve_smplx_model_file(tmp_path, "male")

    assert resolved == model_file
    assert ext == "npz"


def test_point_to_triangle_distance_handles_face_and_edge_regions() -> None:
    torch = pytest.importorskip("torch")

    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    candidate_faces = torch.tensor([[[0, 1, 2]], [[0, 1, 2]]], dtype=torch.long)
    points = torch.tensor([[0.25, 0.25, 2.0], [2.0, 0.0, 0.0]], dtype=torch.float32)

    distances = c3d_to_smpl._point_to_triangle_distance_torch(points, verts, candidate_faces)

    assert distances.tolist() == pytest.approx([2.0, 1.0], rel=1e-6)


def test_jax_stagei_lbs_extraction_accepts_direct_smplh_style_model() -> None:
    from musclemimic.web_viewer.stagei_jax_solver import _extract_lbs_arrays

    class DirectLbsModel:
        v_template = np.zeros((4, 3), dtype=np.float32)
        shapedirs = np.zeros((4, 3, 16), dtype=np.float32)
        posedirs = np.zeros((18, 12), dtype=np.float32)
        J_regressor = np.zeros((3, 4), dtype=np.float32)
        parents = np.array([-1, 0, 1], dtype=np.int64)
        lbs_weights = np.zeros((4, 3), dtype=np.float32)
        pose_mean = np.arange(9, dtype=np.float32)

    lbs = _extract_lbs_arrays(DirectLbsModel())

    assert lbs["pose_dim"] == 9
    assert lbs["full_pose_dim"] == 9
    np.testing.assert_array_equal(lbs["parents"], np.array([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(lbs["pose_mean"], np.arange(9, dtype=np.float32))


def test_jax_stagei_lbs_extraction_requires_16_betas() -> None:
    from musclemimic.web_viewer.stagei_jax_solver import JaxStageIUnsupportedError, _extract_lbs_arrays

    class TenBetaLbsModel:
        v_template = np.zeros((4, 3), dtype=np.float32)
        shapedirs = np.zeros((4, 3, 10), dtype=np.float32)
        posedirs = np.zeros((18, 12), dtype=np.float32)
        J_regressor = np.zeros((3, 4), dtype=np.float32)
        parents = np.array([-1, 0, 1], dtype=np.int64)
        lbs_weights = np.zeros((4, 3), dtype=np.float32)

    with pytest.raises(JaxStageIUnsupportedError, match="16 betas"):
        _extract_lbs_arrays(TenBetaLbsModel())


def test_fit_smpl_to_c3d_cached_suffix_disambiguates_stagei_options(tmp_path, monkeypatch) -> None:
    """Cache filenames must distinguish optimize_toes/pose-prior combos.

    Otherwise flipping a flag silently reuses a cache produced under the
    other configuration and we ship the wrong fit.
    """

    monkeypatch.setattr(
        c3d_to_smpl,
        "fit_smpl_to_c3d",
        lambda *_args, **_kwargs: {
            "pose_aa": np.zeros((1, 72), dtype=np.float32),
            "trans": np.zeros((1, 3), dtype=np.float32),
            "betas": np.zeros(10, dtype=np.float32),
            "fps": 30.0,
            "gender": "neutral",
        },
    )

    prior_path = tmp_path / "pose_body_prior.pkl"
    prior_path.write_bytes(b"prior")
    corr_path = tmp_path / "ssm_head_marker_corr.npz"
    np.savez(corr_path, mrk_labels=np.asarray(["LFHD"]), corr=np.eye(1, dtype=np.float32))

    base_kwargs = {"c3d_path": "sample.c3d", "smpl_model_path": "smpl_dir", "cache_dir": str(tmp_path)}

    c3d_to_smpl.fit_smpl_to_c3d_cached(**base_kwargs)
    c3d_to_smpl.fit_smpl_to_c3d_cached(**base_kwargs, optimize_toes=True)
    c3d_to_smpl.fit_smpl_to_c3d_cached(**base_kwargs, pose_body_prior_path=str(prior_path))
    c3d_to_smpl.fit_smpl_to_c3d_cached(**base_kwargs, head_marker_corr_path=str(corr_path))
    c3d_to_smpl.fit_smpl_to_c3d_cached(
        **base_kwargs,
        optimize_toes=True,
        pose_body_prior_path=str(prior_path),
    )

    written = sorted(p.name for p in tmp_path.glob("sample_smpl_v*.npz"))
    version = c3d_to_smpl.C3D_TO_SMPL_CACHE_VERSION
    expected = [
        f"sample_smpl_v{version}{c3d_to_smpl._cache_suffix_for_fit('smpl_dir', {})}.npz",
        f"sample_smpl_v{version}{c3d_to_smpl._cache_suffix_for_fit('smpl_dir', {'optimize_toes': True})}.npz",
        (
            f"sample_smpl_v{version}"
            f"{c3d_to_smpl._cache_suffix_for_fit('smpl_dir', {'optimize_toes': True, 'pose_body_prior_path': str(prior_path)})}.npz"
        ),
        (
            f"sample_smpl_v{version}"
            f"{c3d_to_smpl._cache_suffix_for_fit('smpl_dir', {'pose_body_prior_path': str(prior_path)})}.npz"
        ),
        (
            f"sample_smpl_v{version}"
            f"{c3d_to_smpl._cache_suffix_for_fit('smpl_dir', {'head_marker_corr_path': str(corr_path)})}.npz"
        ),
    ]
    assert written == sorted(expected)


def test_save_motion_data_as_amass_smplh_npz_expands_pose_to_156(tmp_path) -> None:
    output_path = tmp_path / "motion_smplh.npz"
    source = {
        "pose_aa": np.ones((3, 72), dtype=np.float32),
        "trans": np.arange(9, dtype=np.float32).reshape(3, 3),
        "betas": np.arange(16, dtype=np.float32),
        "fps": 30.0,
        "gender": "neutral",
    }

    written = c3d_to_smpl.save_motion_data_as_amass_smplh_npz(source, output_path)
    data = np.load(written, allow_pickle=True)

    assert written == output_path
    assert data["poses"].shape == (3, 156)
    np.testing.assert_array_equal(data["poses"][:, :72], np.ones((3, 72), dtype=np.float32))
    np.testing.assert_array_equal(data["poses"][:, 72:], np.zeros((3, 84), dtype=np.float32))
    np.testing.assert_array_equal(data["trans"], source["trans"])
    np.testing.assert_array_equal(data["betas"], source["betas"])
    assert float(data["mocap_framerate"]) == 30.0
    assert str(data["gender"]) == "neutral"
