from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .asset_paths import resolve_pose_hand_prior_path


def _resolve_smplx_model_file(model_path: str | Path, gender: str) -> tuple[Path, str]:
    path = Path(model_path)
    gender = gender.lower()
    if path.is_file():
        ext = path.suffix.lower().lstrip(".")
        if ext not in {"pkl", "npz"}:
            raise ValueError(f"Unsupported SMPL-X model file extension: {path}")
        return path, ext

    roots = [path]
    if path.name != "smplx" and (path / "smplx").exists():
        roots.append(path / "smplx")
    candidates: list[Path] = []
    for root in roots:
        if gender in {"male", "female", "neutral"}:
            candidates.extend(
                [
                    root / gender / "model.pkl",
                    root / f"SMPLX_{gender.upper()}.pkl",
                    root / f"SMPLX_{gender.upper()}.npz",
                ]
            )
        candidates.extend([root / "model.pkl", root / "SMPLX_NEUTRAL.pkl", root / "SMPLX_NEUTRAL.npz"])
    for candidate in candidates:
        if candidate.exists():
            ext = candidate.suffix.lower().lstrip(".")
            return candidate, ext
    raise FileNotFoundError(
        f"No SMPL-X model found under {path}. Expected {gender}/model.pkl, "
        f"SMPLX_{gender.upper()}.pkl, or SMPLX_{gender.upper()}.npz."
    )


class _FakeChumpyCh:
    """Enough of chumpy.ch.Ch to unpickle MoSh++ SMPL-X model.pkl files."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


def _load_moshpp_smplx_pkl(model_pkl_path: str | Path) -> dict:
    import sys
    import types

    inserted_chumpy = "chumpy" not in sys.modules
    inserted_chumpy_ch = "chumpy.ch" not in sys.modules
    if inserted_chumpy:
        chumpy_module = types.ModuleType("chumpy")
        sys.modules["chumpy"] = chumpy_module
    else:
        chumpy_module = sys.modules["chumpy"]
    if inserted_chumpy_ch:
        ch_module = types.ModuleType("chumpy.ch")
        ch_module.Ch = _FakeChumpyCh
        sys.modules["chumpy.ch"] = ch_module
    else:
        ch_module = sys.modules["chumpy.ch"]
    if not hasattr(chumpy_module, "ch"):
        chumpy_module.ch = ch_module

    try:
        with Path(model_pkl_path).open("rb") as handle:
            return pickle.load(handle, encoding="latin1")
    finally:
        if inserted_chumpy_ch:
            sys.modules.pop("chumpy.ch", None)
        if inserted_chumpy:
            sys.modules.pop("chumpy", None)


def _to_dense_numpy(value, *, dtype=np.float32) -> np.ndarray:
    if isinstance(value, _FakeChumpyCh):
        value = value.__dict__.get("x")
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=dtype)


def _resolve_smplx_pose_hand_prior_path(
    pose_hand_prior_path: str | Path | None,
    *,
    model_pkl_path: str | Path,
) -> Path:
    configured = resolve_pose_hand_prior_path(pose_hand_prior_path)
    if configured is not None:
        return configured

    model_path = Path(model_pkl_path).expanduser()
    candidates = [
        model_path.parent / "pose_hand_prior.npz",
        model_path.parent.parent / "pose_hand_prior.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "SMPL-X pkl backend requires MoSh++ pose_hand_prior.npz. "
        "Place it next to the model, set MUSCLEMIMIC_MOSHPP_ASSETS_PATH, "
        "or set MUSCLEMIMIC_MOSHPP_POSE_HAND_PRIOR_PATH. "
        f"Tried model-adjacent paths: {', '.join(str(candidate) for candidate in candidates)}"
    )


class _SMPLXPklLBSModel:
    """Minimal SMPL-X LBS model for MoSh++ `model.pkl` files.

    These files omit the facial landmark metadata required by `smplx.SMPLX`,
    but C3D fitting only needs vertices and joints. This keeps model loading
    independent of chumpy/psbody while preserving the MoSh++ surface model.
    """

    def __init__(
        self,
        model_pkl_path: str | Path,
        *,
        num_betas: int = 16,
        pose_hand_prior_path: str | Path | None = None,
        use_hands_mean: bool = True,
        dof_per_hand: int = 24,
    ):
        import torch

        model_pkl_path = Path(model_pkl_path)
        data = _load_moshpp_smplx_pkl(model_pkl_path)
        self.faces = np.asarray(data["f"], dtype=np.int64)
        self.num_betas = int(num_betas)
        self.v_template = torch.as_tensor(_to_dense_numpy(data["v_template"]), dtype=torch.float32)
        shapedirs = _to_dense_numpy(data["shapedirs"])[:, :, : self.num_betas]
        self.shapedirs = torch.as_tensor(shapedirs, dtype=torch.float32)
        posedirs = _to_dense_numpy(data["posedirs"])
        num_pose_basis = posedirs.shape[-1]
        self.posedirs = torch.as_tensor(np.reshape(posedirs, [-1, num_pose_basis]).T, dtype=torch.float32)
        self.J_regressor = torch.as_tensor(_to_dense_numpy(data["J_regressor"]), dtype=torch.float32)
        parents = np.asarray(data["kintree_table"][0], dtype=np.int64).copy()
        parents[0] = -1
        self.parents = torch.as_tensor(parents, dtype=torch.long)
        self.lbs_weights = torch.as_tensor(_to_dense_numpy(data["weights"]), dtype=torch.float32)
        self.full_pose_dim = 165
        self.pose_body_dof = int(num_pose_basis // 3 - 90 + 3)
        self.pose_hand_dof = int(dof_per_hand * 2)
        self.pose_dim = self.pose_body_dof + self.pose_hand_dof

        prior_path = _resolve_smplx_pose_hand_prior_path(pose_hand_prior_path, model_pkl_path=model_pkl_path)
        hand_prior = np.load(prior_path)
        components_l = np.asarray(hand_prior["componentsl"], dtype=np.float32)
        components_r = np.asarray(hand_prior["componentsr"], dtype=np.float32)
        selected_components = np.vstack(
            (
                np.hstack((components_l[:dof_per_hand], np.zeros_like(components_l[:dof_per_hand]))),
                np.hstack((np.zeros_like(components_r[:dof_per_hand]), components_r[:dof_per_hand])),
            )
        )
        if use_hands_mean:
            hands_mean = np.concatenate(
                (
                    np.asarray(hand_prior["hands_meanl"], dtype=np.float32),
                    np.asarray(hand_prior["hands_meanr"], dtype=np.float32),
                )
            )
        else:
            hands_mean = np.zeros(selected_components.shape[1], dtype=np.float32)
        self.selected_components = torch.as_tensor(selected_components, dtype=torch.float32)
        self.hands_mean = torch.as_tensor(hands_mean, dtype=torch.float32)

    def to(self, device: str):
        self.v_template = self.v_template.to(device)
        self.shapedirs = self.shapedirs.to(device)
        self.posedirs = self.posedirs.to(device)
        self.J_regressor = self.J_regressor.to(device)
        self.parents = self.parents.to(device)
        self.lbs_weights = self.lbs_weights.to(device)
        self.selected_components = self.selected_components.to(device)
        self.hands_mean = self.hands_mean.to(device)
        return self

    def to_full_pose(self, pose):
        import torch

        pose = pose.float()
        if pose.shape[-1] == self.full_pose_dim:
            return pose
        if pose.shape[-1] != self.pose_dim:
            raise ValueError(f"Expected SMPL-X pose dim {self.pose_dim} or {self.full_pose_dim}, got {pose.shape[-1]}.")

        body_pose = pose[..., : self.pose_body_dof]
        hand_coeffs = pose[..., self.pose_body_dof : self.pose_body_dof + self.pose_hand_dof]
        hand_pose = self.hands_mean.to(pose.device) + torch.matmul(
            hand_coeffs,
            self.selected_components.to(pose.device),
        )
        return torch.cat((body_pose, hand_pose), dim=-1)

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        from smplx.lbs import lbs

        pose = self.to_full_pose(pose.reshape(pose.shape[0], -1))
        batch_size = pose.shape[0]
        if th_betas is None:
            th_betas = pose.new_zeros(batch_size, self.num_betas)
        else:
            th_betas = th_betas.float()
            if th_betas.shape[-1] > self.num_betas:
                th_betas = th_betas[:, : self.num_betas]
            if th_betas.shape[0] == 1 and batch_size > 1:
                th_betas = th_betas.expand(batch_size, -1)
        verts, joints = lbs(
            th_betas,
            pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            pose2rot=True,
        )
        if th_trans is not None:
            trans = th_trans.float()
            verts = verts + trans[:, None, :]
            joints = joints + trans[:, None, :]
        return verts, joints


class SMPLXParserAdapter:
    """Minimal adapter exposing the SMPLH_Parser methods used by this fitter."""

    def __init__(
        self,
        model_path: str,
        *,
        gender: str = "neutral",
        num_betas: int = 16,
        create_transl: bool = False,
        pose_hand_prior_path: str | Path | None = None,
        use_hands_mean: bool = True,
        dof_per_hand: int = 24,
    ):
        try:
            import smplx
        except ImportError as exc:
            raise ImportError("surface_model_type='smplx' requires the `smplx` package.") from exc

        resolved_model_path, model_ext = _resolve_smplx_model_file(model_path, gender)
        if model_ext == "pkl":
            self.model = _SMPLXPklLBSModel(
                resolved_model_path,
                num_betas=num_betas,
                pose_hand_prior_path=pose_hand_prior_path,
                use_hands_mean=use_hands_mean,
                dof_per_hand=dof_per_hand,
            )
            self._uses_pkl_lbs = True
            self.pose_dim = self.model.pose_dim
        else:
            self.model = smplx.SMPLX(
                model_path=str(resolved_model_path),
                gender=gender,
                num_betas=num_betas,
                use_pca=False,
                flat_hand_mean=False,
                create_transl=create_transl,
                ext=model_ext,
            )
            self._uses_pkl_lbs = False
            self.pose_dim = 165
        self.faces = self.model.faces
        self.num_betas = num_betas

    def to(self, device: str):
        self.model = self.model.to(device)
        return self

    def to_full_pose(self, pose):
        if self._uses_pkl_lbs:
            return self.model.to_full_pose(pose)
        return pose

    def get_joints_verts(self, pose, th_betas=None, th_trans=None):
        if self._uses_pkl_lbs:
            return self.model.get_joints_verts(pose, th_betas=th_betas, th_trans=th_trans)

        if pose.shape[1] != 165:
            pose = pose.reshape(-1, 165)
        pose = pose.float()
        if th_betas is not None:
            th_betas = th_betas.float()
            if th_betas.shape[-1] > self.num_betas:
                th_betas = th_betas[:, : self.num_betas]

        out = self.model(
            betas=th_betas,
            transl=th_trans,
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:66],
            jaw_pose=pose[:, 66:69],
            leye_pose=pose[:, 69:72],
            reye_pose=pose[:, 72:75],
            left_hand_pose=pose[:, 75:120],
            right_hand_pose=pose[:, 120:165],
        )
        return out.vertices, out.joints


def _make_surface_model(
    *,
    surface_model_type: str,
    smpl_model_path: str,
    gender: str,
    device: str,
):
    if surface_model_type == "smplh":
        from loco_mujoco.smpl import SMPLH_Parser

        return SMPLH_Parser(model_path=smpl_model_path, gender=gender, create_transl=False).to(device), 156
    if surface_model_type == "smplx":
        smpl = SMPLXParserAdapter(model_path=smpl_model_path, gender=gender, create_transl=False).to(device)
        return smpl, smpl.pose_dim
    raise ValueError(f"Unsupported surface_model_type={surface_model_type!r}; expected 'smplh' or 'smplx'.")
