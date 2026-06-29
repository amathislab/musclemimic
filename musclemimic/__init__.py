"""MuscleMimic: Accelerated Motion Imitation for Biomechanical Models"""

__version__ = "0.1.0"

# Lazy imports to avoid dependency issues during package discovery
__all__ = [
    "set_all_caches",
    "set_amass_path",
    "set_c3d_model_path",
    "set_converted_amass_path",
    "set_converted_c3d_path",
    "set_converted_lafan1_path",
    "set_lafan1_path",
    "set_moshpp_assets_path",
    "set_smpl_model_path",
]


def __getattr__(name):
    """Lazy import for utils functions"""
    if name in __all__:
        from .utils import (
            set_all_caches,
            set_amass_path,
            set_c3d_model_path,
            set_converted_amass_path,
            set_converted_c3d_path,
            set_converted_lafan1_path,
            set_lafan1_path,
            set_moshpp_assets_path,
            set_smpl_model_path,
        )

        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
