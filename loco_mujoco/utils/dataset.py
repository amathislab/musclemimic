import argparse
import os
from pathlib import Path

import yaml

import loco_mujoco


def set_amass_path():
    """
    Set the path to the AMASS dataset.
    """
    parser = argparse.ArgumentParser(description="Set the AMASS dataset path.")
    parser.add_argument("--path", type=str, help="Path to the AMASS dataset.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "MUSCLEMIMIC_AMASS_PATH", path_to_conf=loco_mujoco.get_variables_path())


def set_smpl_model_path():
    """
    Set the path to the SMPL model.
    """
    parser = argparse.ArgumentParser(description="Set the SMPL model path.")
    parser.add_argument("--path", type=str, help="Path to the SMPL model.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "MUSCLEMIMIC_SMPL_MODEL_PATH", path_to_conf=loco_mujoco.get_variables_path())


def set_c3d_model_path():
    """
    Set the SMPL-X/SMPL-H model path used by C3D marker fitting.
    """
    parser = argparse.ArgumentParser(description="Set the C3D fitting body model path.")
    parser.add_argument("--path", type=str, help="Path to SMPL-X/SMPL-H models used by C3D marker fitting.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "MUSCLEMIMIC_C3D_MODEL_PATH", path_to_conf=loco_mujoco.get_variables_path())


def set_moshpp_assets_path():
    """
    Set the path to MoSh++ auxiliary assets used by C3D fitting.
    """
    parser = argparse.ArgumentParser(description="Set the MoSh++ auxiliary assets path.")
    parser.add_argument("--path", type=str, help="Directory containing pose_body_prior.pkl and MoSh++ .npz assets.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "MUSCLEMIMIC_MOSHPP_ASSETS_PATH", path_to_conf=loco_mujoco.get_variables_path())


def set_all_caches():
    """
    Set the path to which all converted datasets will be stored. This sets the following Variables:
    - MUSCLEMIMIC_CONVERTED_AMASS_PATH
    - MUSCLEMIMIC_CONVERTED_C3D_PATH
    - MUSCLEMIMIC_CONVERTED_LAFAN1_PATH
    - MUSCLEMIMIC_CONVERTED_DEFAULT_PATH

    Returns:

    """
    parser = argparse.ArgumentParser(description="Set the path to which all converted datasets will be stored.")
    parser.add_argument("--path", type=str, help="Path to which all converted datasets will be stored.")
    args = parser.parse_args()
    amass_path = os.path.join(args.path, "AMASS")
    _set_path_in_yaml_conf(
        amass_path, "MUSCLEMIMIC_CONVERTED_AMASS_PATH", path_to_conf=loco_mujoco.get_variables_path()
    )
    c3d_path = os.path.join(args.path, "C3D")
    _set_path_in_yaml_conf(c3d_path, "MUSCLEMIMIC_CONVERTED_C3D_PATH", path_to_conf=loco_mujoco.get_variables_path())
    lafan1_path = os.path.join(args.path, "LAFAN1")
    _set_path_in_yaml_conf(
        lafan1_path, "MUSCLEMIMIC_CONVERTED_LAFAN1_PATH", path_to_conf=loco_mujoco.get_variables_path()
    )
    default_path = os.path.join(args.path, "DEFAULT")
    _set_path_in_yaml_conf(
        default_path, "MUSCLEMIMIC_CONVERTED_DEFAULT_PATH", path_to_conf=loco_mujoco.get_variables_path()
    )


def set_converted_amass_path():
    """
    Set the path to which the converted AMASS dataset is stored.
    """
    parser = argparse.ArgumentParser(description="Set the path to which the converted AMASS dataset is stored.")
    parser.add_argument("--path", type=str, help="Path to which the converted AMASS dataset is stored.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "MUSCLEMIMIC_CONVERTED_AMASS_PATH", path_to_conf=loco_mujoco.get_variables_path())


def set_converted_c3d_path():
    """
    Set the path to which the converted C3D dataset is stored.
    """
    parser = argparse.ArgumentParser(description="Set the path to which the converted C3D dataset is stored.")
    parser.add_argument("--path", type=str, help="Path to which the converted C3D dataset is stored.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "MUSCLEMIMIC_CONVERTED_C3D_PATH", path_to_conf=loco_mujoco.get_variables_path())


def set_lafan1_path():
    """
    Set the path to the LAFAN1 dataset.
    """
    parser = argparse.ArgumentParser(description="Set the LAFAN1 dataset path.")
    parser.add_argument("--path", type=str, help="Path to the LAFAN1 dataset.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(args.path, "MUSCLEMIMIC_LAFAN1_PATH", path_to_conf=loco_mujoco.get_variables_path())


def set_converted_lafan1_path():
    """
    Set the path to which the converted LAFAN1 dataset is stored.
    """
    parser = argparse.ArgumentParser(description="Set the path to which the converted LAFAN1 dataset is stored.")
    parser.add_argument("--path", type=str, help="Path to which the converted LAFAN1 dataset is stored.")
    args = parser.parse_args()
    _set_path_in_yaml_conf(
        args.path, "MUSCLEMIMIC_CONVERTED_LAFAN1_PATH", path_to_conf=loco_mujoco.get_variables_path()
    )


def _set_path_in_yaml_conf(path: str, attr: str, path_to_conf: str):
    """
    Set the path in the yaml configuration file.
    """
    config_path = Path(path_to_conf)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # create an empty yaml file if it does not exist
    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump({}, file)

    # load yaml file
    data = loco_mujoco.load_path_config(config_path)

    # set the path
    data[attr] = path

    # save the yaml file
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=True)

    print(f"Set {attr} to {path} in file {config_path}.")
