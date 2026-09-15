import os

import torch
import yaml
from scipy.spatial.transform import Rotation as R


def _default_policy_config_path():
    env_path = os.environ.get("ICROWDNAV_POLICY_CONFIG")
    if env_path:
        return env_path
    try:
        import rospkg
        return os.path.join(rospkg.RosPack().get_path("drl_policy"), "config", "default.yaml")
    except Exception:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
        )


def load_policy_config(path=None):
    path = path or _default_policy_config_path()
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def build_camera_matrices(camera_cfg, device):
    intrinsics = torch.tensor(
        [[camera_cfg["fx"], 0.0, camera_cfg["cx"]],
         [0.0, camera_cfg["fy"], camera_cfg["cy"]],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    extrinsics = torch.eye(4, dtype=torch.float32, device=device).repeat(2, 1, 1)
    for idx, side in enumerate(("left", "right")):
        side_cfg = camera_cfg[side]
        extrinsics[idx, :3, :3] = torch.tensor(
            R.from_euler("xyz", side_cfg["rpy_deg"], degrees=True).as_matrix(),
            dtype=torch.float32,
            device=device,
        )
        extrinsics[idx, :3, 3] = torch.tensor(
            side_cfg["translation"], dtype=torch.float32, device=device
        )
    return intrinsics, extrinsics
