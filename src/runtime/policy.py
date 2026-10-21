
import numpy as np
from gymnasium import spaces

import torch
from ..agents.policies import make_policy
from ..common.types import N_RANGES, OBS_DIM
from ..guidance.cues import N_ACTIONS


def load_policy(path: str = "artifacts/policy.pt", sample: bool = True, seed: int = 0):
    """Load a checkpoint with sampled policy"""
    

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
    policy = make_policy(obs_space, spaces.Discrete(N_ACTIONS), ckpt["cfg"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    rng = np.random.default_rng(seed)

    def act(obs_array: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
            probs = policy._dist(x).probs.squeeze(0).numpy().astype(np.float64)
        return int(rng.choice(len(probs), p=probs / probs.sum())) if sample else int(probs.argmax())

    # Fan width the policy was trained on. Checkpoints saved before env_cfg was
    # stored all used the 120 deg default.
    env_cfg = ckpt.get("env_cfg")
    act.fov_deg = env_cfg.fov_deg if env_cfg is not None else 120.0
    return act


def load_greedy_policy(path: str = "artifacts/policy.pt"):
    return load_policy(path, sample=False)


def check_fan_coverage(cam, fov_deg: float):
    """Warn when the policy's fan is wider than the camera can see: the outer
    rays then just copy the edge of the image instead of measuring anything."""
    cam_deg = np.degrees(cam.hfov_rad)
    if fov_deg > cam_deg + 1.0:
        rays = np.linspace(-fov_deg / 2, fov_deg / 2, N_RANGES)
        n = int(np.sum(np.abs(rays) > cam_deg / 2))
        print(f"WARNING: policy fan is {fov_deg:.0f} deg but the camera sees {cam_deg:.0f} deg; "
              f"{n} of {N_RANGES} rays are copies of the image edge. "
              f"Retrain with fov_deg <= {cam_deg:.0f}.")
