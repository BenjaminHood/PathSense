
import numpy as np
from gymnasium import spaces

from ..common.types import OBS_DIM
from ..guidance.cues import N_ACTIONS


def load_greedy_policy(path: str = "artifacts/policy.pt"):
    import torch
    from ..agents.policies import make_policy

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obs_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
    policy = make_policy(obs_space, spaces.Discrete(N_ACTIONS), ckpt["cfg"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    def act(obs_array: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
            return int(policy._dist(x).probs.argmax().item())

    return act
