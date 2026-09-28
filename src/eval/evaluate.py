import numpy as np
import torch

from src.agents.config import PPOConfig
from src.agents.policies import make_policy
from src.envs.config import EnvConfig
from src.envs.pathsense_env import PathSenseEnv
from src.eval.baselines import run_episodes, random_policy


def load_policy(env, path="artifacts/policy.pt"):
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["cfg"]
    policy = make_policy(env.observation_space, env.action_space, cfg)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy


def greedy(policy):
    def act(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return int(policy._dist(obs_t).probs.argmax().item())
    return act


if __name__ == "__main__":
    env = PathSenseEnv(EnvConfig())

    print("random:", run_episodes(random_policy(env), env, n_episodes=100))

    policy = load_policy(env)
    print("ppo:   ", run_episodes(greedy(policy), env, n_episodes=100))