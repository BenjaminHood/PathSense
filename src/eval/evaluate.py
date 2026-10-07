import numpy as np
import torch
import matplotlib.pyplot as plt

from src.eval.plots import plot_eval
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

def sampled(policy):
    def act(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _ = policy.act(obs_t)
            return int(action.item())
    return act


if __name__ == "__main__":
    env = PathSenseEnv(EnvConfig())
    policy = load_policy(env)
    results = {
        "Random":        run_episodes(random_policy(env), env, n_episodes=1000),
        "PPO (greedy)":  run_episodes(greedy(policy), env, n_episodes=1000),
        "PPO (sampled)": run_episodes(sampled(policy), env, n_episodes=1000),
    }
    for k, v in results.items():
        print(f"{k:16s} {v}")

    plot_eval(results)
    plt.show()