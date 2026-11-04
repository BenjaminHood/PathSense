"""Per-layout evaluation, including how often the policy turns someone when nothing
but known walls is ahead (the corridor over-cueing we saw live).

    python -m scripts.eval_layouts artifacts/layouts/nav_policy.pt
"""
import sys

import numpy as np

from src.envs.config import EnvConfig
from src.envs.layouts import rays_circles, rays_rects
from src.envs.pathsense_env import CUE_LEFT, CUE_RIGHT, PathSenseEnv
from src.eval.baselines import run_episodes, silent_policy
from src.eval.evaluate import greedy, load_policy, sampled


def clutter_ahead(env, reach=3.0):
    """True if clutter (not walls) is within reach anywhere in the camera fan."""
    half = np.deg2rad(env.cfg.fov_deg) / 2
    th = env.heading + np.linspace(-half, half, 16)
    d = np.stack([np.cos(th), np.sin(th)], 1)
    circ = np.array(env.obstacles).reshape(-1, 3)
    hit = np.minimum(rays_rects(env.pos, d, env.boxes), rays_circles(env.pos, d, circ))
    return bool(hit.min() < reach)


def needless_turns_per_min(policy_fn, env, n=200):
    turns, minutes = 0, 0.0
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        while True:
            a = policy_fn(obs)
            if a in (CUE_LEFT, CUE_RIGHT) and not clutter_ahead(env):
                turns += 1
            obs, _, te, tr, info = env.step(a)
            minutes += info["dt"] / 60
            if te or tr:
                break
    return turns / minutes


if __name__ == "__main__":
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    for lay in ["room", "corridors", "aisles"]:
        env = PathSenseEnv(EnvConfig(layout_probs={lay: 1.0}))
        policy = load_policy(env, path)
        for name, fn in [("Silent", silent_policy(env)), ("PPO greedy", greedy(policy)),
                         ("PPO sampled", sampled(policy))]:
            r = run_episodes(fn, env, n_episodes=n)
            nt = needless_turns_per_min(fn, env, n)
            print(f"{lay:9s} {name:11s} collision {r['collision']:.2f}  path {r['progress_m']:5.1f} m  "
                  f"cues/min {r['cues_per_min']:5.1f}  turns/min with only walls ahead {nt:5.1f}  "
                  f"stopped {r['stopped']:.2f}", flush=True)
