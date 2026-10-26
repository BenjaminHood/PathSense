from src.envs.config import EnvConfig
from src.envs.pathsense_env import PathSenseEnv
from src.eval.baselines import random_policy, run_episodes, silent_policy

env = PathSenseEnv(EnvConfig())

# 1. progress reward per step: walking on their own path earns ~speed * dt
env.reset(seed=0)
print("progress per step (silent, walking on their own path):")
for _ in range(10):
    _, r, term, trunc, info = env.step(0)                  # STAY_SILENT
    print(f"  {info['progress_m']:+.4f} m   reward {r:+.4f}   stopped={info['stopped']}")
    if term or trunc:
        break

# 2. the bar a trained policy has to beat: the walker on their own, and random cues
print("\nbaselines over 100 episodes:")
for name, fn in [("silent", silent_policy(env)), ("random", random_policy(env))]:
    m = run_episodes(fn, env, n_episodes=100)
    print(f"  {name:7s} collision {m['collision']:.2f} | {m['progress_m']:.1f} m along path | "
          f"{m['cues_per_min']:.0f} cues/min | stopped {m['stopped']:.2f}")
