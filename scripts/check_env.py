from src.envs.config import EnvConfig
from src.envs.pathsense_env import PathSenseEnv

env = PathSenseEnv(EnvConfig())

# 1. progress reward magnitude per step
env.reset(seed=0)
print("progress per step:")
for _ in range(20):
    prev = env.prev_dist_to_goal
    env.step(0)                    # STAY_SILENT
    print(f"  {env.cfg.reward_progress_scale * (prev - env.prev_dist_to_goal):+.4f}")

# 2. can a crude controller reach the goal at all?
print("\nreachability:")
for seed in range(5):
    obs, info = env.reset(seed=seed)
    for _ in range(env.cfg.max_steps):
        action = 1 if env.goal[1] < env.pos[1] else 2   # CUE_LEFT / CUE_RIGHT
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            break
    print(f"  seed {seed}: steps={env.step_count} "
          f"dist={info['dist_to_goal']:.2f} reached={info['dist_to_goal'] <= env.cfg.goal_radius}")