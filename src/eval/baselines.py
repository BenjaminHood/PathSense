import numpy as np


def run_episodes(policy_fn, env, n_episodes=100, seed=0):
    """Mean per-episode metrics for the path-keeping task.

    collision     fraction of episodes that ended in a collision
    progress_m    metres walked along the walker's own intended path
    cues_per_min  cues issued per minute
    stopped       fraction of time spent standing still
    offset_deg    mean |heading offset from their path|, degrees
    """
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total, steps, cues, progress, stopped, offset = 0.0, 0, 0, 0.0, 0, 0.0

        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            cues += action != 0          # STAY_SILENT
            progress += info["progress_m"]
            stopped += info["stopped"]
            offset += abs(np.degrees(info["path_offset"]))
            if terminated or truncated:
                break

        results.append({
            "return": total,
            "collision": float(info["collision"]),
            "progress_m": progress,
            "cues_per_min": cues / (steps * info["dt"] / 60),
            "stopped": stopped / steps,
            "offset_deg": offset / steps,
            "steps": steps,
        })

    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


def random_policy(env):
    return lambda obs: env.action_space.sample()


def silent_policy(env=None):
    """Never cues: what the walker does unaided. The bar to beat."""
    return lambda obs: 0

if __name__ == "__main__":
    from src.envs.pathsense_env import PathSenseEnv
    from src.envs.config import EnvConfig

    env = PathSenseEnv(EnvConfig())
    print("random:", run_episodes(random_policy(env), env, n_episodes=100))
    print("silent:", run_episodes(silent_policy(env), env, n_episodes=100))