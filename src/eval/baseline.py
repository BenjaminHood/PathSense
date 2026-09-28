import numpy as np


def run_episodes(policy_fn, env, n_episodes=100, seed=0):
    """policy_fn(obs) -> action. Returns aggregate metrics."""
    results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward, steps, cues, reached = 0.0, 0, 0, False

        while True:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if action != 0:          # STAY_SILENT
                cues += 1
            if terminated:
                reached = info["dist_to_goal"] <= env.cfg.goal_radius
                break
            if truncated:
                break

        results.append({
            "return": total_reward,
            "steps": steps,
            "cues": cues,
            "success": float(reached),
        })

    return {k: float(np.mean([r[k] for r in results])) for k in results[0]}


def random_policy(env):
    return lambda obs: env.action_space.sample()