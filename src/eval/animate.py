"""Render one episode as an animation."""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from src.common.types import N_RANGES, MAX_RANGE
from src.eval.plots import (_style, _save, FIGDIR,
                            C_BLUE, C_ORANGE, C_AQUA,
                            GRID, AXIS, MUTED, INK, INK_2)

ACTIONS = ["stay silent", "cue left", "cue right", "cue stop", "cue straight"]


def rollout(env, policy_fn, seed=0):
    """Run one episode, recording everything needed to redraw it."""
    obs, info = env.reset(seed=seed)
    frames = []

    while True:
        action = policy_fn(obs)
        frames.append({
            "pos": env.pos.copy(),
            "heading": float(env.heading),
            "ranges": np.asarray(obs[:N_RANGES], dtype=float) * MAX_RANGE,
            "action": int(action),
            "dist": float(np.linalg.norm(env.goal - env.pos)),
        })
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            frames.append({
                "pos": env.pos.copy(),
                "heading": float(env.heading),
                "ranges": np.asarray(obs[:N_RANGES], dtype=float) * MAX_RANGE,
                "action": 0,
                "dist": float(info["dist_to_goal"]),
            })
            if terminated and not info["collision"]:
                outcome = "goal reached"
            elif info["collision"]:
                outcome = "collision"
            else:
                outcome = "timed out"
            break

    return frames, outcome


def animate_episode(env, policy_fn, seed=0, name="episode.gif",
                    fps=15, label=""):
    frames, outcome = rollout(env, policy_fn, seed)
    size = env.cfg.arena_size
    half_fov = np.deg2rad(env.cfg.fov_deg) / 2.0
    offsets = np.linspace(-half_fov, half_fov, N_RANGES)

    _style()
    fig, ax = plt.subplots(figsize=(7, 7.4))

    for o in env.obstacles:
        ax.add_patch(plt.Circle(o[:2], o[2], facecolor=GRID,
                                edgecolor=AXIS, lw=0.8, zorder=1))
    ax.add_patch(plt.Circle(env.goal, env.cfg.goal_radius, facecolor="none",
                            edgecolor=C_AQUA, lw=1.2, ls=":", zorder=2))
    ax.scatter(*env.goal, s=240, marker="*", color=C_AQUA, zorder=6)

    rays = LineCollection([], colors=C_BLUE, linewidths=0.8,
                          alpha=0.40, zorder=3)
    ax.add_collection(rays)
    trail, = ax.plot([], [], color=C_BLUE, lw=1.6, zorder=4)
    cue_dots = ax.scatter(np.empty(0), np.empty(0), s=16,
                          color=C_ORANGE, zorder=5)
    walker = ax.scatter(np.empty(0), np.empty(0), s=80,
                        color=C_BLUE, zorder=7)

    headline = ax.text(0.02, 0.975, "", transform=ax.transAxes, va="top",
                       fontsize=11, color=INK)
    sub = ax.text(0.02, 0.935, "", transform=ax.transAxes, va="top",
                  fontsize=9, color=MUTED)

    ax.legend(handles=[
        Line2D([], [], color=C_BLUE, lw=1.6, label="walked path"),
        Line2D([], [], color=C_BLUE, lw=0.8, alpha=0.4, label="range readings"),
        Line2D([], [], color=C_ORANGE, marker=".", ls="none", label="cue issued"),
        Line2D([], [], color=C_AQUA, marker="*", ls="none",
               markersize=11, label="goal"),
    ], loc="upper right", fontsize=8)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    fig.tight_layout()

    def update(k):
        f = frames[k]
        p, h = f["pos"], f["heading"]
        dirs = np.stack([np.cos(h + offsets), np.sin(h + offsets)], axis=1)
        ends = p + dirs * f["ranges"][:, None]
        rays.set_segments([[p, e] for e in ends])

        path = np.array([fr["pos"] for fr in frames[:k + 1]])
        trail.set_data(path[:, 0], path[:, 1])
        walker.set_offsets(p.reshape(1, 2))

        cues = [fr["pos"] for fr in frames[:k + 1] if fr["action"] != 0]
        cue_dots.set_offsets(np.array(cues) if cues else np.empty((0, 2)))

        headline.set_text(ACTIONS[f["action"]])
        tail = f"  ·  {outcome}" if k == len(frames) - 1 else ""
        sub.set_text(f"{label}   step {k}  ·  {f['dist']:.1f} m to goal"
                     f"  ·  {sum(1 for c in cues)} cues{tail}")
        return rays, trail, walker, cue_dots, headline, sub

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / fps, blit=False)

    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    if name.endswith(".gif"):
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    print("wrote", path, f"({len(frames)} frames, {outcome})")
    return outcome


if __name__ == "__main__":
    from src.envs.config import EnvConfig
    from src.envs.pathsense_env import PathSenseEnv
    from src.eval.evaluate import load_policy, greedy, sampled

    env = PathSenseEnv(EnvConfig())
    policy = load_policy(env, "artifacts/nav_policy.pt")

    for seed in (3, 7, 11):
        animate_episode(env, sampled(policy), seed=seed,
                        name=f"episode_sampled_s{seed}.gif",
                        label="PPO (sampled)")

    animate_episode(env, greedy(policy), seed=3,
                    name="episode_greedy_s3.gif", label="PPO (greedy)")