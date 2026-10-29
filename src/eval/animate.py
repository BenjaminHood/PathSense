"""Render one episode as an animation."""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from src.envs.layouts import corners
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from src.common.path import wrap
from src.common.types import N_RANGES, MAX_RANGE
from src.guidance.cues import PHRASES
from src.eval.plots import (_style, _save, FIGDIR,
                            C_BLUE, C_ORANGE, C_AQUA,
                            GRID, AXIS, MUTED, INK, INK_2)

# What the walker hears. The action names are mirrored relative to the phrases
# (SIM_CUES_MIRRORED), so label by phrase: "bear right" really turns them right.
ACTIONS = ["stay silent"] + [f'"{PHRASES[a]}"' for a in range(1, 5)]


def _frame(env, obs, action):
    return {
        "pos": env.pos.copy(),
        "heading": float(env.heading),
        "path_heading": float(env.path_heading),
        "ranges": np.asarray(obs[:N_RANGES], dtype=float) * MAX_RANGE,
        "action": int(action),
        "offset": float(np.degrees(wrap(env.heading - env.path_heading))),
    }


def rollout(env, policy_fn, seed=0):
    """Run one episode, recording everything needed to redraw it."""
    obs, info = env.reset(seed=seed)
    frames = []

    while True:
        action = policy_fn(obs)
        frames.append(_frame(env, obs, action))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            frames.append(_frame(env, obs, 0))
            outcome = "collision" if info["collision"] else "completed, no collision"
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

    for w in env.walls:            # walls the walker knows: solid
        ax.add_patch(plt.Polygon(corners(w), facecolor=AXIS, edgecolor="none", zorder=1))
    for b in env.boxes:            # clutter: like the circles
        ax.add_patch(plt.Polygon(corners(b), facecolor=GRID, edgecolor=AXIS, lw=0.8, zorder=1))
    for o in env.obstacles:
        ax.add_patch(plt.Circle(o[:2], o[2], facecolor=GRID,
                                edgecolor=AXIS, lw=0.8, zorder=1))

    rays = LineCollection([], colors=C_BLUE, linewidths=0.8,
                          alpha=0.40, zorder=3)
    ax.add_collection(rays)
    trail, = ax.plot([], [], color=C_BLUE, lw=1.6, zorder=4)
    intended, = ax.plot([], [], color=C_AQUA, lw=2.2, zorder=6)    # walker's own path
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
        Line2D([], [], color=C_AQUA, lw=2.2, label="intended direction"),
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
        tip = p + 1.8 * np.array([np.cos(f["path_heading"]), np.sin(f["path_heading"])])
        intended.set_data([p[0], tip[0]], [p[1], tip[1]])

        cues = [fr["pos"] for fr in frames[:k + 1] if fr["action"] != 0]
        cue_dots.set_offsets(np.array(cues) if cues else np.empty((0, 2)))

        headline.set_text(ACTIONS[f["action"]])
        tail = f"  ·  {outcome}" if k == len(frames) - 1 else ""
        off = "on path" if abs(f["offset"]) < 5 else (
            f"{abs(f['offset']):.0f}° {'left' if f['offset'] > 0 else 'right'} of path")
        sub.set_text(f"{label}   step {k}  ·  {off}"
                     f"  ·  {len(cues)} cues{tail}")
        return rays, trail, walker, intended, cue_dots, headline, sub

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
    policy = load_policy(env, "artifacts/ben_new/nav_policy.pt")

    for seed in (3, 7, 11):
        animate_episode(env, sampled(policy), seed=seed,
                        name=f"episode_sampled_s{seed}.gif",
                        label="PPO (sampled)")

    animate_episode(env, greedy(policy), seed=3,
                    name="episode_greedy_s3.gif", label="PPO (greedy)")