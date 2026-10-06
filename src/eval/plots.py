import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- style
# Validated categorical slots (first three pass all-pairs CVD checks).
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_BLUE_LIGHT = "#86b6ef"          # same hue, lighter step — for raw-vs-smoothed

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

FIGDIR = "artifacts/figures"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    print("wrote", path)
    return path


def _smooth(y, w=9):
    if len(y) < w:
        return np.asarray(y, dtype=float)
    kernel = np.ones(w) / w
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="same")


# ---------------------------------------------------------------- data in
def parse_log(path):
    """Read a saved console log into the same shape as the JSON history.

    Tolerates the older `k1` label and pprint's quote-wrapped line breaks,
    so logs captured before history-saving existed still plot.
    """
    text = open(path, encoding="utf-8").read()
    text = text.replace("'\n '", "").replace("('", "").replace("')", "")

    pattern = (
        r"steps\s+(\d+)\s*\|\s*return\s+(-?[\d.]+)\s*\|\s*"
        r"entropy\s+(-?[\d.]+)\s*\|\s*k[l1]\s+(-?[\d.]+)\s*\|\s*"
        r"clip\s+(-?[\d.]+)\s*\|\s*ev\s+(-?[\d.]+)"
    )
    rows = re.findall(pattern, text)
    if not rows:
        raise ValueError(f"no training lines found in {path}")

    keys = ["steps", "return", "entropy", "approx_kl", "clip_frac", "ev"]
    return [dict(zip(keys, (float(v) for v in r))) for r in rows]


def load_history(path):
    return json.load(open(path, encoding="utf-8"))


# ---------------------------------------------------------------- figure 1
def plot_training(history, name="training.png", title=None):
    """Five metrics, five panels. Never one axis with five scales."""
    _style()
    steps = [h["steps"] for h in history]

    panels = [
        ("return",    "Mean episodic return", "20-episode moving average"),
        ("entropy",   "Policy entropy",       "ln 5 = 1.609 at initialisation"),
        ("ev",        "Explained variance",   "value-function fit, 1.0 = perfect"),
        ("approx_kl", "Approximate KL",       "policy movement per update"),
        ("clip_frac", "Clip fraction",        "share of samples clipped"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 11), sharex=True)

    for ax, (key, heading, sub) in zip(axes, panels):
        y = [h[key] for h in history]

        if key == "return":
            ax.plot(steps, y, color=C_BLUE_LIGHT, lw=1.0, label="per update")
            ax.plot(steps, _smooth(y), color=C_BLUE, lw=2.0, label="smoothed")
            ax.legend(loc="lower right")
        else:
            ax.plot(steps, y, color=C_BLUE, lw=1.6)

        if key in ("return", "ev"):
            ax.axhline(0, color=AXIS, lw=0.8, zorder=0)

        ax.set_title(heading, loc="left", pad=8)
        ax.text(0.0, 1.0, sub, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=8, color=MUTED)
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("environment steps")
    if title:
        fig.suptitle(title, x=0.0, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------- figure 2
def plot_eval(results, name="evaluation.png"):
    """results: {"Random": {...}, "PPO (greedy)": {...}, ...} from run_episodes."""
    _style()

    metrics = [
        ("success", "Success rate",       "{:.2f}"),
        ("return",  "Mean return",        "{:.1f}"),
        ("steps",   "Mean episode length","{:.0f}"),
        ("cues",    "Mean cues issued",   "{:.1f}"),
    ]
    labels = list(results)
    colors = [C_BLUE, C_ORANGE, C_AQUA][:len(labels)]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))

    for ax, (key, heading, fmt) in zip(axes.ravel(), metrics):
        vals = [results[k][key] for k in labels]
        bars = ax.bar(x, vals, width=0.62, color=colors, zorder=2)

        # Direct value labels: also satisfies the contrast relief rule for aqua.
        for bar, v in zip(bars, vals):
            off = 0.02 * (max(vals) - min(min(vals), 0) or 1)
            ax.annotate(fmt.format(v),
                        (bar.get_x() + bar.get_width() / 2,
                         v + (off if v >= 0 else -off)),
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8, color=INK_2)

        if min(vals) < 0:
            ax.axhline(0, color=AXIS, lw=0.8, zorder=1)

        ax.set_title(heading, loc="left", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.margins(y=0.18)

    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------- figure 3
def plot_trajectory(env, policy_fn, seed=0, name="trajectory.png", label=""):
    """One episode: obstacles, walked path, and where cues were issued."""
    _style()

    obs, info = env.reset(seed=seed)
    path = [env.pos.copy()]
    cue_points = []
    reached = False

    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        path.append(env.pos.copy())
        if action != 0:
            cue_points.append(env.pos.copy())
        if terminated:
            reached = not info.get("collision", False)
            break
        if truncated:
            break

    path = np.asarray(path)
    size = env.cfg.arena_size

    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    for o in env.obstacles:
        ax.add_patch(plt.Circle(o[:2], o[2], facecolor=GRID,
                                edgecolor=AXIS, lw=0.8, zorder=1))

    ax.plot(path[:, 0], path[:, 1], color=C_BLUE, lw=1.8, zorder=3,
            label="walked path")

    if cue_points:
        cp = np.asarray(cue_points)
        ax.scatter(cp[:, 0], cp[:, 1], s=14, color=C_ORANGE, zorder=4,
                   label=f"cue issued (n={len(cp)})")

    ax.scatter(*path[0], s=90, facecolor="none", edgecolor=C_BLUE, lw=2,
               zorder=5, label="start")
    ax.scatter(*env.goal, s=240, marker="*", color=C_AQUA, zorder=5,
               label="goal")
    ax.add_patch(plt.Circle(env.goal, env.cfg.goal_radius, facecolor="none",
                            edgecolor=C_AQUA, lw=1.0, ls=":", zorder=2))

    outcome = "goal reached" if reached else (
        "collision" if info.get("collision") else "timed out")
    heading = f"{label} — {outcome}" if label else outcome
    ax.set_title(heading, loc="left", pad=10)
    ax.text(0.0, 1.0,
            f"seed {seed} · {len(path) - 1} steps · {len(cue_points)} cues",
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8, color=MUTED)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(False)

    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------- driver
if __name__ == "__main__":
    from src.envs.config import EnvConfig
    from src.envs.pathsense_env import PathSenseEnv
    from src.eval.baselines import run_episodes, random_policy
    from src.eval.evaluate import load_policy, greedy, sampled

    # 1. training curve — from saved history, or fall back to a console log
    hist_path = "artifacts/nav_policy_history.json"
    if os.path.exists(hist_path):
        history = load_history(hist_path)
    else:
        history = parse_log("artifacts/train_log.txt")
    plot_training(history, "training_nav.png", "PathSense — PPO training")

    # 2. evaluation comparison
    env = PathSenseEnv(EnvConfig())
    policy = load_policy(env)

    results = {
        "Random":         run_episodes(random_policy(env), env, n_episodes=100),
        "PPO (greedy)":   run_episodes(greedy(policy), env, n_episodes=100),
        "PPO (sampled)":  run_episodes(sampled(policy), env, n_episodes=100),
    }
    for k, v in results.items():
        print(f"{k:16s} {v}")
    plot_eval(results)

    # 3. one episode each, same seed, so the two are directly comparable
    plot_trajectory(env, sampled(policy), seed=3,
                    name="trajectory_sampled.png", label="PPO (sampled)")
    plot_trajectory(env, greedy(policy), seed=3,
                    name="trajectory_greedy.png", label="PPO (greedy)")