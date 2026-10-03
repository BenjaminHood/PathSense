# Meta-Glasses-Indoor-Navigation

# PathSense - System Architecture

Shared reference for all three tracks. If you change anything in **§3 The Contract**, tell the other two before you push.

Status: draft. Sections marked **OPEN** are not yet decided and block other people resolve them this week.

---

## 1. What the system does

A guidance agent for indoor navigation. A camera on the user's head produces a view of what's ahead; a trained policy decides which way they should go and when to tell them; the system speaks a cue.

The same policy runs in simulation (for training) and on real hardware (for the demo). This only works because both sides produce the *same observation type* — see §3.

```
                    TRAINING                            DEPLOYMENT
                    ────────                            ──────────

              ┌──────────────────┐                ┌──────────────────┐
              │  Simulator       │                │  Camera          │
              │  (MiniWorld /    │                │  (glasses or     │
              │   Habitat)       │                │   phone/webcam)  │
              └────────┬─────────┘                └────────┬─────────┘
                       │                                   │
                       │                          ┌────────▼─────────┐
                       │                          │  Perception      │
                       │                          │  depth → ranges  │
                       │                          └────────┬─────────┘
                       │                                   │
                       └──────────┐             ┌──────────┘
                                  │             │
                            ┌─────▼─────────────▼─────┐
                            │      Observation        │   ← the contract
                            └───────────┬─────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  Policy (PPO)     │
                              └─────────┬─────────┘
                                        │
                            ┌───────────▼─────────────┐
                            │    GuidanceAction       │   ← the contract
                            └─────┬─────────────┬─────┘
                                  │             │
                       ┌──────────▼──┐     ┌────▼──────────┐
                       │  Sim step   │     │  Guidance     │
                       │  + reward   │     │  → TTS cue    │
                       └─────────────┘     └───────────────┘
```

The key property: **the policy never knows which side it's on.** Swapping sim for hardware is a config change, not a code change.

---

## 2. Repo layout

```
pathsense/
├── pyproject.toml
├── README.md
├── configs/
│   ├── env/
│   ├── agent/            # ppo_cartpole.yaml, ppo_nav.yaml
│   └── experiment/
├── src/pathsense/
│   ├── common/
│   │   ├── types.py      # THE CONTRACT — see §3
│   │   └── logging.py
│   ├── envs/
│   │   ├── config.py     # EnvConfig                           [Track A]
│   │   ├── pathsense_env.py   # Gymnasium env                  [Track A]
│   ├── agents/
│   │   ├── config.py     # PPOConfig
│   │   ├── policies.py   # ActorCritic family, make_policy()   [Track B]
│   │   └── ppo.py        # buffer, GAE, update, train loop     [Track B]
│   ├── perception/       # depth model, obs construction       [Track C]
│   ├── guidance/         # GuidanceAction → speech cue         [Track C]
│   ├── runtime/          # real-time loop, camera I/O, TTS     [Track C]
│   └── eval/             # metrics, baselines, rollout recording
├── scripts/              # train.py, evaluate.py, run_live.py
├── tests/
├── docs/                 # this file, MDP spec, report drafts
└── artifacts/            # checkpoints, logs, videos (gitignored)
```

---

## 3. The contract

`src/pathsense/common/types.py`. Everything in §1's diagram meets here. **Frozen changes require agreement from all three.**

```python
from dataclasses import dataclass
import numpy as np

N_RANGES = 16          # rays in the depth fan
MAX_RANGE = 8.0        # metres; readings clipped to this


@dataclass(frozen=True)
class Observation:
    """What the policy sees. Must be producible by BOTH the simulator
    and the real camera pipeline, in the same units."""

    ranges: np.ndarray      # (N_RANGES,) float32, metres, left→right across FOV,
                            #   clipped to MAX_RANGE, MAX_RANGE means "clear"
    goal_vector: np.ndarray # (2,) float32, goal position in AGENT frame, metres
                            #   (+x = forward, +y = left)
    speed: float            # float32, m/s, forward speed last step

    def to_array(self) -> np.ndarray:
        """Flat vector fed to the network. Normalised to roughly [-1, 1]."""
        return np.concatenate([
            self.ranges / MAX_RANGE,
            self.goal_vector / MAX_RANGE,
            [self.speed],
        ]).astype(np.float32)


OBS_DIM = N_RANGES + 3


@dataclass(frozen=True)
class GuidanceAction:
    """What the policy decides."""

    turn: float             # radians, positive = right
    speak: bool             # emit a spoken cue this step?


@dataclass(frozen=True)
class Pose:
    """Ground truth in sim; estimated (or absent) on real hardware.
    NOT part of the observation — evaluation and reward only."""

    x: float
    y: float
    heading: float          # radians
```

### Rules

1. **If it can't be computed from a live camera, it doesn't go in `Observation`.** This is the rule that keeps sim-to-real tractable. No ground-truth pose, no object labels the perception model can't produce, no map.
2. `Pose` is for reward computation and metrics. The policy must never receive it.
3. Units are metres and radians everywhere. No degrees, no pixels, no normalised-to-image-width coordinates crossing this boundary.
4. `to_array()` is the single place the observation becomes a tensor. Nobody flattens an observation by hand anywhere else.

---

## 4. Tracks and ownership

| Track | Owner | Owns | Depends on |
|---|---|---|---|
| **A — Environment** | | `envs/`, reward function, MDP spec | contract only |
| **B — RL agent** |  | `agents/`, baselines, training + eval scripts | contract only |
| **C — Perception & runtime** | | `perception/`, `guidance/`, `runtime/` | contract only |

Nobody imports from another track's package. All three import `common.types`. That's the whole point — three people, three machines, no blocking.

### Track A — Environment

Produces a Gymnasium-compatible env whose `observation_space` matches `OBS_DIM` and whose `step()` accepts the action encoding agreed in §6.

```python
class PathSenseEnv(gym.Env):
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]: ...
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]: ...
```

`info` must carry `Pose` and a `collision` flag so evaluation can compute metrics without peeking at internals.

Reward terms (draft — Track A owns the final weights, document them in `docs/mdp.md`):

- progress toward goal (dense, the main signal)
- collision penalty
- per-step time penalty
- goal-reached bonus
- cue penalty, if we go with the cue-policy framing

### Track B — RL agent

Consumes `OBS_DIM` vectors, produces actions. Validated on `CartPole-v1` before ever touching `NavEnv`.

Also owns the **baselines**, which are needed for the report regardless of how PPO performs:

- random policy
- hand-coded heuristic (turn away from nearest obstacle, head toward goal)
- A\* oracle with full map access (upper bound)

### Track C — Perception & runtime

Two jobs. First, a function with exactly this shape:

```python
def observation_from_frame(frame, goal_vector, speed) -> Observation: ...
```

Monocular depth estimation → sample `N_RANGES` rays across the FOV → clip to `MAX_RANGE`. This is what closes the sim-to-real gap.

Second, the real-time loop: capture → observe → policy → guidance → TTS, with a stated latency budget and graceful degradation when perception drops out.

---

## 5. Integration milestones

| When | Milestone |
|---|---|
| End week 1 | Contract frozen. Hardware spike answered in writing. MDP spec written. |
| End week 2 | **Vertical slice**: sim env + random policy + printed cue, one episode end to end. No learning. |
| End week 3 | PPO solves CartPole. Baselines implemented. |
| End week 5 | PPO trains on `NavEnv`, beats the heuristic baseline. |
| End week 6 | `observation_from_frame` produces valid observations from a real camera. |
| End week 7 | Live demo running. **Demo video recorded.** |
| Week 8 | Report. Buffer. |

The week-2 vertical slice is the one that matters. If it slips, cut scope rather than the deadline.

---

## 6. OPEN decisions

These block other people. Resolve this week.

**OPEN-1 — Action encoding.** `GuidanceAction` has a continuous `turn` and a boolean `speak` — a hybrid space that vanilla PPO doesn't take off the shelf. Options:

- *Discretise*: 5 turn bins × 2 speak states = 10 discrete actions. Simplest, trains fastest, and defensible (you can't give a human a 0.03-radian instruction).
- *Two heads*: Gaussian for turn, Bernoulli for speak, sum the log-probs. More elegant, more ways to get log-prob arithmetic wrong.

Recommendation: discretise. Blocks Track A (`action_space`) and Track B (policy class).

**OPEN-2 — What the RL is actually for.** Indoor navigation can be solved with SLAM + A\* and no RL at all, which a marker in an RL subject will notice. Pick a framing where the policy is load-bearing:

- *Egocentric navigation under partial observability* — no map, no global state, so there's nothing to plan over.
- *Cue policy* — a planner supplies the route; the agent decides when and what to say, trading off arrival against overwhelming the user.

Blocks the MDP spec, which blocks the reward function.

**OPEN-3 — Simulator.** MiniWorld (fast, ugly, installs cleanly) vs Habitat (photorealistic, real floorplans, painful install). Track A decides after a two-day install spike on all three machines.

**OPEN-4 — Hardware.** Whether Meta's Wearables Device Access Toolkit gives us usable frame access and audio output at acceptable latency. Until answered, **the phone/webcam fallback is the default** and the glasses are an upgrade.

---

## 7. Conventions (three machines, one repo)

- Python 3.11. `pip install -e .` so imports resolve identically everywhere.
- Pin versions in `pyproject.toml` — especially `gymnasium` and `torch`. Silent version drift across machines produces bugs that only reproduce for one person.
- Gymnasium API only: `obs, info = reset()`, `obs, reward, terminated, truncated, info = step()`. Never the old 4-tuple `done`.
- Seed everything. Any reported result is the mean over **three seeds** minimum — RL variance across seeds is large enough that a single run tells you almost nothing.
- Checkpoints and logs go in `artifacts/`, which is gitignored. Don't commit weights.
- Branch per track, PR into `main`. Nobody pushes to `main` directly.
- Changes to `common/types.py` get announced before they're pushed.