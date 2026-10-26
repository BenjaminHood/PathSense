import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..common.path import PathTracker, wrap
from ..common.types import Observation, Pose, N_RANGES, MAX_RANGE, OBS_DIM
from ..guidance.cues import CueGate
from .config import EnvConfig
from .layouts import LAYOUTS, rays_circles, rays_rects, rect_clearance

# Actions
STAY_SILENT, CUE_LEFT, CUE_RIGHT, CUE_STOP, CUE_STRAIGHT = range(5)
N_ACTIONS = 5

REPEAT_WINDOW = 15

def _ray_circle_hit(origin, direction, center, radius):

    oc = origin - center
    b = 2.0 * np.dot(direction, oc)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t = (-b - sqrt_disc) / 2.0
    if t < 0:
        t = (-b + sqrt_disc) / 2.0
    return t if t > 1e-6 else None

def _ray_box_hit(origin, direction, box_min, box_max):

    t_min, t_max = 0.0, np.inf
    for i in range(2):
        if abs(direction[i]) < 1e-9:
            if origin[i] < box_min[i] or origin[i] > box_max[i]:
                return None
            continue
        t1 = (box_min[i] - origin[i]) / direction[i]
        t2 = (box_max[i] - origin[i]) / direction[i]
        t1, t2 = min(t1, t2), max(t1, t2)
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    return t_max if t_max > 1e-6 else None

class PathSenseEnv(gym.Env):
    """The walker has their own intended path (path_heading). The agent's job
    is to keep them safe on it: steer them past obstacles and back again, with
    as few cues as possible. There is no goal; episodes have a fixed length."""

    def __init__(self, cfg):
        super().__init__()

        # Config
        self.cfg = cfg or EnvConfig()

        # 5 Discrete actions
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Observations
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

        # State
        self.pos = np.full(2, self.cfg.arena_size / 2.0)
        self.tracker = PathTracker()

        self._rng = np.random.default_rng()

    # Episode setup
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.cfg

        # Layout: walls the walker knows, clutter they don't (src/envs/layouts.py).
        names = list(cfg.layout_probs)
        p = np.array([cfg.layout_probs[k] for k in names], float)
        self.layout = names[int(self._rng.choice(len(names), p=p / p.sum()))]
        self.walls, self.boxes, circles, self.structured = LAYOUTS[self.layout](self._rng, cfg)
        self.obstacles = list(circles)
        self._place_walker()

        # Per-episode simulated-human parameters (this is the thing PPO has
        # to learn to be robust to -- it's not observed).
        self.reaction_delay = int(self._rng.integers(*cfg.reaction_delay_range))
        self.compliance_prob = float(self._rng.uniform(*cfg.compliance_range))
        self.heading_noise_std = float(self._rng.uniform(*cfg.heading_noise_std_range))
        self.next_intent_change = int(self._rng.integers(*cfg.intent_change_steps_range))

        self.dt = float(self._rng.uniform(*cfg.dt_range)) if cfg.dt_range else cfg.dt
        self.tracker.reset()
        # The same rate limit the live system puts on speech, on the sim clock.
        self.sim_t = 0.0
        self.gate = CueGate(clock=lambda: self.sim_t)

        # walker's current behaviour; some episodes start stopped so "speed 0"
        # isn't an unseen state the policy can get stuck in
        stopped = self._rng.uniform() < cfg.start_stopped_prob
        self.intent = "stopped" if stopped else "straight"
        self.pending_cue = None      # {"type": int, "remaining": int} or None
        self.last_cue_step = {}      # action_id -> step it was last issued
        self._committed_until = 0    # walker's own wall turn: walk it out before re-deciding
        self.step_count = 0
        self.prev_clearance = self._min_obstacle_clearance()
        self._last_ranges = None

        obs = self._build_observation(lag=True)
        info = {
            "pose": Pose(float(self.pos[0]), float(self.pos[1]), float(self.heading)),
            "collision": False,
        }

        return obs.to_array(), info

    def _place_walker(self):
        """A free start, facing somewhere they could actually walk."""
        size, rng = self.cfg.arena_size, self._rng
        m = 1.0 if self.structured else 4.0
        for _ in range(500):
            self.pos = rng.uniform(m, size - m, size=2).astype(np.float64)
            if rect_clearance(self.pos, self.walls) < 0.6 or self._clutter_clearance() < 1.5:
                continue
            if self.structured:     # along a corridor / aisle axis, the most open way
                axes = np.arange(4) * np.pi / 2
                free = np.array([self._free_length(a) for a in axes])
                ok = np.flatnonzero(free >= 4.0)
                a = axes[rng.choice(ok)] if ok.size else axes[np.argmax(free)]
                self.path_heading = float(wrap(a))
            else:
                self.path_heading = float(rng.uniform(-np.pi, np.pi))   # where THEY want to go
            self.heading = self.path_heading
            # at least 3 m clear straight ahead, so the first cue has time to land
            if self._clear_ahead() >= 3.0:
                break

    def _sweep(self, heading, walls_only):
        """Free length along heading for a walker's body: three parallel rays,
        centre and 0.35 m either side, so they can't slide along an edge."""
        d = np.array([np.cos(heading), np.sin(heading)])
        side = np.array([-d[1], d[0]])
        circ = np.array(self.obstacles).reshape(-1, 3)
        best = np.inf
        for off in (-0.35, 0.0, 0.35):
            o, dd = self.pos + off * side, d[None]
            t = min(self._arena_exit(dd, o)[0], rays_rects(o, dd, self.walls)[0])
            if not walls_only:
                t = min(t, rays_rects(o, dd, self.boxes)[0], rays_circles(o, dd, circ)[0])
            best = min(best, t)
        return float(best)

    def _clear_ahead(self):
        return self._sweep(self.heading, walls_only=False)

    def _free_length(self, heading):
        """How far the walker could go this way past the walls they know (not clutter)."""
        return self._sweep(heading, walls_only=True)

    def _clutter_clearance(self):
        circ = np.array(self.obstacles).reshape(-1, 3)
        c = (np.linalg.norm(self.pos - circ[:, :2], axis=1) - circ[:, 2]).min() if len(circ) else np.inf
        return min(c, rect_clearance(self.pos, self.boxes))

    def _arena_exit(self, dirs, origin=None):
        size = self.cfg.arena_size
        p = self.pos if origin is None else origin
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dirs > 0, (size - p) / dirs, np.where(dirs < 0, -p / dirs, np.inf))
        return np.maximum(t.min(axis=1), 0.0)

    def step(self, action):
        cfg = self.cfg

        self.step_count += 1
        self.sim_t += self.dt
        reward = cfg.reward_step

        # 1. Cue cost + repeat penalty, then queue the cue if the live system
        #    would actually say it. A blocked cue is silence for the walker but
        #    still costs, so rapid-fire cues are pointless, as they are live.
        spoken = False
        if action != STAY_SILENT:
            reward += cfg.reward_cue
            last = self.last_cue_step.get(action)
            if last is not None and (self.step_count - last) < REPEAT_WINDOW:
                reward += cfg.reward_repeat_cue
            self.last_cue_step[action] = self.step_count
            spoken = self.gate.allow(action)
            if spoken:
                # A fresh instruction overrides whatever was already pending.
                self.pending_cue = {"type": action, "remaining": self.reaction_delay}
                if action in (CUE_LEFT, CUE_RIGHT):
                    self.tracker.on_turn_cue()

        heading_before = self.heading

        # 2. Advance the pending cue; resolve compliance when it lands.
        if self.pending_cue is not None:
            self.pending_cue["remaining"] -= 1
            if self.pending_cue["remaining"] <= 0:
                if self._rng.uniform() < self.compliance_prob:
                    self._apply_cue(self.pending_cue["type"])
                self.pending_cue = None

        # 3. The walker's own intentions: corners, doors, not walking into walls.
        self._update_intent()

        # 4. Walker dynamics. Heading noise is their own wandering while
        #    walking, so it moves their path too; a cue-induced turn is held
        #    until another cue (they don't drift back by themselves).
        #    In corridors and aisles their path is the layout's axis: noise is
        #    veering off it, partly corrected by feel (cane on the wall).
        step = np.zeros(2)
        if self.intent != "stopped":
            noise = self._rng.normal(0.0, self.heading_noise_std)
            self.heading = wrap(self.heading + noise)
            if not self.structured:
                self.path_heading = wrap(self.path_heading + noise)
            else:
                if not self.tracker.active:
                    self.heading = wrap(self.heading + cfg.self_correct_rate
                                        * wrap(self.path_heading - self.heading))
                self.heading = wrap(self.heading + self._cane())
            step = cfg.walker_speed * self.dt * np.array([np.cos(self.heading), np.sin(self.heading)])
        else:
            reward += cfg.reward_stopped
        self.pos = np.clip(self.pos + step, 0.0, cfg.arena_size)

        # Same measurement the live loop gets from optical flow. When tracking
        # ends, the walker takes their current direction as their path.
        was_tracking = self.tracker.active
        self.tracker.update(wrap(self.heading - heading_before), self.dt)
        if was_tracking and not self.tracker.active:
            self.path_heading = self._snap(self.heading)

        # 5. Progress along the walker's own intended path.
        progress = float(step @ np.array([np.cos(self.path_heading), np.sin(self.path_heading)]))
        reward += cfg.reward_progress_scale * progress

        # 6. Proximity / termination checks (arena walls count as obstacles).
        clearance = self._min_obstacle_clearance()
        terminated = False
        collision = bool(clearance <= cfg.collision_margin)
        if collision:
            reward += cfg.reward_collision
            terminated = True
        elif clearance <= cfg.near_miss_margin and clearance < self.prev_clearance:
            reward += cfg.reward_near_miss      # closing in; standing near something is fine
        self.prev_clearance = clearance

        truncated = self.step_count >= cfg.max_steps

        obs = self._build_observation(lag=True)
        info = {
            "pose": Pose(float(self.pos[0]), float(self.pos[1]), float(self.heading)),
            "collision": collision,
            "progress_m": progress,
            "path_offset": wrap(self.heading - self.path_heading),   # true offset, radians
            "tracking": self.tracker.active,
            "cue_spoken": spoken,
            "stopped": self.intent == "stopped",
            "compliance_prob": self.compliance_prob,
            "reaction_delay": self.reaction_delay,
            "dt": self.dt,
        }

        return obs.to_array(), reward, terminated, truncated, info

    def _apply_cue(self, cue_type: int):
        turn = np.deg2rad(self.cfg.turn_deg)
        if cue_type == CUE_LEFT:
            self.heading -= turn
            self.intent = "straight"
        elif cue_type == CUE_RIGHT:
            self.heading += turn
            self.intent = "straight"
        elif cue_type == CUE_STOP:
            self.intent = "stopped"
        elif cue_type == CUE_STRAIGHT:
            self.intent = "straight"

    def _cane(self):
        """Turn (rad) away from a known wall within cane reach, on either side."""
        reach = self.cfg.cane_reach_m
        h = self.heading
        sides = np.array([[np.cos(h + a), np.sin(h + a)] for a in (np.pi / 3, -np.pi / 3)])
        left, right = np.minimum(rays_rects(self.pos, sides, self.walls), self._arena_exit(sides))
        if min(left, right) >= reach:
            return 0.0
        return -0.1 if left < right else 0.1     # + = turn left

    def _snap(self, heading):
        """In corridors/aisles a path runs along an axis; elsewhere it is free."""
        if not self.structured:
            return heading
        return wrap(np.round(heading / (np.pi / 2)) * np.pi / 2)

    def _update_intent(self):
        """The walker's own turns, around the walls they know (not the clutter).
        They hold them off while an avoidance turn is in progress."""
        if self.tracker.active:
            return
        cfg, rng = self.cfg, self._rng
        # in a building they know the walls, so they turn later (cross-aisles are short)
        look = min(cfg.wall_lookahead_m, 1.5) if self.structured else cfg.wall_lookahead_m
        if self.step_count < self._committed_until:      # just turned: walk it out first
            return
        turn = 0.0
        if self.step_count >= self.next_intent_change:
            # their own choice: a side corridor in a building, any angle in a room;
            # if that way is a wall, try again next step (i.e. at the next junction)
            if self.structured:
                t = float(np.pi / 2 * rng.choice([-1, 1]))
            else:
                lo, hi = np.deg2rad(cfg.intent_turn_deg_range)
                t = float(rng.uniform(lo, hi) * rng.choice([-1, 1]))
            if self._free_length(self.path_heading + t) >= look:
                turn = t
                self.next_intent_change = self.step_count + int(
                    rng.integers(*cfg.intent_change_steps_range))
        if not turn and self._free_length(self.path_heading) < look:
            # wall or dead end ahead: take the most open way they know of
            # (a U-turn only if nothing else is open)
            opts = np.deg2rad([90, -90, 180] if self.structured else [45, -45, 90, -90, 135, -135])
            free = np.array([self._free_length(self.path_heading + o) for o in opts])
            ok = np.flatnonzero(free >= look)
            if ok.size > 1 and self.structured:
                ok = ok[opts[ok] != np.pi]
            turn = float(opts[rng.choice(ok)] if ok.size else opts[np.argmax(free)])
            self._committed_until = self.step_count + 5      # < 1 m, inside the look-ahead
        if turn:
            self.path_heading = wrap(self.path_heading + turn)
            self.heading = wrap(self.heading + turn)

    # Geometry helpers
    def _min_obstacle_clearance(self):
        size = self.cfg.arena_size
        arena = min(self.pos[0], self.pos[1], size - self.pos[0], size - self.pos[1])
        return min(arena, rect_clearance(self.pos, self.walls), self._clutter_clearance())

    def _build_observation(self, lag=False) -> Observation:
        cfg = self.cfg
        half_fov = np.deg2rad(cfg.fov_deg) / 2.0
        offsets = np.linspace(-half_fov, half_fov, N_RANGES)
        theta = self.heading + offsets
        dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        hits = np.minimum.reduce([
            self._arena_exit(dirs),
            rays_rects(self.pos, dirs, self.walls),
            rays_rects(self.pos, dirs, self.boxes),
            rays_circles(self.pos, dirs, np.array(self.obstacles).reshape(-1, 3)),
        ])
        if cfg.range_noise_std:        # depth from a camera is never exact
            hits = hits * (1.0 + self._rng.normal(0.0, cfg.range_noise_std, N_RANGES))
        ranges = np.clip(hits, 0.0, MAX_RANGE).astype(np.float32)

        if lag:
            # Live depth is slower than the control loop, so sometimes the
            # policy acts on the previous fan.
            if self._last_ranges is not None and self._rng.uniform() < cfg.stale_ranges_prob:
                ranges, self._last_ranges = self._last_ranges, ranges
            else:
                self._last_ranges = ranges

        return Observation(
            ranges=ranges,
            path_offset=self.tracker.offset,
            speed=float(cfg.walker_speed if self.intent != "stopped" else 0.0),
        )


if __name__ == "__main__":
    # Smoke test: random policy for a couple of episodes.
    env = PathSenseEnv(EnvConfig())
    for ep in range(2):
        obs, info = env.reset(seed=ep)
        total = 0.0
        for _ in range(env.cfg.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        print(f"episode {ep}: return={total:.2f} steps={env.step_count} info={info}")
