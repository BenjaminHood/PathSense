import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..common.path import PathTracker, wrap
from ..common.types import Observation, Pose, N_RANGES, MAX_RANGE, OBS_DIM
from ..guidance.cues import CueGate
from .config import EnvConfig

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

        m = 4.0
        self.pos = self._rng.uniform(m, cfg.arena_size - m, size=2).astype(np.float64)
        self.path_heading = float(self._rng.uniform(-np.pi, np.pi))   # where THEY want to go
        self.heading = self.path_heading
        self.obstacles = self._sample_obstacles()

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
        self.step_count = 0
        self.prev_clearance = self._min_obstacle_clearance()
        self._last_ranges = None

        obs = self._build_observation(lag=True)
        info = {
            "pose": Pose(float(self.pos[0]), float(self.pos[1]), float(self.heading)),
            "collision": False,
        }

        return obs.to_array(), info

    def _sample_obstacles(self):
        cfg = self.cfg
        n = int(self._rng.integers(cfg.n_obstacles_range[0], cfg.n_obstacles_range[1] + 1))
        obstacles = []
        attempts = 0
        while len(obstacles) < n and attempts < n * 20:
            attempts += 1
            c = self._rng.uniform(2.0, cfg.arena_size - 2.0, size=2)
            r = self._rng.uniform(*cfg.obstacle_radius_range)
            if np.linalg.norm(c - self.pos) < 2.0:
                continue
            if any(np.linalg.norm(c - o[:2]) < r + o[2] + 0.3 for o in obstacles):
                continue
            obstacles.append(np.array([c[0], c[1], r]))
        return obstacles

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
        step = np.zeros(2)
        if self.intent != "stopped":
            noise = self._rng.normal(0.0, self.heading_noise_std)
            self.heading = wrap(self.heading + noise)
            self.path_heading = wrap(self.path_heading + noise)
            step = cfg.walker_speed * self.dt * np.array([np.cos(self.heading), np.sin(self.heading)])
        else:
            reward += cfg.reward_stopped
        self.pos = np.clip(self.pos + step, 0.0, cfg.arena_size)

        # Same measurement the live loop gets from optical flow. When tracking
        # ends, the walker takes their current direction as their path.
        was_tracking = self.tracker.active
        self.tracker.update(wrap(self.heading - heading_before), self.dt)
        if was_tracking and not self.tracker.active:
            self.path_heading = self.heading

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

    def _update_intent(self):
        """The walker's own turns. They turn by the same amount as their path,
        and hold them off while an avoidance turn is in progress."""
        if self.tracker.active:
            return
        cfg = self.cfg
        turn = 0.0
        if self.step_count >= self.next_intent_change:
            lo, hi = np.deg2rad(cfg.intent_turn_deg_range)
            turn = float(self._rng.uniform(lo, hi) * self._rng.choice([-1, 1]))
            self.next_intent_change = self.step_count + int(
                self._rng.integers(*cfg.intent_change_steps_range))
        ahead = self.pos + cfg.wall_lookahead_m * np.array(
            [np.cos(self.path_heading + turn), np.sin(self.path_heading + turn)])
        if np.any(ahead < 1.0) or np.any(ahead > cfg.arena_size - 1.0):
            to_centre = np.full(2, cfg.arena_size / 2.0) - self.pos
            new = np.arctan2(to_centre[1], to_centre[0]) + self._rng.uniform(-np.pi / 4, np.pi / 4)
            turn = wrap(new - self.path_heading)
        if turn:
            self.path_heading = wrap(self.path_heading + turn)
            self.heading = wrap(self.heading + turn)

    # Geometry helpers
    def _min_obstacle_clearance(self):
        size = self.cfg.arena_size
        walls = min(self.pos[0], self.pos[1], size - self.pos[0], size - self.pos[1])
        if not self.obstacles:
            return walls
        return min(walls, min(np.linalg.norm(self.pos - o[:2]) - o[2] for o in self.obstacles))

    def _build_observation(self, lag=False) -> Observation:
        cfg = self.cfg
        half_fov = np.deg2rad(cfg.fov_deg) / 2.0
        offsets = np.linspace(-half_fov, half_fov, N_RANGES)
        ranges = np.full(N_RANGES, MAX_RANGE, dtype=np.float32)

        box_min = np.array([0.0, 0.0])
        box_max = np.array([cfg.arena_size, cfg.arena_size])

        for i, off in enumerate(offsets):
            theta = self.heading + off
            direction = np.array([np.cos(theta), np.sin(theta)])
            best = _ray_box_hit(self.pos, direction, box_min, box_max) or MAX_RANGE
            for obs in self.obstacles:
                hit = _ray_circle_hit(self.pos, direction, obs[:2], obs[2])
                if hit is not None and hit < best:
                    best = hit
            ranges[i] = min(best, MAX_RANGE)

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
