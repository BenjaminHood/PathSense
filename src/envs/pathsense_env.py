import numpy as np
import gymnasium as gym
from gymnasium import spaces
 
from ..common.types import Observation, N_RANGES, MAX_RANGE, OBS_DIM
from ..agents.config import PPOConfig

# Actions
STAY_SILENT, CUE_LEFT, CUE_RIGHT, CUE_STOP, CUE_STRAIGHT = range(5)
N_ACTIONS = 5

TURN_STEP = np.deg2rad(30.0)
REPEAT_WINDOW = 15

def _ray_circle_hit(origin, direction, center, radius):
    """Distance along `direction` (unit vector) from `origin` to the nearest
    intersection with a circle, or None if there isn't one ahead of us."""
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
    """Distance to the arena boundary (axis-aligned box) along `direction`."""
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
    def __init__(self, cfg):
        super().__init__()

        # Config
        self.cfg = cfg or PPOConfig()

        # 5 Discrete actions 
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Observations
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

        # State
        self.pos = np.array([1.0, cfg.arena_size / 2.0], dtype=np.float64)

        self._rng = np.random.default_rng()

    # Episode setup 
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.cfg
 
        self.pos = np.array([1.0, cfg.arena_size / 2.0], dtype=np.float64)
        self.heading = 0.0
        self.goal = self._sample_goal()
        self.obstacles = self._sample_obstacles()
 
        # Per-episode simulated-human parameters (this is the thing PPO has
        # to learn to be robust to -- it's not observed).
        self.reaction_delay = int(self._rng.integers(*cfg.reaction_delay_range))
        self.compliance_prob = float(self._rng.uniform(*cfg.compliance_range))
        self.heading_noise_std = float(self._rng.uniform(*cfg.heading_noise_std_range))
 
        self.intent = "straight"     # walker's current behaviour
        self.pending_cue = None      # {"type": int, "remaining": int} or None
        self.last_cue_step = {}      # action_id -> step it was last issued
        self.step_count = 0
        self.prev_dist_to_goal = self._dist_to_goal()
 
        obs = self._build_observation()
        info = {"pose": (self.pos[0], self.pos[1], self.heading)}

        return obs.to_array(), info

    def _sample_goal(self):
        cfg = self.cfg
        return np.array(
            [self._rng.uniform(cfg.arena_size * 0.6, cfg.arena_size - 1.0),
             self._rng.uniform(1.0, cfg.arena_size - 1.0)],
            dtype=np.float64,
        )
 
    def _sample_obstacles(self):
        cfg = self.cfg
        n = int(self._rng.integers(cfg.n_obstacles_range[0], cfg.n_obstacles_range[1] + 1))
        obstacles = []
        attempts = 0
        while len(obstacles) < n and attempts < n * 20:
            attempts += 1
            c = self._rng.uniform(2.0, cfg.arena_size - 2.0, size=2)
            r = self._rng.uniform(*cfg.obstacle_radius_range)
            if np.linalg.norm(c - self.pos) < 2.0 or np.linalg.norm(c - self.goal) < 2.0:
                continue
            if any(np.linalg.norm(c - o[:2]) < r + o[2] + 0.3 for o in obstacles):
                continue
            obstacles.append(np.array([c[0], c[1], r]))
        return obstacles

    def step(self, action):
        cfg = self.cfg

        self.step_count += 1
        reward = cfg.reward_step
 
        # 1. Cue cost + repeat penalty, then queue the cue.
        if action != STAY_SILENT:
            reward += cfg.reward_cue
            last = self.last_cue_step.get(action)
            if last is not None and (self.step_count - last) < REPEAT_WINDOW:
                reward += cfg.reward_repeat_cue
            self.last_cue_step[action] = self.step_count
            # A fresh instruction overrides whatever was already pending.
            self.pending_cue = {"type": action, "remaining": self.reaction_delay}
 
        # 2. Advance the pending cue; resolve compliance when it lands.
        if self.pending_cue is not None:
            self.pending_cue["remaining"] -= 1
            if self.pending_cue["remaining"] <= 0:
                if self._rng.uniform() < self.compliance_prob:
                    self._apply_cue(self.pending_cue["type"])
                self.pending_cue = None
 
        # 3. Walker dynamics: heading noise every step, move if not stopped.
        self.heading += self._rng.normal(0.0, self.heading_noise_std)
        if self.intent != "stopped":
            self.pos = self.pos + cfg.walker_speed * cfg.dt * np.array(
                [np.cos(self.heading), np.sin(self.heading)]
            )
        self.pos = np.clip(self.pos, 0.0, cfg.arena_size)
 
        # 4. Progress reward.
        dist = self._dist_to_goal()
        reward += cfg.reward_progress_scale * (self.prev_dist_to_goal - dist)
        self.prev_dist_to_goal = dist
 
        # 5. Proximity / termination checks.
        clearance = self._min_obstacle_clearance()
        terminated = False
        if clearance <= cfg.collision_margin:
            reward += cfg.reward_collision
            terminated = True
        elif clearance <= cfg.near_miss_margin:
            reward += cfg.reward_near_miss
 
        if not terminated and dist <= cfg.goal_radius:
            reward += cfg.reward_goal
            terminated = True

        truncated = self.step_count >= cfg.max_steps
 
        obs = self._build_observation()
        info = {
            "pose": (self.pos[0], self.pos[1], self.heading),
            "dist_to_goal": dist,
            "compliance_prob": self.compliance_prob,
            "reaction_delay": self.reaction_delay,
        }

        return obs.to_array(), reward, terminated, truncated, info

    def _apply_cue(self, cue_type: int):
        if cue_type == CUE_LEFT:
            self.heading -= TURN_STEP
            self.intent = "straight"
        elif cue_type == CUE_RIGHT:
            self.heading += TURN_STEP
            self.intent = "straight"
        elif cue_type == CUE_STOP:
            self.intent = "stopped"
        elif cue_type == CUE_STRAIGHT:
            self.intent = "straight"

    # Geometry helpers
    def _dist_to_goal(self):
        return float(np.linalg.norm(self.goal - self.pos))
 
    def _min_obstacle_clearance(self):
        if not self.obstacles:
            return np.inf
        return min(np.linalg.norm(self.pos - o[:2]) - o[2] for o in self.obstacles)
 
    def _build_observation(self) -> Observation:
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
 
        rot = np.array([[np.cos(-self.heading), -np.sin(-self.heading)],
                         [np.sin(-self.heading), np.cos(-self.heading)]])
        goal_vector = rot @ (self.goal - self.pos)
 
        return Observation(
            ranges=ranges,
            goal_vector=goal_vector.astype(np.float32),
            speed=float(cfg.walker_speed if self.intent != "stopped" else 0.0),
        )
 
 
if __name__ == "__main__":
    # Smoke test: random policy for a couple of episodes.
    env = PathSenseEnv()
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