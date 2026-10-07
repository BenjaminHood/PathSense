
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..common.types import MAX_RANGE, N_RANGES, Observation
from ..guidance.cues import (CUE_STOP, PHRASES, STAY_SILENT, CueGate,
                             action_to_guidance)
from ..perception.goal import GoalTracker
from ..perception.ranges import depth_to_ranges


@dataclass
class RuntimeConfig:
    control_hz: float = 10.0               # = 1 / EnvConfig.dt
    walking_speed: float = 1.2             # = EnvConfig.walker_speed
    goal_radius: float = 0.8               # = EnvConfig.goal_radius
    max_frame_age_s: float = 0.3
    max_ranges_age_s: float = 0.5
    fov_deg: float = 120.0                 # = EnvConfig.fov_deg (the policy's fan)
    stop_range_m: float = 0.5              # safety override: anything closer → STOP


class CueSpeedModel:
   

    def __init__(self, walking_speed: float):
        self.walking = walking_speed
        self.stopped = False

    @property
    def speed(self) -> float:
        return 0.0 if self.stopped else self.walking

    def on_cue(self, action: int):
        if action == CUE_STOP:
            self.stopped = True
        elif action != STAY_SILENT:
            self.stopped = False


class PerceptionWorker:
   

    def __init__(self, camera, depth_model, cam, fov_deg=120.0, clock=time.monotonic):
        self.camera, self.depth_model, self.cam = camera, depth_model, cam
        self.fov_deg, self.clock = fov_deg, clock
        self._ranges, self._t, self._frame_t = None, 0.0, None
        self.last_latency_s = None
        self.last_error = None
        self._lock = threading.Lock()
        self._running = False

    def run_once(self) -> bool:
        frame, t_frame = self.camera.latest()
        if frame is None or t_frame == self._frame_t:
            return False
        t0 = time.perf_counter()
        try:
            rgb = np.ascontiguousarray(frame[..., ::-1])
            depth = self.depth_model(rgb)
            cam = self.cam.resized(depth.shape[1], depth.shape[0])
            ranges = depth_to_ranges(depth, cam, self.fov_deg)
        except Exception as e:            # model failure = perception dropout
            self.last_error = repr(e)
            return False
        with self._lock:
            # Timestamp with the FRAME time: the fan is as old as its image.
            self._ranges, self._t, self._frame_t = ranges, t_frame, t_frame
            self.last_latency_s = time.perf_counter() - t0
        return True

    def latest(self):
        with self._lock:
            return self._ranges, self._t

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            if not self.run_once():
                time.sleep(0.005)

    def stop(self):
        self._running = False


class LiveLoop:
    def __init__(self, camera, policy_fn, speaker, worker: PerceptionWorker,
                 goal_tracker: GoalTracker, yaw_estimator=None,
                 cfg: RuntimeConfig | None = None, gate: CueGate | None = None,
                 clock=time.monotonic):
        self.cfg = cfg or RuntimeConfig()
        self.camera, self.policy_fn, self.speaker = camera, policy_fn, speaker
        self.worker, self.goal_tracker, self.yaw = worker, goal_tracker, yaw_estimator
        self.gate = gate or CueGate(clock=clock)
        self.speed_model = CueSpeedModel(self.cfg.walking_speed)
        self.clock = clock
        self.degraded_reason = None
        self.arrived = False
        self._last_frame_t = None

    def _say(self, action):
        if self.gate.allow(action):
            self.speaker.say(PHRASES[action])
            return True
        return False

    def step(self) -> dict:
        
        cfg = self.cfg
        dt = 1.0 / cfg.control_hz
        now = self.clock()
        t0 = time.perf_counter()
        rec = {"t": now}

        frame, t_frame = self.camera.latest()
        fresh = frame is not None and t_frame != self._last_frame_t
        self._last_frame_t = t_frame
        yaw = self.yaw(frame) if (fresh and self.yaw is not None) else 0.0
        goal = self.goal_tracker.update(dt, self.speed_model.speed, yaw,
                                        frame if fresh else None)
        ranges, t_ranges = self.worker.latest()

        reason = None
        if frame is None or now - t_frame > cfg.max_frame_age_s:
            reason = "camera"
        elif ranges is None or now - t_ranges > cfg.max_ranges_age_s:
            reason = "perception"
        elif goal is None:
            reason = "no_goal"

        if reason is not None:
            if self.degraded_reason is None:          # entering degraded mode
                self.speed_model.on_cue(CUE_STOP)
                rec["spoke"] = self._say(CUE_STOP)
            self.degraded_reason = reason
            rec.update(degraded=reason, action=CUE_STOP)
            return rec
        if self.degraded_reason is not None:
            rec["recovered_from"] = self.degraded_reason
            self.degraded_reason = None

        if np.linalg.norm(goal) <= cfg.goal_radius:
            if not self.arrived:
                self.speaker.say("You have arrived")
                self.arrived = True
            rec.update(arrived=True, goal=goal)
            return rec

        obs = Observation(ranges=np.clip(ranges, 0, MAX_RANGE).astype(np.float32),
                          goal_vector=np.asarray(goal, np.float32),
                          speed=float(self.speed_model.speed))
        assert obs.ranges.shape == (N_RANGES,)
        action = int(self.policy_fn(obs.to_array()))
        rec["policy_action"] = action
        if obs.ranges.min() < cfg.stop_range_m:
            action = CUE_STOP              # safety override, whatever the policy says
        guidance = action_to_guidance(action)
        spoke = guidance.speak and self._say(action)
        if spoke:
            self.speed_model.on_cue(action)

        rec.update(action=action, turn=guidance.turn, spoke=spoke,
                   ranges=obs.ranges, goal=obs.goal_vector, speed=obs.speed,
                   frame_age_s=now - t_frame, ranges_age_s=now - t_ranges,
                   tick_latency_s=time.perf_counter() - t0)
        return rec

    def run(self, max_seconds: float | None = None, on_tick=None):
        period = 1.0 / self.cfg.control_hz
        start = time.monotonic()
        next_t = start
        while not self.arrived:
            rec = self.step()
            if on_tick is not None and on_tick(rec) is False:
                break
            if max_seconds is not None and time.monotonic() - start > max_seconds:
                break
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
