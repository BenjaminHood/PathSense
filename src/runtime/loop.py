
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..common.path import PathTracker
from ..common.types import MAX_RANGE, N_RANGES, Observation
from ..guidance.cues import (CUE_LEFT, CUE_RIGHT, CUE_STOP, PHRASES, STAY_SILENT,
                             CueGate, action_to_guidance)
from ..perception.ranges import depth_to_ranges


@dataclass
class RuntimeConfig:
    control_hz: float = 10.0               # = 1 / EnvConfig.dt
    walking_speed: float = 1.2             # = EnvConfig.walker_speed
    max_frame_age_s: float = 0.3
    max_ranges_age_s: float = 0.5
    fov_deg: float = 100.0                 # the policy's fan; set from the checkpoint
    stop_range_m: float = 0.5              # safety override: anything closer → STOP
    # Stop filter: a "stop" from the policy only counts if something is within
    # stop_filter_m inside the walking path (rays within ±stop_filter_deg of
    # straight ahead, ~0.4 m either side at 1.5 m). Otherwise it's treated as
    # silence. None disables the filter.
    stop_filter_m: float | None = 1.5
    stop_filter_deg: float = 15.0


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
        self._view = None #behind the latest FOV for the depth view

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
            # Timestamp with the frame time
            self._ranges, self._t, self._frame_t = ranges, t_frame, t_frame
            self._view = (frame, depth)
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
    """Keeps the human safe on the path they are already walking with out a goal.

    The policy sees the depth fan, its speed, and how far its own turn cues
    have taken the human off their path (PathTracker, fed by optical-flow yaw)"""

    def __init__(self, camera, policy_fn, speaker, worker: PerceptionWorker,
                 path_tracker: PathTracker | None = None, yaw_estimator=None,
                 cfg: RuntimeConfig | None = None, gate: CueGate | None = None,
                 clock=time.monotonic):
        self.cfg = cfg or RuntimeConfig()
        self.camera, self.policy_fn, self.speaker = camera, policy_fn, speaker
        self.worker, self.yaw = worker, yaw_estimator
        self.path = path_tracker or PathTracker()
        self.gate = gate or CueGate(clock=clock)
        self.speed_model = CueSpeedModel(self.cfg.walking_speed)
        self.clock = clock
        self.degraded_reason = None
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
        offset = self.path.update(yaw, dt)
        ranges, t_ranges = self.worker.latest()

        reason = None
        if frame is None or now - t_frame > cfg.max_frame_age_s:
            reason = "camera"
        elif ranges is None or now - t_ranges > cfg.max_ranges_age_s:
            reason = "perception"

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

        obs = Observation(ranges=np.clip(ranges, 0, MAX_RANGE).astype(np.float32),
                          path_offset=offset,
                          speed=float(self.speed_model.speed))
        assert obs.ranges.shape == (N_RANGES,)
        action = int(self.policy_fn(obs.to_array()))
        rec["policy_action"] = action
        if action == CUE_STOP and cfg.stop_filter_m is not None:
            ray_deg = np.linspace(-cfg.fov_deg / 2, cfg.fov_deg / 2, N_RANGES)
            ahead = obs.ranges[np.abs(ray_deg) <= cfg.stop_filter_deg]
            if ahead.size and ahead.min() >= cfg.stop_filter_m:
                action = STAY_SILENT       # nothing close in the walking path
                rec["stop_filtered"] = True
        if obs.ranges.min() < cfg.stop_range_m:
            action = CUE_STOP              # safety override, whatever the policy says
        guidance = action_to_guidance(action)
        spoke = guidance.speak and self._say(action)
        if spoke:
            self.speed_model.on_cue(action)
            if action in (CUE_LEFT, CUE_RIGHT):
                self.path.on_turn_cue()    # from now on, count how far they turn

        rec.update(action=action, turn=guidance.turn, spoke=spoke,
                   ranges=obs.ranges, path_offset=offset, tracking=self.path.active,
                   speed=obs.speed,
                   frame_age_s=now - t_frame, ranges_age_s=now - t_ranges,
                   tick_latency_s=time.perf_counter() - t0)
        return rec

    def run(self, max_seconds: float | None = None, on_tick=None):
        period = 1.0 / self.cfg.control_hz
        start = time.monotonic()
        next_t = start
        while True:
            rec = self.step()
            if on_tick is not None and on_tick(rec) is False:
                break
            if max_seconds is not None and time.monotonic() - start > max_seconds:
                break
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
