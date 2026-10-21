
import time

import numpy as np

from ..common.types import GuidanceAction

STAY_SILENT, CUE_LEFT, CUE_RIGHT, CUE_STOP, CUE_STRAIGHT = range(5)
N_ACTIONS = 5
TURN_STEP = float(np.deg2rad(45.0))

SIM_CUES_MIRRORED = True

if SIM_CUES_MIRRORED:
    PHRASES = {CUE_LEFT: "bear right", CUE_RIGHT: "bear left", CUE_STOP: "stop",
               CUE_STRAIGHT: "go straight"}
else:
    PHRASES = {CUE_LEFT: "bear left", CUE_RIGHT: "bear right", CUE_STOP: "stop",
               CUE_STRAIGHT: "go straight"}


def action_to_guidance(action: int) -> GuidanceAction:
    action = int(action)
    if action == STAY_SILENT:
        return GuidanceAction(turn=0.0, speak=False)
    if action in (CUE_LEFT, CUE_RIGHT):
        right = (action == CUE_RIGHT) != SIM_CUES_MIRRORED
        return GuidanceAction(turn=TURN_STEP if right else -TURN_STEP, speak=True)
    if action in (CUE_STOP, CUE_STRAIGHT):
        return GuidanceAction(turn=0.0, speak=True)
    raise ValueError(f"unknown action {action}")


class CueGate:
    

    def __init__(self, same_cue_cooldown_s=2.0, any_cue_cooldown_s=0.8, stop_repeat_s=1.5,
                 clock=time.monotonic):
        self.same, self.any, self.clock = same_cue_cooldown_s, any_cue_cooldown_s, clock
        self.stop_repeat = stop_repeat_s
        self.last_time, self.last_any, self.last_cue = {}, -np.inf, None

    def allow(self, action: int) -> bool:
        if action == STAY_SILENT:
            return False
        now = self.clock()
        if action == CUE_STOP:
            # Never held back by other cues, but don't repeat "stop" every tick
            # while the walker is already stopped.
            if self.last_cue == CUE_STOP and now - self.last_time[CUE_STOP] < self.stop_repeat:
                return False
        else:
            if now - self.last_any < self.any:
                return False
            if now - self.last_time.get(action, -np.inf) < self.same:
                return False
        self.last_time[action] = now
        self.last_any = now
        self.last_cue = action
        return True
