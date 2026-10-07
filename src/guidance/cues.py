"""Policy action id -> spoken words. The ONLY place that decides what is said.

WARNING: in the current sim, CUE_LEFT does heading -= 30 deg, which under the
contract's +y = left turns the walker RIGHT (and CUE_RIGHT turns left).
Until Track A fixes it, the words follow what the action actually DOES in sim.
After the fix, set SIM_CUES_MIRRORED = False and re-train/re-test.

Action ids mirror src/envs/pathsense_env.py (copied, not imported, to keep
tracks independent); tests/test_camera_pipeline.py keeps them in sync.
"""
import time

import numpy as np

from ..common.types import GuidanceAction

STAY_SILENT, CUE_LEFT, CUE_RIGHT, CUE_STOP, CUE_STRAIGHT = range(5)
N_ACTIONS = 5
TURN_STEP = float(np.deg2rad(30.0))

SIM_CUES_MIRRORED = True

if SIM_CUES_MIRRORED:
    PHRASES = {CUE_LEFT: "bear right", CUE_RIGHT: "bear left", CUE_STOP: "stop",
               CUE_STRAIGHT: "go straight"}
else:
    PHRASES = {CUE_LEFT: "bear left", CUE_RIGHT: "bear right", CUE_STOP: "stop",
               CUE_STRAIGHT: "go straight"}


def action_to_guidance(action: int) -> GuidanceAction:
    """What the cue physically asks for. turn: radians, + = right (contract)."""
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
    """Human-side rate limiting on top of whatever the policy learned.
    STOP is never suppressed."""

    def __init__(self, same_cue_cooldown_s=2.0, any_cue_cooldown_s=0.8, clock=time.monotonic):
        self.same, self.any, self.clock = same_cue_cooldown_s, any_cue_cooldown_s, clock
        self.last_time, self.last_any = {}, -np.inf

    def allow(self, action: int) -> bool:
        if action == STAY_SILENT:
            return False
        now = self.clock()
        if action != CUE_STOP:
            if now - self.last_any < self.any:
                return False
            if now - self.last_time.get(action, -np.inf) < self.same:
                return False
        self.last_time[action] = now
        self.last_any = now
        return True
