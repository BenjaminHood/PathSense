import numpy as np


def wrap(a: float) -> float:
    """Angle to (-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class PathTracker:
    """How far the walker has been turned off the path they were already on.

    Used identically by the simulator and the live loop, so the policy sees the
    same signal in both. The user's own turns are their business: while no
    avoidance turn is in progress the offset is 0, whatever they do. A turn cue
    starts tracking; from then on every measured heading change adds to the
    offset, so the policy can tell when (and which way) to bring them back.
    Tracking ends once they have been back within on_path_deg for settle_s, or
    after max_s (by then the new direction is taken as the one they chose).

    offset > 0: walker is heading LEFT of their path (radians).
    """

    def __init__(self, on_path_deg=10.0, settle_s=2.0, max_s=15.0):
        self.on_path = np.deg2rad(on_path_deg)
        self.settle_s, self.max_s = settle_s, max_s
        self.reset()

    def reset(self):
        self.active, self.offset = False, 0.0
        self.t_active = self.t_settled = 0.0

    def on_turn_cue(self):
        if not self.active:
            self.active, self.offset, self.t_active = True, 0.0, 0.0
        self.t_settled = 0.0

    def update(self, yaw_left: float, dt: float) -> float:
        """Feed the heading change since the last update; returns the offset."""
        if not self.active:
            return 0.0
        self.offset = wrap(self.offset + yaw_left)
        self.t_active += dt
        self.t_settled = self.t_settled + dt if abs(self.offset) < self.on_path else 0.0
        if self.t_settled >= self.settle_s or self.t_active >= self.max_s:
            self.reset()
        return self.offset
