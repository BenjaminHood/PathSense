from dataclasses import dataclass
import numpy as np

N_RANGES = 16          # rays in the depth fan
MAX_RANGE = 8.0        # metres; readings clipped to this


@dataclass(frozen=True)
class Observation:


    ranges: np.ndarray      # (N_RANGES,) float32, metres, left→right across FOV,
                            #   clipped to MAX_RANGE, MAX_RANGE means "clear"
    path_offset: float      # radians, how far avoidance cues have turned the walker off
                            #   their own path (+ = now heading left of it); 0 when no
                            #   avoidance turn is in progress (see PathTracker)
    speed: float            # float32, m/s, forward speed last step

    def to_array(self) -> np.ndarray:

        return np.concatenate([
            self.ranges / MAX_RANGE,
            [np.cos(self.path_offset), np.sin(self.path_offset)],
            [self.speed],
        ]).astype(np.float32)


OBS_DIM = N_RANGES + 3


@dataclass(frozen=True)
class GuidanceAction:


    turn: float # radians, positive = right
    speak: bool # emit a spoken cue this step?


@dataclass(frozen=True)
class Pose:
   

    x: float
    y: float
    heading: float # radians