from dataclasses import dataclass
import numpy as np

N_RANGES = 16          # rays in the depth fan
MAX_RANGE = 8.0        # metres; readings clipped to this


@dataclass(frozen=True)
class Observation:
    

    ranges: np.ndarray      # (N_RANGES,) float32, metres, left→right across FOV,
                            #   clipped to MAX_RANGE, MAX_RANGE means "clear"
    goal_vector: np.ndarray # (2,) float32, goal position in AGENT frame, metres
                            #   (+x = forward, +y = left)
    speed: float            # float32, m/s, forward speed last step

    def to_array(self) -> np.ndarray:
       
        return np.concatenate([
            self.ranges / MAX_RANGE,
            self.goal_vector / MAX_RANGE,
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