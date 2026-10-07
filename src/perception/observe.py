"Track C contract"
import numpy as np
from ..common.types import Observation
from .ranges import depth_to_ranges


def observation_from_frame(frame_rgb, goal_vector, speed, *, depth_model, cam, fov_deg):
    depth = depth_model(frame_rgb)
    ranges = depth_to_ranges(depth, cam, fov_deg)
    return Observation(ranges=ranges,
                       goal_vector=np.asarray(goal_vector, np.float32),
                       speed=float(speed))
