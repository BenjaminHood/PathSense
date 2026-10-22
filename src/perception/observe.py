"Track C contract"
import numpy as np
from ..common.types import Observation
from .ranges import depth_to_ranges


def observation_from_frame(frame_rgb, path_offset, speed, *, depth_model, cam, fov_deg):
    depth = depth_model(frame_rgb)
    ranges = depth_to_ranges(depth, cam, fov_deg)
    return Observation(ranges=np.asarray(ranges, np.float32),
                       path_offset=float(path_offset),
                       speed=float(speed))
