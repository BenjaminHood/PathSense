"""Camera intrinsics: how pixels map to directions in metres."""
from dataclasses import dataclass
import json
import numpy as np


@dataclass
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    height_m: float = 1.6    # camera height above the floor (glasses ~ eye height)
    pitch_rad: float = 0.0   # + = looking down. Later: from IMU gravity.

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)

    @property
    def hfov_rad(self) -> float:
        """Horizontal field of view the camera actually sees."""
        return 2 * np.arctan(self.width / (2 * self.fx))

    def resized(self, width: int, height: int) -> "CameraModel":
        sx, sy = width / self.width, height / self.height
        return CameraModel(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                           width, height, self.height_m, self.pitch_rad)

    @classmethod
    def from_fov(cls, width, height, hfov_deg, **kw) -> "CameraModel":
        """Rough model when you haven't calibrated yet (square pixels, centred)."""
        fx = width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
        return cls(fx, fx, width / 2, height / 2, width, height, **kw)

    @classmethod
    def load(cls, path) -> "CameraModel":
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
