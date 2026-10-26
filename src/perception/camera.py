
# How mixals map to directions in metres
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

    def cropped(self, width: int, height: int) -> "CameraModel":
        """through the same lens centre-cropped to a new frame size"""
        # the focal length in pixels stream as it is, so the FOV shrinks with the frame. The glasses'
        # stream changes size this way (1472/1408/1392 px wide clips all measure f ~ 1230 px), not by rescaling.
        dx, dy = (self.width - width) / 2, (self.height - height) / 2
        return CameraModel(self.fx, self.fy, self.cx - dx, self.cy - dy,
                           width, height, self.height_m, self.pitch_rad)

    def fitted(self, width: int, height: int, mode: str = "auto") -> "CameraModel":
        """Adapt a calibration to a frame of another size.

        crop:  same lens, smaller window (focal length in px unchanged)
        scale: same view, resampled (focal length scales with the frame)
        auto:  crop when both sides are within 10% of the calibration (the
               glasses' own small size changes), scale otherwise -- a frame a
               third the width is a downscaled stream, not a 23 deg window
        """
        if (width, height) == (self.width, self.height):
            return self
        if mode == "auto":
            close = (abs(width / self.width - 1) <= 0.1 and abs(height / self.height - 1) <= 0.1)
            mode = "crop" if close else "scale"
        return self.cropped(width, height) if mode == "crop" else self.resized(width, height)

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
