"""Goal vector from an ArUco marker stuck at the goal (door, exit sign...)."""
import numpy as np
import cv2


class MarkerGoal:
    def __init__(self, cam, marker_id=0, marker_size_m=0.15,
                 dictionary=cv2.aruco.DICT_4X4_50):
        self.cam, self.marker_id, self.size = cam, marker_id, marker_size_m
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary))
        s = marker_size_m / 2
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)

    def __call__(self, frame_bgr):
        """Returns goal_vector (2,) in AGENT frame (+x forward, +y left), or None."""
        corners, ids, _ = self.detector.detectMarkers(frame_bgr)
        if ids is None or self.marker_id not in ids.flatten():
            return None
        c = corners[list(ids.flatten()).index(self.marker_id)][0].astype(np.float32)
        ok, _, tvec = cv2.solvePnP(self.obj, c, self.cam.K, None,
                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        x_right, z_fwd = float(tvec[0, 0]), float(tvec[2, 0])
        return np.array([z_fwd, -x_right], np.float32)   # camera -> agent frame


# ---------------------------------------------------------------------------
# Between marker sightings: dead reckoning with visually estimated yaw.

def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


class VisualYawEstimator:
    """Yaw change between consecutive frames from sparse optical flow.

    Turning LEFT shifts the scene RIGHT in the image by ~fx * tan(yaw). The
    median horizontal flow ignores most of the left/right-symmetric flow
    from walking forward. Returns radians, + = turned left (contract frame).
    """

    def __init__(self, cam, max_corners=150):
        self.cam, self.max_corners, self.prev = cam, max_corners, None

    def __call__(self, frame_bgr) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, gray
        if prev is None or prev.shape != gray.shape:
            return 0.0
        pts = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 8)
        if pts is None or len(pts) < 10:
            return 0.0
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        ok = status.ravel() == 1
        if ok.sum() < 10:
            return 0.0
        dx = float(np.median((nxt[ok] - pts[ok])[:, 0, 0]))
        return float(np.arctan2(dx, self.cam.fx))


class GoalTracker:
    """Marker fix when visible; otherwise carry the goal forward ourselves.
    Goal is in the agent frame (+x forward, +y left), metres."""

    def __init__(self, initial_goal=None, marker=None):
        self.goal = None if initial_goal is None else np.asarray(initial_goal, np.float64)
        self.marker = marker
        self.last_fix_age_s = 0.0 if self.goal is not None else np.inf

    def update(self, dt, speed, yaw_left, frame_bgr=None):
        if self.goal is not None:
            # We moved forward speed*dt then turned yaw_left; the goal moves the
            # opposite way in our frame.
            self.goal = _rot(-yaw_left) @ (self.goal - np.array([speed * dt, 0.0]))
            self.last_fix_age_s += dt
        if self.marker is not None and frame_bgr is not None:
            fix = self.marker(frame_bgr)
            if fix is not None:
                self.goal = np.asarray(fix, np.float64)
                self.last_fix_age_s = 0.0
        return None if self.goal is None else self.goal.astype(np.float32)
