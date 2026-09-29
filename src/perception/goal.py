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
