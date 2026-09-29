import numpy as np, cv2
from src.perception.camera import CameraModel
from src.perception.goal import MarkerGoal


def test_marker_gives_agent_frame_goal():
    """Marker 3 m ahead, 0.5 m to the RIGHT -> goal_vector ~ (3.0, -0.5)."""
    cam = CameraModel.from_fov(1280, 720, hfov_deg=70)
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 200)
    tag = cv2.copyMakeBorder(tag, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    s, pad = 0.15 / 2, 0.15 * 40 / 200                       # marker half-size, white border (m)
    S = s + pad
    obj = np.array([[-S, -S, 0], [S, -S, 0], [S, S, 0], [-S, S, 0]], np.float32)
    img_pts, _ = cv2.projectPoints(obj, np.zeros(3), np.array([0.5, 0.0, 3.0]), cam.K, None)
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [279, 0], [279, 279], [0, 279]]),
                                    img_pts.reshape(4, 2).astype(np.float32))
    frame = cv2.warpPerspective(cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR), H, (1280, 720),
                                borderValue=(128, 128, 128))
    g = MarkerGoal(cam)(frame)
    assert g is not None
    assert np.allclose(g, [3.0, -0.5], atol=0.1)
