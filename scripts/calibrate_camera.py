"""Step 2. Print a 9x6 checkerboard (inner corners), measure one square in metres.
Take ~20 photos of it from different angles/distances into a folder, then:
    python -m scripts.calibrate_camera photos/ --square 0.025 --out configs/camera.json
"""
import argparse, glob
import cv2, numpy as np
from src.perception.camera import CameraModel

ap = argparse.ArgumentParser()
ap.add_argument("folder"); ap.add_argument("--square", type=float, required=True)
ap.add_argument("--cols", type=int, default=9); ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--height", type=float, default=1.6); ap.add_argument("--out", default="configs/camera.json")
a = ap.parse_args()

grid = np.zeros((a.cols * a.rows, 3), np.float32)
grid[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
obj, img, size = [], [], None
for p in sorted(glob.glob(f"{a.folder}/*")):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None: continue
    ok, corners = cv2.findChessboardCorners(g, (a.cols, a.rows))
    print(("found  " if ok else "MISSED ") + p)
    if ok:
        corners = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj.append(grid); img.append(corners); size = g.shape[::-1]
rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
cam = CameraModel(K[0, 0], K[1, 1], K[0, 2], K[1, 2], size[0], size[1], height_m=a.height)
cam.save(a.out)
print(f"RMS reprojection error {rms:.2f} px (want < 1.0)")
print(f"fx={cam.fx:.1f} fy={cam.fy:.1f} cx={cam.cx:.1f} cy={cam.cy:.1f}  HFOV={np.rad2deg(cam.hfov_rad):.1f} deg")
print("distortion", np.round(dist.ravel(), 3), "(large values -> undistort frames first)")
