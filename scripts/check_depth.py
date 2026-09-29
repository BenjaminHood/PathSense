"""Step 3. Run the depth model on one photo, report the centre distance, save a
heatmap + the 16 ranges.  python -m scripts.check_depth wall_2m.jpg --true 2.0"""
import argparse
import cv2, numpy as np, matplotlib.pyplot as plt
from src.perception.camera import CameraModel
from src.perception.depth import DepthAnythingMetric
from src.perception.ranges import depth_to_ranges

ap = argparse.ArgumentParser()
ap.add_argument("image"); ap.add_argument("--true", type=float)
ap.add_argument("--camera", default="configs/camera.json"); ap.add_argument("--fov", type=float, default=120)
a = ap.parse_args()

rgb = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
cam = CameraModel.load(a.camera).resized(rgb.shape[1], rgb.shape[0])
depth = DepthAnythingMetric()(rgb)
h, w = depth.shape
centre = float(np.median(depth[h//2 - 10:h//2 + 10, w//2 - 10:w//2 + 10]))
print(f"centre depth {centre:.2f} m" + (f"  true {a.true:.2f}  error {centre - a.true:+.2f} m" if a.true else ""))
ranges = depth_to_ranges(depth, cam, a.fov)
print("ranges (index 0 = right):", np.round(ranges, 2))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(rgb); ax[0].set_title("frame")
im = ax[1].imshow(depth, cmap="magma_r"); fig.colorbar(im, ax=ax[1], label="m"); ax[1].set_title("depth")
ax[2].bar(range(16), ranges[::-1]); ax[2].set_ylim(0, 8)
ax[2].set_xlabel("ray (left -> right, as you see the photo)"); ax[2].set_ylabel("m")
plt.tight_layout(); out = a.image.rsplit(".", 1)[0] + "_check.png"; plt.savefig(out); print("saved", out)
