# In the simulator and through the camera pipeline
import numpy as np, matplotlib.pyplot as plt
from src.common.types import N_RANGES
from src.envs.config import EnvConfig
from src.envs.pathsense_env import PathSenseEnv
from src.perception.camera import CameraModel
from src.perception.ranges import depth_to_ranges
from tests.synthetic import render_depth

scene = [(2.5, 1.0, 0.5), (3.0, -1.5, 0.7)]     # (forward m, left m, radius m)
env = PathSenseEnv(EnvConfig()); env.reset(seed=0)
env.pos, env.heading = np.array([5.0, 10.0]), 0.0
env.obstacles = [np.array([5 + f, 10 + l, r]) for f, l, r in scene]
sim = env._build_observation().ranges
cam = CameraModel.from_fov(640, 480, hfov_deg=125, height_m=1.6)
real = depth_to_ranges(render_depth(cam, cylinders=scene), cam, EnvConfig().fov_deg)

x = np.arange(N_RANGES)
plt.figure(figsize=(8, 3.5))
plt.bar(x - 0.2, sim[::-1], 0.4, label="simulator")
plt.bar(x + 0.2, real[::-1], 0.4, label="camera pipeline")
plt.xlabel("ray (left -> right)"); plt.ylabel("range (m)"); plt.ylim(0, 8.5); plt.legend()
plt.title("Same scene: simulator vs camera pipeline")
plt.tight_layout(); plt.savefig("artifacts/sim_vs_camera.png", dpi=120)
print("max abs diff where both see something:",
      np.abs(sim - real)[(sim < 8) & (real < 8)].max().round(3))
