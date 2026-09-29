"""Step 7. Webcam -> depth -> Observation -> policy -> speech, at the sim's 10 Hz.
    python -m scripts.run_live --camera configs/camera.json [--no-speech]
Keys: q quits."""
import argparse, time
import cv2, numpy as np, torch
from src.common.types import MAX_RANGE
from src.envs.config import EnvConfig
from src.perception.camera import CameraModel
from src.perception.depth import DepthAnythingMetric
from src.perception.goal import MarkerGoal
from src.perception.ranges import depth_to_ranges
from src.common.types import Observation
from src.guidance.cues import PHRASES
from src.eval.evaluate import load_policy
from src.envs.pathsense_env import PathSenseEnv

ap = argparse.ArgumentParser()
ap.add_argument("--camera", default="configs/camera.json"); ap.add_argument("--device", type=int, default=0)
ap.add_argument("--no-speech", action="store_true"); ap.add_argument("--speed", type=float, default=1.2)
a = ap.parse_args()

env_cfg = EnvConfig(); DT = env_cfg.dt                     # 0.1 s -> 10 decisions/s
policy = load_policy(PathSenseEnv(env_cfg))
cap = cv2.VideoCapture(a.device)
ok, frame = cap.read(); assert ok, "no camera"
cam = CameraModel.load(a.camera).resized(frame.shape[1], frame.shape[0])
depth_model, goal_finder = DepthAnythingMetric(), MarkerGoal(cam)
speaker = None if a.no_speech else __import__("src.guidance.cues", fromlist=["Speaker"]).Speaker()

last_goal, last_ranges, last_ok = np.array([5.0, 0.0], np.float32), None, 0.0
while True:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    if not ok: break
    t1 = time.perf_counter()
    depth = depth_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); t2 = time.perf_counter()
    ranges = depth_to_ranges(depth, cam, env_cfg.fov_deg);     t3 = time.perf_counter()
    g = goal_finder(frame)
    if g is not None: last_goal = g
    obs = Observation(ranges=ranges, goal_vector=last_goal, speed=a.speed)
    with torch.no_grad():
        action = int(policy._dist(torch.as_tensor(obs.to_array())[None]).probs.argmax())
    if ranges.min() < 0.5: action = 3                        # safety override: too close -> stop
    t4 = time.perf_counter()
    if speaker: speaker.say(action)
    print(f"cap {1e3*(t1-t0):4.0f}ms depth {1e3*(t2-t1):4.0f}ms ranges {1e3*(t3-t2):3.0f}ms "
          f"policy {1e3*(t4-t3):3.0f}ms | min {ranges.min():.1f}m goal {np.round(last_goal,1)} "
          f"-> {PHRASES.get(action, '(silent)')}")
    cv2.imshow("PathSense", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
    time.sleep(max(0.0, DT - (time.perf_counter() - t0)))   # hold the 10 Hz tick
