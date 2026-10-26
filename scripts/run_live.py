
import argparse
import os
import time

import numpy as np

from src.common.path import PathTracker
from src.guidance.cues import PHRASES
from src.guidance.tts import make_speaker
from src.perception.camera import CameraModel
from src.perception.goal import VisualYawEstimator
from src.runtime.loop import LiveLoop, PerceptionWorker, RuntimeConfig
from src.runtime.policy import check_fan_coverage, load_policy
from src.runtime.camera import StaticCamera, ThreadedCamera
from tests.synthetic import render_depth


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="configs/camera.json", help="calibration (Step 2)")
    ap.add_argument("--hfov", type=float, default=70.0, help="used if --camera file is missing")
    ap.add_argument("--frame-fit", choices=["auto", "crop", "scale"], default="auto",
                    help="how frames of another size relate to the calibration: crop, scale, "
                         "or auto (crop if within 10%% of the calibration size, else scale)")
    ap.add_argument("--device", default="0", help="webcam index, video file or stream URL")
    ap.add_argument("--checkpoint", default="artifacts/nav_policy.pt")
    ap.add_argument("--speed", type=float, default=1.2, help="walking speed, m/s")
    ap.add_argument("--greedy", action="store_true",
                    help="always take the most likely action instead of sampling")
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_cam(a, width, height):
    if os.path.exists(a.camera):
        return CameraModel.load(a.camera).fitted(width, height, a.frame_fit)
    print(f"{a.camera} not found; using an uncalibrated {a.hfov:.0f} deg model")
    return CameraModel.from_fov(width, height, hfov_deg=a.hfov)


def build(a):
    
    if a.dry_run:
        cam = CameraModel.from_fov(320, 240, hfov_deg=a.hfov)
        depth = render_depth(cam, cylinders=[(3.0, 0.4, 0.4)])     # pillar 3 m ahead, a bit left
        camera = StaticCamera(np.zeros((240, 320, 3), np.uint8))
        depth_model = lambda rgb: depth                            # noqa: E731
    else:
        from src.perception.depth import DepthAnythingMetric

        camera = ThreadedCamera(int(a.device) if a.device.isdigit() else a.device)
        frame = None
        for _ in range(100):
            frame, _ = camera.latest()
            if frame is not None:
                break
            time.sleep(0.05)
        assert frame is not None, "no camera"
        cam = load_cam(a, frame.shape[1], frame.shape[0])
        print("loading depth model ...", flush=True)
        depth_model = DepthAnythingMetric()

    policy = load_policy(a.checkpoint, sample=not a.greedy)
    check_fan_coverage(cam, policy.fov_deg)
    cfg = RuntimeConfig(walking_speed=a.speed, fov_deg=policy.fov_deg)
    worker = PerceptionWorker(camera, depth_model, cam, cfg.fov_deg)
    loop = LiveLoop(camera, policy, make_speaker(not a.no_speech), worker, PathTracker(),
                    yaw_estimator=None if a.dry_run else VisualYawEstimator(cam), cfg=cfg)
    return loop, worker, camera


def _path_text(rec):
    if not rec.get("tracking"):
        return "on path"
    return f"off path {np.degrees(rec.get('path_offset', 0.0)):+.0f} deg"


def show(rec, camera):
    import cv2

    frame, _ = camera.latest()
    if frame is None:
        return True
    img = frame.copy()
    txt = rec.get("degraded") or PHRASES.get(rec.get("action"), "(silent)")
    cv2.putText(img, f"{txt}  {_path_text(rec)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("PathSense", img)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def main():
    a = parse_args()
    loop, worker, camera = build(a)
    worker.start()
    time.sleep(0.2 if a.dry_run else 1.0)
    stats = {"ticks": 0, "degraded": 0, "lat": []}

    def on_tick(rec):
        stats["ticks"] += 1
        stats["degraded"] += "degraded" in rec
        if "tick_latency_s" in rec:
            stats["lat"].append(rec["tick_latency_s"])
        if stats["ticks"] % 10 == 0:
            d = worker.last_latency_s
            mn = np.min(rec["ranges"]) if "ranges" in rec else float("nan")
            print(f"t={stats['ticks'] / 10:5.1f}s depth={'-' if d is None else f'{d * 1e3:.0f}ms'} "
                  f"min {mn:.1f}m {_path_text(rec)} -> "
                  f"{rec.get('degraded') or PHRASES.get(rec.get('action'), '(silent)')}", flush=True)
        return show(rec, camera) if a.show else True

    try:
        loop.run(a.max_seconds, on_tick=on_tick)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        camera.close()
        lat = np.array(stats["lat"]) * 1e3
        if len(lat):
            print(f"\n{stats['ticks']} ticks, {stats['degraded']} degraded, loop latency "
                  f"p50 {np.percentile(lat, 50):.1f}ms p95 {np.percentile(lat, 95):.1f}ms")


if __name__ == "__main__":
    main()
