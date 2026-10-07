
import argparse
import os
import time

import numpy as np

from src.guidance.cues import PHRASES
from src.guidance.tts import make_speaker
from src.perception.camera import CameraModel
from src.perception.goal import GoalTracker, MarkerGoal, VisualYawEstimator
from src.runtime.loop import LiveLoop, PerceptionWorker, RuntimeConfig
from src.runtime.policy import load_greedy_policy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="configs/camera.json", help="calibration (Step 2)")
    ap.add_argument("--hfov", type=float, default=70.0, help="used if --camera file is missing")
    ap.add_argument("--device", default="0", help="webcam index, video file or stream URL")
    ap.add_argument("--checkpoint", default="artifacts/policy.pt")
    ap.add_argument("--goal", default=None, help="initial goal 'x,y' metres (fwd, left)")
    ap.add_argument("--marker-id", type=int, default=None)
    ap.add_argument("--marker-size", type=float, default=0.15)
    ap.add_argument("--speed", type=float, default=1.2, help="walking speed, m/s")
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def load_cam(a, width, height):
    if os.path.exists(a.camera):
        return CameraModel.load(a.camera).resized(width, height)
    print(f"{a.camera} not found; using an uncalibrated {a.hfov:.0f} deg model")
    return CameraModel.from_fov(width, height, hfov_deg=a.hfov)


def build(a):
    from src.runtime.camera import StaticCamera, ThreadedCamera

    if a.dry_run:
        from tests.synthetic import render_depth

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

    marker = MarkerGoal(cam, a.marker_id, a.marker_size) if a.marker_id is not None else None
    goal0 = [float(v) for v in a.goal.split(",")] if a.goal else None
    if goal0 is None and marker is None:
        raise SystemExit("need --goal and/or --marker-id")

    cfg = RuntimeConfig(walking_speed=a.speed)
    worker = PerceptionWorker(camera, depth_model, cam, cfg.fov_deg)
    loop = LiveLoop(camera, load_greedy_policy(a.checkpoint), make_speaker(not a.no_speech),
                    worker, GoalTracker(goal0, marker),
                    yaw_estimator=None if a.dry_run else VisualYawEstimator(cam), cfg=cfg)
    return loop, worker, camera


def show(rec, camera):
    import cv2

    frame, _ = camera.latest()
    if frame is None:
        return True
    img = frame.copy()
    txt = rec.get("degraded") or PHRASES.get(rec.get("action"), "(silent)")
    cv2.putText(img, f"{txt}  goal={np.round(rec.get('goal', []), 1)}", (10, 25),
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
                  f"min {mn:.1f}m goal {np.round(rec.get('goal', [np.nan] * 2), 1)} -> "
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
