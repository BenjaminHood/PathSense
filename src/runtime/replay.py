"""Run the live PathSense pipeline over recorded glasses video.

Drives the real PerceptionWorker + LiveLoop (depth, ranges, goal tracking,
safety override, cue gate, speed model) one frame at a time on a clock taken
from the video timestamps, so results are deterministic and don't depend on
how fast the depth model runs.

  python -m src.runtime.replay --video glasses.mp4 --goal 10,0 --annotate
"""
import argparse
import csv
import os

import cv2
import numpy as np

from ..common.types import MAX_RANGE, N_RANGES
from ..guidance.cues import CUE_STOP, PHRASES
from ..perception.camera import CameraModel
from ..perception.goal import GoalTracker, MarkerGoal, VisualYawEstimator
from .loop import LiveLoop, PerceptionWorker, RuntimeConfig
from .policy import load_greedy_policy


class SimClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class VideoFrameCamera:
    """Camera stand-in: serves whichever video frame the replay last pushed."""

    def __init__(self):
        self.frame, self.t = None, 0.0

    def push(self, frame, t):
        self.frame, self.t = frame, t

    def latest(self):
        return self.frame, self.t

    def close(self):
        pass


class RecordingSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text: str):
        self.said.append(text)

    def pop(self) -> str:
        text, self.said = " | ".join(self.said), []
        return text


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", default="artifacts/policy.pt")
    ap.add_argument("--camera", default="configs/camera.json", help="calibration file")
    ap.add_argument("--hfov", type=float, default=70.0, help="used if --camera file is missing")
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("--goal", default="10,0", help="initial goal 'x,y' metres (fwd, left)")
    ap.add_argument("--marker-id", type=int, default=None)
    ap.add_argument("--marker-size", type=float, default=0.15)
    ap.add_argument("--speed", type=float, default=1.2, help="walking speed, m/s")
    ap.add_argument("--every", type=int, default=3, help="process every Nth frame")
    ap.add_argument("--no-yaw", action="store_true", help="disable optical-flow yaw")
    ap.add_argument("--out", default="artifacts/replay")
    ap.add_argument("--annotate", action="store_true")
    return ap.parse_args()


def load_cam(a, width, height):
    if os.path.exists(a.camera):
        return CameraModel.load(a.camera).resized(width, height)
    print(f"{a.camera} not found; using an uncalibrated {a.hfov:.0f} deg model")
    return CameraModel.from_fov(width, height, hfov_deg=a.hfov)


def annotate(frame, rec, text):
    img = frame.copy()
    h, w = img.shape[:2]
    ranges = rec.get("ranges")
    if ranges is not None:
        for i, r in enumerate(ranges):
            x = int(w * (i + 0.5) / N_RANGES)
            bar = int((r / MAX_RANGE) * 120)
            cv2.line(img, (x, h - 20), (x, h - 20 - bar), (214, 120, 42), 6)
    goal = rec.get("goal")
    status = rec.get("degraded") or ("arrived" if rec.get("arrived") else
                                     PHRASES.get(rec.get("action"), "silent"))
    cv2.putText(img, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    if text:
        cv2.putText(img, f"said: {text}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 220, 255), 2)
    if goal is not None:
        cv2.putText(img, f"goal {np.round(goal, 1)}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main():
    a = parse_args()
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"no frames in {a.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    from ..perception.depth import DepthAnythingMetric

    cam = load_cam(a, frame.shape[1], frame.shape[0])
    print("loading depth model ...", flush=True)
    depth_model = DepthAnythingMetric(scale=a.depth_scale)

    clock, camera, speaker = SimClock(), VideoFrameCamera(), RecordingSpeaker()
    marker = MarkerGoal(cam, a.marker_id, a.marker_size) if a.marker_id is not None else None
    goal0 = [float(v) for v in a.goal.split(",")] if a.goal else None
    if goal0 is None and marker is None:
        raise SystemExit("need --goal and/or --marker-id")

    # One control tick per processed frame, so dead-reckoning uses the real time step.
    cfg = RuntimeConfig(control_hz=fps / a.every, walking_speed=a.speed)
    worker = PerceptionWorker(camera, depth_model, cam, cfg.fov_deg, clock=clock)
    loop = LiveLoop(camera, load_greedy_policy(a.checkpoint), speaker, worker,
                    GoalTracker(goal0, marker),
                    yaw_estimator=None if a.no_yaw else VisualYawEstimator(cam),
                    cfg=cfg, clock=clock)

    os.makedirs(a.out, exist_ok=True)
    out = None
    stats = {"ticks": 0, "degraded": 0, "override": 0, "spoken": 0, "depth_ms": []}
    k = -1
    with open(os.path.join(a.out, "log.csv"), "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "t", "degraded", "policy_action", "action", "cue", "said",
                      "override", "min_range", "goal_x", "goal_y", "speed", "depth_ms"]
                     + [f"r{i}" for i in range(N_RANGES)])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            k += 1
            if k % a.every:
                continue

            clock.t = k / fps
            camera.push(frame, clock.t)
            if worker.run_once():
                stats["depth_ms"].append(worker.last_latency_s * 1e3)
            elif worker.last_error:
                print(f"frame {k}: perception error {worker.last_error}")
                worker.last_error = None
            rec = loop.step()
            said = speaker.pop()

            ranges, goal = rec.get("ranges"), rec.get("goal")
            override = rec.get("policy_action", CUE_STOP) != rec.get("action", CUE_STOP)
            stats["ticks"] += 1
            stats["degraded"] += "degraded" in rec
            stats["override"] += override
            stats["spoken"] += bool(said)
            log.writerow(
                [k, f"{clock.t:.3f}", rec.get("degraded", ""), rec.get("policy_action", ""),
                 rec.get("action", ""), PHRASES.get(rec.get("action"), ""), said,
                 int(override), "" if ranges is None else f"{ranges.min():.2f}",
                 *(["", ""] if goal is None else [f"{goal[0]:.2f}", f"{goal[1]:.2f}"]),
                 f"{rec.get('speed', '')}",
                 "" if worker.last_latency_s is None else f"{worker.last_latency_s * 1e3:.0f}"]
                + ([""] * N_RANGES if ranges is None else [f"{r:.2f}" for r in ranges]))

            if a.annotate:
                img = annotate(frame, rec, said)
                if out is None:
                    h, w = img.shape[:2]
                    out = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"),
                                          cv2.VideoWriter_fourcc(*"mp4v"), fps / a.every, (w, h))
                out.write(img)

            if rec.get("arrived"):
                print(f"arrived at frame {k} (t={clock.t:.1f}s); "
                      "pass a farther --goal to replay the whole clip")
                break

    cap.release()
    if out is not None:
        out.release()
    lat = np.array(stats["depth_ms"])
    print(f"done: {k + 1} frames read, {stats['ticks']} ticks, {stats['degraded']} degraded, "
          f"{stats['override']} safety overrides, {stats['spoken']} cues spoken"
          + (f", depth p50 {np.percentile(lat, 50):.0f}ms" if len(lat) else "")
          + f" -> {a.out}/")


if __name__ == "__main__":
    main()
