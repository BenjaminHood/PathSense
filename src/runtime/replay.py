"""Run the live PathSense pipeline over recorded glasses video.

Drives the real PerceptionWorker + LiveLoop (depth, ranges, path tracking,
safety override, cue gate, speed model) one frame at a time on a clock taken
from the video timestamps, so results are deterministic and don't depend on
how fast the depth model runs.

  python -m src.runtime.replay --video glasses.mp4 --annotate
"""
import argparse
import csv
import os

import cv2
import numpy as np

from ..common.path import PathTracker
from ..common.types import MAX_RANGE, N_RANGES
from ..guidance.cues import CUE_STOP, CUE_STRAIGHT, PHRASES, STAY_SILENT
from ..perception.camera import CameraModel
from ..perception.goal import VisualYawEstimator
from ..perception.ranges import INDEX0_IS_RIGHT
from .loop import LiveLoop, PerceptionWorker, RuntimeConfig
from .policy import check_fan_coverage, load_policy


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


class ConstantSpeed:
    """Speed model for open-loop replay: the recorded wearer never heard the
    cues, so they kept walking whatever the policy said."""

    def __init__(self, walking_speed: float):
        self.speed = walking_speed

    def on_cue(self, action: int):
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
    ap.add_argument("--checkpoint", default="artifacts/nav_policy.pt")
    ap.add_argument("--camera", default="configs/camera.json", help="calibration file")
    ap.add_argument("--hfov", type=float, default=70.0, help="used if --camera file is missing")
    ap.add_argument("--frame-fit", choices=["auto", "crop", "scale"], default="auto",
                    help="how frames of another size relate to the calibration: crop (focal "
                         "length in px unchanged), scale (resized video), or auto (crop if "
                         "within 10%% of the calibration size, else scale)")
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=1.2, help="walking speed, m/s")
    ap.add_argument("--every", type=int, default=3, help="process every Nth frame")
    ap.add_argument("--no-yaw", action="store_true",
                    help="disable optical-flow yaw (the path offset then stays at 0)")
    ap.add_argument("--greedy", action="store_true",
                    help="always take the most likely action instead of sampling")
    ap.add_argument("--seed", type=int, default=0, help="action-sampling seed")
    ap.add_argument("--speed-model", choices=["cues", "constant"], default="cues",
                    help="cues: assume the wearer obeys stop/go (as live); constant: assume they "
                         "kept walking, as the recorded footage did")
    ap.add_argument("--out", default="artifacts/replay")
    ap.add_argument("--annotate", action="store_true")
    return ap.parse_args()


def load_cam(a, width, height):
    if os.path.exists(a.camera):
        return CameraModel.load(a.camera).fitted(width, height, a.frame_fit)
    print(f"{a.camera} not found; using an uncalibrated {a.hfov:.0f} deg model")
    return CameraModel.from_fov(width, height, hfov_deg=a.hfov)


FONT, AA = cv2.FONT_HERSHEY_DUPLEX, cv2.LINE_AA
WHITE, GREY, DIM = (255, 255, 255), (185, 185, 185), (110, 110, 110)          # BGR
GREEN, YELLOW, ORANGE, RED = (90, 205, 80), (60, 215, 245), (40, 150, 255), (70, 70, 235)
DEGRADED_TEXT = {"camera": "camera feed lost", "perception": "waiting for depth"}
SAID_HOLD_S = 2.0        # keep a spoken cue on screen this long so it can be read


def _shade(img, x0, y0, x1, y1, alpha=0.68):
    """Darken a box so text on top of it stays readable over any background."""
    roi = img[max(y0, 0):y1, max(x0, 0):x1]
    roi[:] = (roi * (1 - alpha)).astype(img.dtype)


class Annotator:
    """Draws a HUD over each replay frame.

    Top: the current cue as a big coloured headline with an icon, plus why it was
    chosen (safety override, cooldown, degraded mode), time/speed, and a dial
    showing how far cues have turned the wearer off their own path and which way
    it lies. Middle: the last spoken phrase, held for SAID_HOLD_S.
    Bottom: the obstacle fan, laid out as the wearer sees it (left on the left),
    coloured by distance, with rays outside the camera's view faded.
    """

    def __init__(self, width, height, cam_hfov_deg, fan_fov_deg, stop_range_m,
                 cam, depth_view):
        self.s = min(width, height) / 1000          # scale everything with resolution
        self.cam_half, self.fan_fov = cam_hfov_deg / 2, fan_fov_deg
        self.stop_range = stop_range_m
        self.said, self.said_t = "", -np.inf
        self.cam = cam
        self.depth_view = depth_view

    def __call__(self, frame, rec, said, t, k, depth):
        img = frame.copy()
        show_depth = depth is not None and self.depth_view != "off"
        if show_depth and self.depth_view == "blend":
            self._depth(img, depth)
        if said:
            self.said, self.said_t = said, t
        head_bottom = self._header(img, rec, t, k)
        fan_top = self._fan(img, rec.get("ranges"))
        if show_depth and self.depth_view == "inset":
            self._depth(img, depth, top=head_bottom)
        self._ray_lines(img, rec.get("ranges"), head_bottom, fan_top)
        if show_depth:
            self._put(img, "depth: red = near, blue = 8 m+",
                      (int(24 * self.s), head_bottom + int(30 * self.s)), 0.45, WHITE)
        if t - self.said_t <= SAID_HOLD_S:
            self._subtitle(img, self.said, fan_top)
        return img

    def _put(self, img, text, org, size, color=WHITE, bold=False, anchor="left", max_w=None):
        scale, thick = size * self.s, max(1, round((3 if bold else 1.6) * self.s))
        tw, th = cv2.getTextSize(text, FONT, scale, thick)[0]
        if max_w and tw > max_w:                     # shrink rather than run off the panel
            scale *= max_w / tw
            tw, th = cv2.getTextSize(text, FONT, scale, thick)[0]
        x = org[0] - (tw // 2 if anchor == "center" else tw if anchor == "right" else 0)
        cv2.putText(img, text, (int(x), int(org[1])), FONT, scale, color, thick, AA)
        return tw, th

    def _status(self, rec):
        """Headline, its colour, icon name and explanatory note lines for one tick."""
        action, notes = rec.get("action"), []
        if rec.get("degraded"):
            reason = DEGRADED_TEXT.get(rec["degraded"], rec["degraded"])
            return "PAUSED", ORANGE, "warn", [(f"Told to stop: {reason}", ORANGE)]
        if action is None or action == STAY_SILENT:
            head, color, icon = "NO NEW CUE", GREY, "dots"
            notes.append(("Policy stays quiet: keep following the last cue", GREY))
        else:
            phrase = PHRASES[action]
            head, icon = phrase.upper(), phrase
            color = RED if action == CUE_STOP else GREEN if action == CUE_STRAIGHT else YELLOW
            if not rec.get("spoke"):
                notes.append(("Not spoken: same cue was given too recently", GREY))
        policy = rec.get("policy_action")
        if rec.get("stop_filtered"):
            notes.insert(0, ("Policy said 'stop' - ignored: nothing close in the walking path",
                             GREY))
        elif policy is not None and policy != action and rec.get("ranges") is not None:
            wanted = PHRASES.get(policy, "stay quiet")
            notes.insert(0, (f"SAFETY STOP: obstacle {rec['ranges'].min():.2f} m away "
                             f"(policy wanted '{wanted}')", RED))
        return head, color, icon, notes

    def _header(self, img, rec, t, k,):
        s, w = self.s, img.shape[1]
        pad, r = int(24 * s), int(40 * s)
        head, color, icon, notes = self._status(rec)

        speed = rec.get("speed")
        info = f"t = {t:.1f} s    frame {k}"
        if speed is not None:
            info += f"    {'walking ' + format(speed, '.1f') + ' m/s' if speed > 0 else 'standing still'}"

        comp_r = int(70 * s)
        comp_cx = w - pad - comp_r - int(30 * s)
        text_w = comp_cx - comp_r - int(40 * s) - pad
        line = int(38 * s)
        y_info = pad + 2 * r + int(42 * s)
        bottom = max(y_info + line * len(notes), pad + 2 * comp_r + int(110 * s)) + pad
        _shade(img, 0, 0, w, bottom)

        self._icon(img, icon, (pad + r, pad + r), r, color)
        self._put(img, head, (pad + 2 * r + int(24 * s), pad + r + int(20 * s)), 1.5, color,
                  bold=True, max_w=text_w - 2 * r)
        self._put(img, info, (pad, y_info), 0.62, GREY, max_w=text_w)
        for i, (txt, c) in enumerate(notes):
            self._put(img, txt, (pad, y_info + line * (i + 1)), 0.62, c, max_w=text_w)
        self._path_dial(img, rec, (comp_cx, pad + int(36 * s) + comp_r), comp_r)

    def _icon(self, img, kind, c, r, color):
        s, (cx, cy) = self.s, c
        t = max(2, round(6 * s))
        cv2.circle(img, c, r, color, -1 if kind == "stop" else t, AA)
        pt = lambda dx, dy: (int(cx + dx * r), int(cy + dy * r))
        if kind == "stop":
            cv2.rectangle(img, pt(-0.55, -0.15), pt(0.55, 0.15), WHITE, -1)
        elif kind == "go straight":
            cv2.arrowedLine(img, pt(0, 0.55), pt(0, -0.6), color, t, AA, tipLength=0.4)
        elif kind in ("bear left", "bear right"):
            d = -1 if kind == "bear left" else 1
            cv2.arrowedLine(img, pt(-0.35 * d, 0.5), pt(0.5 * d, -0.45), color, t, AA, tipLength=0.4)
        elif kind == "warn":
            self._put(img, "!", (cx, cy + int(0.4 * r)), 1.4, color, bold=True, anchor="center")
        elif kind == "dots":
            for dx in (-0.4, 0, 0.4):
                cv2.circle(img, pt(dx, 0), max(2, int(0.1 * r)), color, -1, AA)

    def _path_dial(self, img, rec, c, r):
        """Top-down view: you at the centre, straight ahead is up, arrow points
        along the path you were walking before the last avoidance turn."""
        s, (cx, cy) = self.s, c
        cv2.circle(img, c, r, DIM, max(1, round(2 * s)), AA)
        cv2.line(img, (cx, cy - r), (cx, cy - r + int(12 * s)), GREY, max(1, round(2 * s)), AA)
        self._put(img, "ahead", (cx, cy - r - int(10 * s)), 0.45, GREY, anchor="center")
        cv2.circle(img, c, max(3, int(6 * s)), WHITE, -1, AA)
        y = cy + r + int(38 * s)
        offset = rec.get("path_offset")
        if offset is None:
            self._put(img, "path unknown", (cx, y), 0.55, GREY, anchor="center")
            return
        # offset > 0: walker is heading left of their path, so the path lies to the right
        bearing = -float(offset)                                # + = left, like the fan
        tip = (int(cx - 0.85 * r * np.sin(bearing)), int(cy - 0.85 * r * np.cos(bearing)))
        tracking = rec.get("tracking", False)
        cv2.arrowedLine(img, c, tip, YELLOW if tracking else GREEN, max(2, round(5 * s)), AA,
                        tipLength=0.3)
        deg = np.degrees(bearing)
        if not tracking or abs(deg) < 5:
            head, sub = "On your path", "no avoidance turn" if not tracking else "back on track"
        else:
            head, sub = f"Path {abs(deg):.0f} deg {'left' if deg > 0 else 'right'}", "turned off by a cue"
        self._put(img, head, (cx, y), 0.6, WHITE, anchor="center")
        self._put(img, sub, (cx, y + int(32 * s)), 0.5, GREY, anchor="center")

    def _dist_color(self, d):
        return RED if d < self.stop_range else ORANGE if d < 1.5 else YELLOW if d < 3.0 else GREEN

    def _fan(self, img, ranges):
        """Bar per depth ray, wearer's left on the left. Returns the panel's top y."""
        s, (h, w) = self.s, img.shape[:2]
        pad, bar_h = int(24 * s), int(170 * s)
        top = h - (2 * pad + bar_h + int(95 * s))
        _shade(img, 0, top, w, h)
        x0, x1 = pad + int(60 * s), w - pad
        base = top + pad + int(40 * s) + bar_h

        self._put(img, "Free space ahead (taller = more room)", (pad, top + pad + int(18 * s)), 0.6,
                  WHITE, max_w=(w - 2 * pad) * 0.6)
        self._put(img, "faded = outside camera view", (x1, top + pad + int(18 * s)), 0.45, GREY,
                  anchor="right")
        for m in range(2, int(MAX_RANGE) + 1, 2):
            y = int(base - m / MAX_RANGE * bar_h)
            cv2.line(img, (x0, y), (x1, y), DIM, 1, AA)
            self._put(img, f"{m}m", (x0 - int(8 * s), y + int(6 * s)), 0.4, GREY, anchor="right")

        if ranges is None:
            self._put(img, "no depth reading", ((x0 + x1) // 2, base - bar_h // 2), 0.7, ORANGE,
                      anchor="center")
        else:
            r = ranges[::-1] if INDEX0_IS_RIGHT else ranges          # screen order: left -> right
            az = np.linspace(-self.fan_fov / 2, self.fan_fov / 2, N_RANGES)
            slot = (x1 - x0) / N_RANGES
            for i, d in enumerate(r):
                color = self._dist_color(d)
                if abs(az[i]) > self.cam_half + 1e-6:        # copied from nearest visible ray
                    color = tuple(int(c * 0.4) for c in color)
                cv2.rectangle(img, (int(x0 + (i + 0.15) * slot), int(base - d / MAX_RANGE * bar_h)),
                              (int(x0 + (i + 0.85) * slot), base), color, -1)
            i = int(np.argmin(r))
            self._put(img, f"{r[i]:.1f} m", (int(x0 + (i + 0.5) * slot),
                                             int(base - r[i] / MAX_RANGE * bar_h - 10 * s)),
                      0.5, self._dist_color(r[i]), bold=True, anchor="center")
        cv2.line(img, (x0, base), (x1, base), GREY, max(1, round(2 * s)), AA)

        y = base + int(40 * s)
        self._put(img, "< LEFT", (x0, y), 0.55, WHITE)
        self._put(img, "RIGHT >", (x1, y), 0.55, WHITE, anchor="right")
        legend = [(RED, f"<{self.stop_range:g}m stop"), (ORANGE, "<1.5m"), (YELLOW, "<3m"),
                  (GREEN, "clear")]
        sw, gap = int(18 * s), int(26 * s)
        widths = [sw + int(8 * s) + cv2.getTextSize(t, FONT, 0.45 * s, 1)[0][0] for _, t in legend]
        x = (x0 + x1 - sum(widths) - gap * (len(legend) - 1)) // 2
        for (c, txt), wd in zip(legend, widths):
            cv2.rectangle(img, (x, y - sw), (x + sw, y), c, -1)
            self._put(img, txt, (x + sw + int(8 * s), y), 0.45, GREY)
            x += wd + gap
        return top

    def _subtitle(self, img, text, bottom):
        s, w = self.s, img.shape[1]
        msg = f'"{text}"'
        scale, thick = 1.1 * s, max(1, round(3 * s))
        tw, th = cv2.getTextSize(msg, FONT, scale, thick)[0]
        lw = cv2.getTextSize("SPOKEN", FONT, 0.45 * s, 1)[0][0]
        pad = int(18 * s)
        y1 = bottom - int(30 * s)
        y0 = y1 - th - 2 * pad - int(26 * s)
        x0, x1 = (w - tw) // 2 - pad, (w + tw) // 2 + pad
        _shade(img, x0, y0, x1, y1, alpha=0.75)
        self._put(img, "SPOKEN", ((w - lw) // 2, y0 + pad + int(10 * s)), 0.45, GREY)
        self._put(img, msg, (w // 2, y1 - pad), 1.1, YELLOW, bold=True, anchor="center")

    def _depth(self, img, depth, top=0):
        """Colour the metric depth map"""

        near = 1.0 - np.clip(depth / MAX_RANGE, 0.0, 1.0)
        colour = cv2.applyColorMap((near * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        h, w = img.shape[:2]
        if colour.shape[:2] != (h, w):
            colour = cv2.resize(colour, (w, h))
        if self.depth_view == "blend":
            cv2.addWeighted(colour, 0.45, img, 0.55, 0, dst=img)
        else:                                   # inset, top right under the header
            iw, ih = w // 3, h // 3
            y0, x0 = top + int(10 * self.s), w - iw - int(10 * self.s)
            img[y0:y0 + ih, x0:x0 + iw] = cv2.resize(colour, (iw, ih))

    def _ray_lines(self, img, ranges, top, bottom):
        """A line at the image column each ray samples, coloured by its range."""
        if ranges is None or self.cam is None:
            return
        half = np.radians(self.fan_fov / 2)
        # azimuth of each ray, + = right, in the same order as depth_to_ranges
        az = np.linspace(half, -half, N_RANGES) if INDEX0_IS_RIGHT else np.linspace(-half, half, N_RANGES)
        for d, a in zip(ranges, az):
            u = self.cam.cx + self.cam.fx * np.tan(a)
            if 0 <= u < img.shape[1]:
                cv2.line(img, (int(u), top), (int(u), bottom), self._dist_color(d),
                         max(1, round(2 * self.s)), AA)


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

    policy = load_policy(a.checkpoint, sample=not a.greedy, seed=a.seed)
    check_fan_coverage(cam, policy.fov_deg)
    # One control tick per processed frame, so the path tracker's timers use the real step.
    cfg = RuntimeConfig(control_hz=fps / a.every, walking_speed=a.speed, fov_deg=policy.fov_deg)
    worker = PerceptionWorker(camera, depth_model, cam, cfg.fov_deg, clock=clock)
    loop = LiveLoop(camera, policy, speaker, worker, PathTracker(),
                    yaw_estimator=None if a.no_yaw else VisualYawEstimator(cam),
                    cfg=cfg, clock=clock)
    if a.speed_model == "constant":
        loop.speed_model = ConstantSpeed(a.speed)

    os.makedirs(a.out, exist_ok=True)
    out = None
    annotator = Annotator(frame.shape[1], frame.shape[0], np.degrees(cam.hfov_rad),
                          cfg.fov_deg, cfg.stop_range_m)
    view = worker.get_latest_view()
    img = annotator(frame, rec, said, clock.t, k,
                    depth=None if view is None else view[1])
    stats = {"ticks": 0, "degraded": 0, "override": 0, "filtered": 0, "spoken": 0,
             "depth_ms": []}
    k = -1
    with open(os.path.join(a.out, "log.csv"), "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame", "t", "degraded", "policy_action", "action", "cue", "said",
                      "override", "stop_filtered", "min_range", "path_offset_deg", "tracking",
                      "speed", "depth_ms"]
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

            ranges, offset = rec.get("ranges"), rec.get("path_offset")
            filtered = bool(rec.get("stop_filtered"))
            override = (not filtered and
                        rec.get("policy_action", CUE_STOP) != rec.get("action", CUE_STOP))
            stats["ticks"] += 1
            stats["degraded"] += "degraded" in rec
            stats["override"] += override
            stats["filtered"] += filtered
            stats["spoken"] += bool(said)
            log.writerow(
                [k, f"{clock.t:.3f}", rec.get("degraded", ""), rec.get("policy_action", ""),
                 rec.get("action", ""), PHRASES.get(rec.get("action"), ""), said,
                 int(override), int(filtered), "" if ranges is None else f"{ranges.min():.2f}",
                 "" if offset is None else f"{np.degrees(offset):.1f}",
                 int(bool(rec.get("tracking"))),
                 f"{rec.get('speed', '')}",
                 "" if worker.last_latency_s is None else f"{worker.last_latency_s * 1e3:.0f}"]
                + ([""] * N_RANGES if ranges is None else [f"{r:.2f}" for r in ranges]))

            if a.annotate:
                img = annotator(frame, rec, said, clock.t, k)
                if out is None:
                    h, w = img.shape[:2]
                    out = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"),
                                          cv2.VideoWriter_fourcc(*"mp4v"), fps / a.every, (w, h))
                out.write(img)

    cap.release()
    if out is not None:
        out.release()
    lat = np.array(stats["depth_ms"])
    print(f"done: {k + 1} frames read, {stats['ticks']} ticks, {stats['degraded']} degraded, "
          f"{stats['override']} safety overrides, {stats['filtered']} stops filtered, "
          f"{stats['spoken']} cues spoken"
          + (f", depth p50 {np.percentile(lat, 50):.0f}ms" if len(lat) else "")
          + f" -> {a.out}/")


if __name__ == "__main__":
    main()
