"""Live PathSense server.
"""
import argparse
import asyncio
import csv
import json
import os
import threading
import time
import traceback
import cv2
import queue as _queue
import pyttsx3
import av
import numpy as np
import websockets
import win32com.client
import pythoncom

from ..common.path import PathTracker
from ..common.types import N_RANGES
from ..guidance.cues import CUE_STOP, PHRASES
from ..perception.camera import CameraModel
from ..perception.goal import VisualYawEstimator
from .loop import LiveLoop, PerceptionWorker, RuntimeConfig
from .policy import check_fan_coverage, load_policy
from .replay import Annotator


# track real time logs using time module
class RealClock:
    def __init__(self):
        self.t0 = time.monotonic()

    def __call__(self):
        return time.monotonic() - self.t0
    
class LocalVoice:
    """Speak cue phrase on the running device"""
    def __init__(self, rate=180):
        self._q = _queue.Queue(maxsize=4)
        self._rate = rate
        threading.Thread(target=self._run, daemon=True).start()
        
    def _run(self):
        pythoncom.CoInitialize()                       # COM must be set up per thread
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = 2                                 # SAPI scale is -10..10, not words/min
        while True:
            phrase = self._q.get()
            try:
                voice.Speak(phrase)                    # blocks until spoken, then loops
            except Exception as exc:
                print(f"[voice] {exc}", flush=True)

                
    def say(self, phrase):
        try: 
            self._q.put_nowait(phrase)
        except _queue.Full:
            pass      # dropping rather than slowing down the control
            

class StreamCamera:
    """Camera stand-in fed by the decoder"""

    def __init__(self):
        self._lock = threading.Lock()
        self.frame, self.t = None, 0.0
        self.count = 0

    def push(self, frame, t):
        with self._lock:
            self.frame, self.t, self.count = frame, t, self.count + 1

    def latest(self):
        with self._lock:
            return self.frame, self.t

    def close(self):
        pass


class LiveSpeaker:
    """Forwards phrases to the connected phone and keeps them for the log."""

    def __init__(self, aloop, out_q, voice = None):
        self.aloop, self.q, self.said = aloop, out_q, []
        self.voice = voice

    def say(self, text):
        self.said.append(text)
        if self.voice is not None:
            self.voice.say(text)    # never blocks : LocalVoice drops when its queue is full
        try:
            self.aloop.call_soon_threadsafe(self.q.put_nowait, text)
        except RuntimeError:               # event loop already closed: shutting down
            pass

    def pop(self):
        text, self.said = " | ".join(self.said), []
        return text

class LiveViewer:
    """Annotates the live frames from the control thread to the websocket"""

    def __init__(self, aloop):
        self.aloop, self.q = aloop, asyncio.Queue(maxsize=1)
        self.wanted = False

    def offer(self, jpeg: bytes):
        def put():
            if self.q.full():
                self.q.get_nowait() # drops the frames which phone did not recieved

            self.q.put_nowait(jpeg)
        try:
            self.aloop.call_soon_threadsafe(put)
        except RuntimeError:               # event loop already closed: shutting down
            pass


class SerialModel:
    """One inference at a time: a closed session's worker can still be
    mid-frame when the next session's worker starts."""

    def __init__(self, model):
        self.model, self._lock = model, threading.Lock()

    def __call__(self, x):
        with self._lock:
            return self.model(x)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--checkpoint", default="artifacts/ben_new/nav_policy.pt")
    ap.add_argument("--camera", default="configs/camera.json")
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--frame-fit", choices=["auto", "crop", "scale"], default="auto",
                    help="how stream frames relate to the calibration: crop, scale, or auto "
                         "(crop if within 10%% of the calibration size, else scale)")
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=1.2)
    ap.add_argument("--control-hz", type=float, default=6.0)
    ap.add_argument("--resume-s", type=float, default=10.0,
                    help="a reconnect within this many seconds continues the walk "
                         "(keeps any avoidance turn in progress); later ones start fresh")
    ap.add_argument("--no-yaw", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/live")
    ap.add_argument("--view-fps", type=float, default=5.0, help="depth-view frames per second sent to the phone")
    ap.add_argument("--depth-view", choices=["blend", "inset", "off"], default="blend")
    ap.add_argument("--show", action="store_true", help="also show how the depth view in a computer window")
    ap.add_argument("--speak-local", action = "store_true", help= "also speak cues through the laptop's speakers")
    return ap.parse_args()


def load_models(a):
    """Depth model + policy; nothing here depends on the frame size."""
    from ..perception.depth import DepthAnythingMetric

    print("loading depth model ...", flush=True)
    depth_model = DepthAnythingMetric(scale=a.depth_scale)
    depth_model(np.zeros((480, 640, 3), np.uint8))     # warm-up: first call is slow
    policy = load_policy(a.checkpoint, sample=not a.greedy, seed=a.seed)
    return SerialModel(depth_model), policy


class Session:
    """One stream of w x h frames from the phone: camera model, perception
    worker and control loop."""

    def __init__(self, a, models, width, height, tracker, clock, speaker):
        depth_model, policy = models
        if os.path.exists(a.camera):
            cam = CameraModel.load(a.camera).fitted(width, height, a.frame_fit)
        else:
            print(f"{a.camera} not found; using an uncalibrated {a.hfov:.0f} deg model")
            cam = CameraModel.from_fov(width, height, hfov_deg=a.hfov)
        print(f"[relay] camera model: {np.degrees(cam.hfov_rad):.0f} deg horizontal")
        check_fan_coverage(cam, policy.fov_deg)

        cfg = RuntimeConfig(control_hz=a.control_hz, walking_speed=a.speed,
                            fov_deg=policy.fov_deg)

        self.size = (width, height)
        self.camera, self.speaker = StreamCamera(), speaker
        self.worker = PerceptionWorker(self.camera, depth_model, cam, cfg.fov_deg, clock=clock)
        self.loop = LiveLoop(self.camera, policy, speaker, self.worker, tracker,
                             yaw_estimator=None if a.no_yaw else VisualYawEstimator(cam),
                             cfg=cfg, clock=clock)
        self.worker.start()

    def close(self):
        self.worker.stop()


class RelayState:
    def __init__(self):
        self.session = None # what the control thread steps
        self.ws = None # the connection feeding it
        self.tracker = None # path tracker left by the last session, for --resume-s
        self.ended_at = -np.inf


LOG_HEADER = (["frame", "t", "degraded", "policy_action", "action", "cue", "said",
               "override", "stop_filtered", "min_range", "path_offset_deg", "tracking",
               "speed", "depth_ms"]
              + [f"r{i}" for i in range(N_RANGES)])      # same columns as replay.py


def control_thread(state, a, clock, stop_evt):
    """Steps the active session's LiveLoop at control_hz and logs every tick."""
    period = 1.0 / a.control_hz
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "log.csv"), "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(LOG_HEADER)
        f.flush()
        k, failing = 0, None
        next_t = time.monotonic()
        while not stop_evt.is_set():
            sess = state.session
            if sess is not None:
                try:
                    _tick(sess, log, clock, k)
                    failing = None
                except Exception as e:
                    if failing is None:
                        # A dead control loop means silence, which the walker
                        # hears as "carry on". Say stop once, then keep retrying.
                        sess.speaker.say(PHRASES[CUE_STOP])
                    if repr(e) != failing:
                        traceback.print_exc()
                    failing = repr(e)
                f.flush()
                k += 1

            next_t += period
            delay = next_t - time.monotonic()
            if delay < -period:            # stalled: resync rather than burst ticks
                print(f"[relay] control loop stalled {-delay * 1e3:.0f} ms")
                next_t, delay = time.monotonic(), 0.0
            stop_evt.wait(max(0.0, delay))


def _tick(sess, log, clock, k):
    rec = sess.loop.step()
    worker = sess.worker
    if worker.last_error:
        print(f"perception error {worker.last_error}")
        worker.last_error = None
    said = sess.speaker.pop()

    ranges, offset = rec.get("ranges"), rec.get("path_offset")
    filtered = bool(rec.get("stop_filtered"))
    override = (not filtered and
                rec.get("policy_action", CUE_STOP) != rec.get("action", CUE_STOP))
    ms = worker.last_latency_s
    log.writerow(
        [sess.camera.count, f"{clock():.3f}", rec.get("degraded", ""),
         rec.get("policy_action", ""), rec.get("action", ""),
         PHRASES.get(rec.get("action"), ""), said, int(override), int(filtered),
         "" if ranges is None else f"{ranges.min():.2f}",
         "" if offset is None else f"{np.degrees(offset):.1f}",
         int(bool(rec.get("tracking"))),
         f"{rec.get('speed', '')}",
         "" if ms is None else f"{ms * 1e3:.0f}"]
        + ([""] * N_RANGES if ranges is None else [f"{r:.2f}" for r in ranges]))
    if k % 5 == 0:
        mn = "--" if ranges is None else f"{ranges.min():4.1f}"
        off = "" if offset is None or not rec.get("tracking") else f" | off path {np.degrees(offset):+.0f} deg"
        print(f"{k:5d} | {('%.0f' % (ms * 1e3)) if ms else '--':>4}ms | "
              f"min {mn} m | {rec.get('degraded') or PHRASES.get(rec.get('action'), 'silent')}{off}")


async def serve(a, clock, state):
    models = asyncio.ensure_future(asyncio.to_thread(load_models, a))
    voice = LocalVoice() if a.speak_local else None

    def end_session():
        sess, state.session = state.session, None
        if sess is not None:
            sess.close()
            state.tracker = sess.loop.path
            state.ended_at = clock()

    def open_session(width, height, speaker):
        end_session()
        resume = state.tracker is not None and clock() - state.ended_at <= a.resume_s
        tracker = state.tracker if resume else PathTracker()
        print(f"[relay] {width}x{height}, " + ("resuming walk" if resume else "new walk"))
        return Session(a, models.result(), width, height, tracker, clock, speaker)

    async def handler(ws):
        if state.ws is not None:
            print("[relay] new connection replaces the previous one")
            end_session()
            asyncio.ensure_future(state.ws.close(1000, "replaced by a new connection"))
        state.ws = ws
        print(f"[relay] phone connected from {ws.remote_address}")

        out_q = asyncio.Queue()
        speaker = LiveSpeaker(asyncio.get_running_loop(), out_q, voice=voice)
        codec = av.CodecContext.create("hevc", "r")
        sess, n_frames, n_bad = None, 0, 0

        async def sender():
            try:
                while True:
                    phrase = await out_q.get()
                    await ws.send(json.dumps({"phrase": phrase}))
                    print(f"  >> {phrase}")
            except websockets.ConnectionClosed:
                pass

        send_task = asyncio.create_task(sender())
        try:
            async for message in ws:
                if state.ws is not ws:             # a newer connection took over
                    break
                if isinstance(message, str):
                    continue
                # Always decode, even before the models are ready, so the
                # decoder sees the stream's parameter sets.
                try:
                    frames = [fr for p in codec.parse(message) for fr in codec.decode(p)]
                except av.error.FFmpegError as e:
                    n_bad += 1
                    if n_bad <= 3 or n_bad % 100 == 0:
                        print(f"[relay] dropped undecodable packet ({n_bad} so far): {e}")
                    continue
                if not frames or not models.done():
                    continue
                for frame in frames:
                    img = frame.to_ndarray(format="bgr24")
                    n_frames += 1
                    h, w = img.shape[:2]
                    if sess is None or sess.size != (w, h):
                        sess = open_session(w, h, speaker)
                    # Stamped on arrival: the phone sends no capture time, so
                    # network delay is not counted in the frame's age.
                    sess.camera.push(img, clock())
                    state.session = sess
        finally:
            send_task.cancel()
            if state.ws is ws:
                state.ws = None
                end_session()
            print(f"[relay] disconnected after {n_frames} frames"
                  + (f", {n_bad} undecodable packets" if n_bad else ""))

    async with websockets.serve(handler, "0.0.0.0", a.port, max_size=4_000_000):
        print(f"[relay] listening on 0.0.0.0:{a.port}")
        await models                       # raises (and shuts down) if loading failed
        print("[relay] models ready")
        await asyncio.Future()


def main():
    a = parse_args()
    clock, state, stop_evt = RealClock(), RelayState(), threading.Event()
    ctrl = threading.Thread(target=control_thread, args=(state, a, clock, stop_evt),
                            daemon=True)
    ctrl.start()
    try:
        asyncio.run(serve(a, clock, state))
    except KeyboardInterrupt:
        print("\n[relay] stopped")
    finally:
        stop_evt.set()
        ctrl.join(timeout=2.0)             # lets the log close cleanly
        if state.session is not None:
            state.session.close()


if __name__ == "__main__":
    main()
