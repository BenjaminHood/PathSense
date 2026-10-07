# Camera pipeline (Track C)

Camera frame → `Observation` → policy → spoken cue, in real time.

```
ThreadedCamera ──newest frame──┬──► PerceptionWorker (own thread)
  (webcam / phone URL)         │      DepthAnythingMetric → metres
                               │      depth_to_ranges(depth, CameraModel, 120°)
                               │
                               └──► LiveLoop.step(), every 100 ms
                                      VisualYawEstimator (optical flow)
                                      GoalTracker (MarkerGoal fix + dead reckoning)
                                      Observation → policy → safety override (< 0.5 m → stop)
                                      PHRASES → CueGate → Speaker (async)
```

## Steps

1. `pip install -r requirements-runtime.txt`
2. Calibrate: `python -m scripts.calibrate_camera photos/ --square 0.025` → `configs/camera.json`
3. Check depth against a tape measure: `python -m scripts.check_depth wall_2m.jpg --true 2.0`
4. Check sim vs camera: `python -m scripts.compare_sim` → `artifacts/sim_vs_camera.png`
5. Plumbing without hardware: `python -m scripts.run_live --dry-run --goal 6,0 --max-seconds 5 --no-speech`
6. Live: `python -m scripts.run_live --goal 10,0 --show` (add `--marker-id 0` with a printed
   DICT_4X4_50 marker at the goal; `--device http://<phone-ip>:8080/video` for a phone)

## Files

| File | What |
|---|---|
| `src/perception/camera.py` | `CameraModel`: intrinsics, height, pitch; load/save calibration |
| `src/perception/depth.py` | Depth Anything V2 metric-indoor (metres) |
| `src/perception/ranges.py` | depth → 16 ranges: back-project, drop floor/ceiling, thin window per ray |
| `src/perception/goal.py` | `MarkerGoal` (ArUco), `VisualYawEstimator`, `GoalTracker` |
| `src/perception/observe.py` | `observation_from_frame` — the contract function |
| `src/guidance/cues.py` | action ids, `PHRASES`, `action_to_guidance`, `CueGate` |
| `src/guidance/tts.py` | non-blocking speech; keeps only the newest cue |
| `src/runtime/camera.py` | threaded newest-frame capture (`ThreadedCamera`), `StaticCamera` for tests |
| `src/runtime/loop.py` | 10 Hz control loop, latency budget, degraded mode |
| `src/runtime/policy.py` | loads `artifacts/policy.pt` as a greedy policy |

## Timing and degradation

The policy learned at dt = 0.1 s, so the loop ticks at 10 Hz and never waits for depth.
Depth runs on its own thread; the loop uses the newest range fan.

- frame older than 300 ms → degraded ("camera")
- range fan older than 500 ms, or the depth model throws → degraded ("perception")
- no goal yet → degraded ("no_goal")

Degraded mode says "stop" once, treats the user as stopped, never feeds stale data to the
policy, and resumes automatically. Separately, any range under 0.5 m forces "stop".

