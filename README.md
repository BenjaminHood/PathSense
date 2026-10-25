# PathSense

Spoken guidance for a person walking indoors. A camera on their head (Meta glasses, a webcam or a recorded video) gives a view of what is ahead. A policy trained with PPO decides whether to say anything: "bear left", "bear right", "stop", "go straight", or nothing. The goal is to get the walker past obstacles and back onto the path they were already walking, using as few cues as possible. There is no map and no destination.

The same policy runs in simulation (training) and on the camera (live), because both sides produce the same `Observation` and use the same five action ids.

## Workflow

```
 OFFLINE: TRAIN                                   LIVE: GUIDE A WALKER
 ──────────────                                   ────────────────────

 PathSenseEnv (2D simulated walker)               Meta glasses ─HEVC─► Android app ─WebSocket─► relay.py
   16 rays ray-cast over the fan                  webcam / phone URL ──────────────────────────► run_live.py
   PathTracker offset, CueGate                    recorded video ──────────────────────────────► replay.py
        │  obs · reward   ▲ action 0–4                                     │ newest frame
        ▼                 │                                                 ▼
 PPO train() ── CategoricalPolicy                 PerceptionWorker (own thread)
        │                                           Depth Anything V2 → depth_to_ranges → 16 ranges
        ▼                                                                   │
 artifacts/nav_policy.pt ─────────────── load_policy() ──────────►  LiveLoop.step()
        │                                                                   │
        ▼                                                                   ▼
 evaluate / plots / animate                       phrase → phone TTS (relay) · laptop TTS (run_live)
                                                           · annotated.mp4 + log.csv (replay)

                    ┌───────────────── the contract (src/common/types.py) ─────────────────┐
                    │ Observation: 16 ranges ÷ 8 m · cos, sin(path_offset) · speed  = 19   │
                    │ Action:      0 SILENT · 1 LEFT · 2 RIGHT · 3 STOP · 4 STRAIGHT        │
                    └───────────────────────────────────────────────────────────────────────┘





                    
```

### 1. Train in simulation

`PathSenseEnv` ([src/envs/pathsense_env.py](src/envs/pathsense_env.py)) is a 20 m arena with 4–10 round obstacles and a simulated walker who:

- follows their own intended path, makes their own 30–90° turns and steers away from walls
- reacts to a cue after 1–4 steps and obeys it 70–100% of the time (randomised per episode, never observed)
- sometimes starts stopped, so the policy learns to restart them

Each step casts 16 rays over a 60° fan (8 m max). The control step is randomised to 0.08–0.15 s, and 30% of steps reuse the previous fan to mimic slow live depth. Rewards (in [src/envs/config.py](src/envs/config.py)) pay for progress along the walker's own path. They charge −0.03 per cue, −0.1 for repeating a cue, −0.5 for closing in on an obstacle, −10 for a collision and −0.03 per step stopped. An episode ends on a collision or after 300 steps.

```
python -m scripts.train_nav
```

PPO ([src/agents/ppo.py](src/agents/ppo.py)) trains a `CategoricalPolicy` (MLP 19 → 64 → 64 → 5 plus a value head) for 2M steps. It saves `artifacts/nav_policy.pt`, which holds the weights, the `PPOConfig` and the `EnvConfig` (the runtime reads `fov_deg` from it). The training history goes to `artifacts/nav_policy_history.json`.

### 2. Evaluate

```
python -m src.eval.evaluate    # Random, Silent, PPO greedy, PPO sampled: 1000 episodes each → evaluation.png
python -m src.eval.plots       # training curve, evaluation bars, trajectories
python -m src.eval.animate     # episode_greedy_s3.gif, episode_sampled_s{3,7,11}.gif
```

### 3. Set up the camera

```
python -m scripts.calibrate_camera photos/ --square 0.025     # checkerboard → configs/camera.json
python -m scripts.check_depth wall_2m.jpg --true 2.0          # predicted vs measured depth → --depth-scale
python -m scripts.compare_sim                                 # sim fan vs camera fan → artifacts/sim_vs_camera.png
pytest
```

`configs/camera.json` holds the intrinsics, camera height and pitch. If it is missing, the runtime falls back to a `--hfov 70` model. `--frame-fit auto|crop|scale` adapts the calibration to the stream's frame size. At startup the runtime warns if the policy's fan is wider than the camera can see.

### 4. Run live

All three entry points drive the same `PerceptionWorker` and `LiveLoop` ([src/runtime/loop.py](src/runtime/loop.py)).

**Glasses (main path).** The glasses stream HEVC to the Android app ([MetaGlasses/CameraAccess](MetaGlasses/CameraAccess)) through the Wearables Device Access Toolkit. The app shows a preview and forwards every packet over a WebSocket to the relay. The relay decodes the packets, runs the pipeline and sends each phrase back as `{"phrase": "..."}`. The app then speaks it with Android TextToSpeech, and a new cue cuts off the last one.

```
adb reverse tcp:8765 tcp:8765          # the app connects to ws://127.0.0.1:8765
python -m src.runtime.relay --camera configs/camera.json --depth-scale 1.0
```

The relay logs every tick to `artifacts/live/log.csv`. If the phone reconnects within `--resume-s` (10 s), the walk resumes with its path tracking intact.

**Laptop only.** A webcam index, a stream URL (for example `http://<phone-ip>:8080/video`) or a video file, with speech from `pyttsx3`:

```
python -m scripts.run_live --device 0 --show
python -m scripts.run_live --dry-run --max-seconds 5 --no-speech    # plumbing check, no hardware
```

**Replay a recording.** This runs offline on the video's own clock, so the result does not depend on how fast depth runs:

```
python -m src.runtime.replay --video glasses.mp4 --annotate    # → artifacts/replay/annotated.mp4 + log.csv
```

### What one live tick does

Depth runs on its own thread and is never waited on. Each `PerceptionWorker` pass does:

1. Depth Anything V2 Metric-Indoor-Small (cuda, mps or cpu) gives a depth map in metres.
2. Back-project it with the `CameraModel` and drop the floor and ceiling (keeping 0.15–1.9 m above the floor).
3. Take the 5th-percentile horizontal distance per ray, giving 16 ranges stamped with the frame's time.

The control loop ticks at a fixed rate (10 Hz in `run_live`, `--control-hz` in the relay, one tick per processed frame in replay). Each `LiveLoop.step()` does:

1. **Read frame and yaw:** get the newest frame and estimate the head turn from optical flow (`VisualYawEstimator`).
2. **PathTracker:** after a bear-left/right cue, add up how far the walker has turned off their path. Tracking stops after 2 s within 10° of the path, or after 15 s.
3. **Freshness gate:** if the frame is older than 0.3 s or the fan older than 0.5 s, say "stop" once and skip the policy until fresh data returns.
4. **Build the Observation:** the fan clipped to 8 m, the path offset, and the speed implied by the last cue.
5. **Policy:** sampled by default, or argmax with `--greedy`.
6. **Stop filter:** a policy "stop" only counts if something is closer than 1.5 m within ±15° ahead.
7. **Safety override:** any ray under 0.5 m forces "stop", whatever the policy says.
8. **CueGate:** at most one cue per 0.8 s and the same cue once per 2 s. "Stop" is never held back by other cues but only repeats after 1.5 s.
9. **Speak** the phrase. A spoken turn cue starts PathTracker tracking.

`PathTracker` ([src/common/path.py](src/common/path.py)) and `CueGate` ([src/guidance/cues.py](src/guidance/cues.py)) are the same classes inside the simulator, so the policy trains against the same tracking and rate limits it meets live.

## The contract

[src/common/types.py](src/common/types.py) is shared by the simulator and the camera pipeline:

- `ranges`: 16 rays, metres, clipped to 8 m (8 means clear). Index 0 is the walker's right in both the sim and `depth_to_ranges` (`INDEX0_IS_RIGHT`).
- `path_offset`: radians the walker has been turned off their own path by avoidance cues, positive to the left, 0 when no turn is in progress. It enters the network as cos and sin.
- `speed`: m/s.
- Actions are `Discrete(5)`. In the sim, `CUE_LEFT` turns the walker clockwise, which is their right, so `SIM_CUES_MIRRORED` speaks it as "bear right" and `CUE_RIGHT` as "bear left".

If a live camera can't compute something, it does not go in the `Observation`. `Pose` exists only in the sim, for reward and metrics.

## Install

```
pip install -e ".[all]"     # Python ≥ 3.10; pyttsx3 needs espeak-ng on Linux
```

Checkpoints, logs and figures go in `artifacts/`.
