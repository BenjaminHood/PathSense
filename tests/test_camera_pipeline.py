
import numpy as np
import pytest

from src.common.types import MAX_RANGE, N_RANGES
from src.envs import pathsense_env as env_mod
from src.envs.config import EnvConfig
from src.envs.pathsense_env import PathSenseEnv
from src.guidance import cues
from src.guidance.cues import CueGate, action_to_guidance
from src.perception.camera import CameraModel
from src.common.path import PathTracker
from src.perception.ranges import depth_to_ranges
from src.runtime.camera import StaticCamera
from src.runtime.loop import LiveLoop, PerceptionWorker, RuntimeConfig
from tests.synthetic import render_depth

FOV = EnvConfig().fov_deg
WIDE = CameraModel.from_fov(1280, 360, hfov_deg=125, height_m=1.6)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- sim equivalence

def _random_scene(seed):
    """Env obstacles near the walker, in the agent frame, walls pushed far away."""
    env = PathSenseEnv(EnvConfig(arena_size=1000.0))
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    env.pos, env.heading = np.array([500.0, 500.0]), rng.uniform(-np.pi, np.pi)
    scene = [(rng.uniform(1.0, 7.0), rng.uniform(-5, 5), rng.uniform(0.3, 0.9)) for _ in range(5)]
    c, s = np.cos(env.heading), np.sin(env.heading)
    env.obstacles = [np.array([500 + f * c - l * s, 500 + f * s + l * c, r]) for f, l, r in scene]
    return env, scene


def _expected(env, window_frac=0.3, n=21):
    """Nearest env hit within each ray's window — what depth_to_ranges measures."""
    half = np.deg2rad(FOV) / 2
    centres = np.linspace(-half, half, N_RANGES)
    w = window_frac * (centres[1] - centres[0]) / 2
    out = []
    for c in centres:
        best = MAX_RANGE
        for off in np.linspace(c - w, c + w, n):
            d = np.array([np.cos(env.heading + off), np.sin(env.heading + off)])
            for o in env.obstacles:
                h = env_mod._ray_circle_hit(env.pos, d, o[:2], o[2])
                if h is not None:
                    best = min(best, h)
        out.append(best)
    return np.array(out)


@pytest.mark.parametrize("seed", range(6))
def test_camera_ranges_equal_env_ranges_random_scenes(seed):
    env, scene = _random_scene(seed)
    sim = env._build_observation().ranges
    got = depth_to_ranges(render_depth(WIDE, cylinders=scene), WIDE, FOV, stride=1)
    assert np.all(got <= sim * 1.02 + 0.05), (got, sim)   # never less cautious than the sim
    np.testing.assert_allclose(got, _expected(env), atol=0.05, rtol=0.02)


# ---------------------------------------------------------------- guidance

def test_action_ids_match_env():
    for name in ("STAY_SILENT", "CUE_LEFT", "CUE_RIGHT", "CUE_STOP", "CUE_STRAIGHT",
                 "N_ACTIONS"):
        assert getattr(cues, name) == getattr(env_mod, name), name
    assert np.isclose(cues.TURN_STEP, np.deg2rad(EnvConfig().turn_deg))


def test_spoken_words_match_what_the_sim_action_does():
    """Under +y = left, which way does each cue actually turn the walker?"""
    env = PathSenseEnv(EnvConfig())
    env.reset(seed=0)
    for action in (cues.CUE_LEFT, cues.CUE_RIGHT):
        env.heading = 0.0
        env._apply_cue(action)
        turned_left = env.heading > 0                        # +heading = towards +y = left
        assert cues.PHRASES[action] == ("bear left" if turned_left else "bear right")
        assert (action_to_guidance(action).turn < 0) == turned_left   # contract: + = right


def test_cue_gate_rate_limits_but_never_blocks_stop():
    clk = FakeClock()
    gate = CueGate(same_cue_cooldown_s=2.0, any_cue_cooldown_s=0.8, clock=clk)
    assert gate.allow(cues.CUE_LEFT)
    clk.t += 0.1
    assert not gate.allow(cues.CUE_RIGHT)
    assert gate.allow(cues.CUE_STOP)
    clk.t += 1.0
    assert not gate.allow(cues.CUE_LEFT)
    assert gate.allow(cues.CUE_RIGHT)


def test_cue_gate_throttles_repeated_stop_only():
    clk = FakeClock()
    gate = CueGate(stop_repeat_s=1.5, clock=clk)
    assert gate.allow(cues.CUE_STOP)
    clk.t += 0.1
    assert not gate.allow(cues.CUE_STOP)           # already stopped: don't nag
    clk.t += 1.5
    assert gate.allow(cues.CUE_STOP)               # still blocked → repeat after the window
    clk.t += 1.0
    assert gate.allow(cues.CUE_STRAIGHT)
    clk.t += 0.1
    assert gate.allow(cues.CUE_STOP)               # stop after a different cue fires at once


# ---------------------------------------------------------------- path tracking

def test_path_tracker_ignores_own_turns_until_a_turn_cue():
    p = PathTracker()
    assert p.update(0.5, 0.1) == 0.0 and not p.active      # the wearer's own turn
    p.on_turn_cue()
    p.update(np.deg2rad(25), 0.1)                          # the cue lands: 25 deg left
    assert p.active and np.isclose(p.offset, np.deg2rad(25))


def test_path_tracker_settles_back_on_path_and_gives_up():
    p = PathTracker(on_path_deg=10, settle_s=2.0, max_s=15.0)
    p.on_turn_cue()
    p.update(np.deg2rad(25), 0.1)
    p.update(np.deg2rad(-22), 0.1)                         # steered back to within 10 deg
    for _ in range(20):
        p.update(0.0, 0.1)
    assert not p.active and p.offset == 0.0                # 2 s back on path: done
    p.on_turn_cue()
    p.update(np.deg2rad(40), 0.1)
    for _ in range(150):
        p.update(0.0, 0.1)
    assert not p.active                                    # 15 s off path: their choice now


def test_crop_vs_scale_auto():
    cam = CameraModel(1230.0, 1230.0, 736.0, 984.0, 1472, 1968)
    assert cam.fitted(1408, 1872).fx == 1230.0             # small change: the glasses crop
    assert np.isclose(cam.fitted(504, 896).hfov_rad, cam.hfov_rad)   # 1/3 width: downscaled


# ---------------------------------------------------------------- live loop

class ListSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


def make_loop(policy_action=cues.CUE_RIGHT, cylinders=(), yaw=None):
    clk = FakeClock()
    cam = CameraModel.from_fov(160, 120, hfov_deg=125)
    depth = render_depth(cam, cylinders=cylinders)
    camera = StaticCamera(np.zeros((120, 160, 3), np.uint8), clock=clk)
    worker = PerceptionWorker(camera, lambda rgb: depth, cam, FOV, clock=clk)
    seen = []

    def policy(x):
        seen.append(x)
        return policy_action

    spk = ListSpeaker()
    loop = LiveLoop(camera, policy, spk, worker, PathTracker(), yaw_estimator=yaw,
                    cfg=RuntimeConfig(fov_deg=FOV), clock=clk)
    return loop, worker, camera, clk, spk, seen


def test_stop_filter_ignores_stop_in_open_space():
    loop, worker, *_, spk, _ = make_loop(policy_action=cues.CUE_STOP)
    worker.run_once()
    rec = loop.step()
    assert rec["stop_filtered"] and rec["action"] == cues.STAY_SILENT and spk.said == []


def test_stop_filter_keeps_stop_with_obstacle_ahead():
    loop, worker, *_, spk, _ = make_loop(policy_action=cues.CUE_STOP, cylinders=[(1.3, 0.0, 0.3)])
    worker.run_once()
    rec = loop.step()
    assert "stop_filtered" not in rec and rec["action"] == cues.CUE_STOP and spk.said == ["stop"]


def test_loop_normal_tick_speaks_policy_cue():
    loop, worker, _, _, spk, seen = make_loop()
    worker.run_once()
    rec = loop.step()
    assert rec["action"] == cues.CUE_RIGHT and rec["spoke"]
    assert spk.said == [cues.PHRASES[cues.CUE_RIGHT]]
    assert seen[0].shape == (N_RANGES + 3,)
    assert seen[0][-3] == 1.0 and seen[0][-2] == 0.0       # on path: (cos 0, sin 0)


def test_loop_tracks_how_far_a_turn_cue_takes_the_wearer():
    turns = iter([0.0, np.deg2rad(25), 0.0])               # optical-flow yaw per fresh frame
    loop, worker, _, clk, spk, seen = make_loop(yaw=lambda frame: next(turns, 0.0))
    worker.run_once()
    assert loop.step()["tracking"]                         # spoken turn cue starts tracking
    clk.t += 0.1
    worker.run_once()
    rec = loop.step()                                      # the wearer turned 25 deg left
    assert np.isclose(rec["path_offset"], np.deg2rad(25))
    assert np.isclose(seen[-1][-2], np.sin(np.deg2rad(25)))   # and the policy saw it


def test_loop_safety_override_stops_when_too_close():
    loop, worker, *_, spk, _ = make_loop(cylinders=[(0.6, 0.0, 0.3)])
    worker.run_once()
    rec = loop.step()
    assert rec["policy_action"] == cues.CUE_RIGHT and rec["action"] == cues.CUE_STOP
    assert spk.said == ["stop"]


def test_loop_degrades_when_camera_freezes_then_recovers():
    loop, worker, camera, clk, spk, _ = make_loop()
    worker.run_once()
    loop.step()
    camera.frozen_at = clk.t
    clk.t += 1.0
    assert loop.step()["degraded"] == "camera"
    assert spk.said[-1] == "stop" and loop.speed_model.speed == 0.0
    n = len(spk.said)
    clk.t += 0.1
    loop.step()
    assert len(spk.said) == n                      # "stop" once, not every tick
    camera.frozen_at = None
    clk.t += 3.0
    worker.run_once()
    rec = loop.step()
    assert rec.get("recovered_from") == "camera" and "action" in rec


def test_loop_degrades_when_depth_model_fails():
    loop, worker, camera, clk, *_ = make_loop()

    def boom(rgb):
        raise RuntimeError("cuda oom")
    worker.depth_model = boom
    assert not worker.run_once() and "cuda oom" in worker.last_error
    assert loop.step()["degraded"] == "perception"
