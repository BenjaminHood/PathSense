import numpy as np
import pytest

from src.common.types import N_RANGES, MAX_RANGE
from src.perception.camera import CameraModel
from src.perception.ranges import depth_to_ranges
from src.envs.config import EnvConfig
from src.envs.pathsense_env import PathSenseEnv
from tests.synthetic import render_depth

FOV = 120.0
# a very wide fake camera so every sim ray is visible in these tests
WIDE = CameraModel.from_fov(640, 480, hfov_deg=125, height_m=1.6)


def test_empty_room_is_clear():
    """Floor and ceiling only: nothing in the walking band -> all MAX_RANGE."""
    r = depth_to_ranges(render_depth(WIDE), WIDE, FOV)
    assert np.allclose(r, MAX_RANGE)


def test_wall_ahead_matches_geometry():
    """A flat wall 2 m ahead: centre rays ~2 m, side rays 2/cos(angle)."""
    r = depth_to_ranges(render_depth(WIDE, wall_dist=2.0), WIDE, FOV)
    ray_az = np.deg2rad(np.linspace(-FOV / 2, FOV / 2, N_RANGES))
    expected = np.minimum(2.0 / np.cos(ray_az), MAX_RANGE)
    assert np.allclose(r, expected, rtol=0.08)


def test_pitched_camera_still_works():
    cam = CameraModel.from_fov(640, 480, hfov_deg=125, height_m=1.6, pitch_rad=np.deg2rad(20))
    r = depth_to_ranges(render_depth(cam, wall_dist=3.0), cam, FOV)
    assert abs(r[N_RANGES // 2] - 3.0) < 0.2


def test_obstacle_on_left_lands_in_high_indices():
    """Contract: +y = left. Sim puts left obstacles in rays 12-15."""
    r = depth_to_ranges(render_depth(WIDE, cylinders=[(2.0, 2.0, 0.5)]), WIDE, FOV)
    blocked = np.flatnonzero(r < MAX_RANGE)
    assert blocked.size and blocked.min() >= N_RANGES // 2


def test_matches_simulator_same_scene():
    """The key test: camera ranges == sim ranges for an identical scene."""
    env = PathSenseEnv(EnvConfig())
    env.reset(seed=0)
    env.pos, env.heading = np.array([5.0, 10.0]), 0.0
    scene = [(2.5, 1.0, 0.5), (3.0, -1.5, 0.7)]          # (forward, left, radius)
    env.obstacles = [np.array([5.0 + f, 10.0 + l, rad]) for f, l, rad in scene]
    sim = env._build_observation().ranges
    cam = depth_to_ranges(render_depth(WIDE, cylinders=scene), WIDE, FOV)
    both_hit = (sim < MAX_RANGE) & (cam < MAX_RANGE)
    assert ((sim < MAX_RANGE) == (cam < MAX_RANGE)).mean() >= 0.85   # same rays blocked
    assert np.allclose(sim[both_hit], cam[both_hit], atol=0.35)


def test_narrow_camera_never_reports_unseen_as_clear():
    cam = CameraModel.from_fov(640, 480, hfov_deg=70, height_m=1.6)
    r = depth_to_ranges(render_depth(cam, wall_dist=2.0), cam, FOV)
    assert r.max() < MAX_RANGE
