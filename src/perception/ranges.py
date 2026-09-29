"""Depth map -> the 16 ranges the simulator's sensor produces.

Sim convention (checked in pathsense_env._build_observation):
ray i points at heading + linspace(-fov/2, +fov/2)[i]. With the contract's
+y = left, ray 0 is the RIGHTMOST ray and ray 15 the LEFTMOST.
"""
import numpy as np
from ..common.types import N_RANGES, MAX_RANGE

INDEX0_IS_RIGHT = True   # flip only if Track A changes the sim


def backproject(depth, cam, stride=4):
    """Pixels + depth -> 3D points in a gravity-aligned frame.
    Returns (right, forward, height_above_floor), each 1-D, metres."""
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    ok = np.isfinite(z) & (z > 0)
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - cam.cx) * z / cam.fx          # camera right
    y = (v - cam.cy) * z / cam.fy          # camera down
    c, s = np.cos(cam.pitch_rad), np.sin(cam.pitch_rad)
    forward = z * c - y * s                # undo head pitch
    down = y * c + z * s
    return x, forward, cam.height_m - down


def depth_to_ranges(depth, cam, fov_deg, min_h=0.15, max_h=1.9, pct=5.0, stride=4,
                    ray_width=0.3):
    """depth: (H, W) metric depth in metres, same size as cam.width x cam.height.
    fov_deg: MUST equal the simulator's EnvConfig.fov_deg.
    ray_width: fraction of the gap between rays each ray looks at. The sim casts
    thin rays, so keep this small (a wide wedge reads too close at the sides)."""
    right, fwd, h = backproject(depth, cam, stride)
    keep = (h > min_h) & (h < max_h) & (fwd > 0.05)       # drop floor, ceiling, behind
    az = np.arctan2(right[keep], fwd[keep])               # + = right
    dist = np.hypot(right[keep], fwd[keep])               # horizontal distance, like the sim

    half = np.deg2rad(fov_deg) / 2
    step = 2 * half / (N_RANGES - 1)
    ray_az = np.linspace(half, -half, N_RANGES) if INDEX0_IS_RIGHT \
        else np.linspace(-half, half, N_RANGES)           # azimuth (+right) of each ray
    window = ray_width * step / 2

    ranges = np.full(N_RANGES, MAX_RANGE, np.float32)
    for i in range(N_RANGES):
        d = dist[np.abs(az - ray_az[i]) <= window]
        if d.size >= 3:                                   # ignore a couple of stray points
            ranges[i] = min(np.percentile(d, pct), MAX_RANGE)

    # Rays the camera cannot see: copy the nearest visible ray (never report "clear").
    seen = np.abs(ray_az) <= cam.hfov_rad / 2 + 1e-6
    if not seen.all() and seen.any():
        vis = np.flatnonzero(seen)
        for i in np.flatnonzero(~seen):
            ranges[i] = ranges[vis[np.argmin(np.abs(vis - i))]]
    return ranges
