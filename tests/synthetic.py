"""Render a fake metric depth map for a simple scene: floor, a wall ahead,
and vertical cylinders (the simulator's obstacles). Used to test the pipeline
without a camera or a depth model."""
import numpy as np


def render_depth(cam, wall_dist=None, cylinders=(), ceiling=2.6):
    """cylinders: list of (forward_m, left_m, radius_m) in the agent frame."""
    v, u = np.mgrid[0:cam.height, 0:cam.width].astype(float)
    # ray direction in camera frame (x right, y down, z forward), z = 1
    x = (u - cam.cx) / cam.fx
    y = (v - cam.cy) / cam.fy
    c, s = np.cos(cam.pitch_rad), np.sin(cam.pitch_rad)
    fwd, down, right = c - y * s, y * c + s, x          # per unit of camera z
    t = np.full(x.shape, np.inf)                        # camera-z depth to first hit
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = np.where(down > 0, cam.height_m / down, np.inf)            # floor
        tc = np.where(down < 0, (cam.height_m - ceiling) / down, np.inf)  # ceiling
        t = np.minimum(t, np.minimum(tf, tc))
        if wall_dist is not None:
            t = np.minimum(t, np.where(fwd > 0, wall_dist / fwd, np.inf))
        for (cf, cl, r) in cylinders:
            cr = -cl                                     # left -> right coordinate
            a = fwd**2 + right**2
            b = -2 * (fwd * cf + right * cr)
            cc = cf**2 + cr**2 - r**2
            disc = b**2 - 4 * a * cc
            th = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
            t = np.minimum(t, np.where(th > 0, th, np.inf))
    t[~np.isfinite(t)] = 50.0
    return t.astype(np.float32)
