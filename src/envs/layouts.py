"""Indoor layouts for PathSenseEnv.

Every layout returns (walls, boxes, circles, structured):
  walls    (N, 5) rectangles the walker knows about and plans around (corridor walls,
           shelves). Rows are (cx, cy, half_w, half_h, angle_rad).
  boxes    (M, 5) rectangles the walker does NOT know about: bins, chairs, boxes, tables.
  circles  (K, 3) (cx, cy, r) poles, plants, people.
  structured  True if the walker should walk along the layout's axes (corridors, aisles).
Only walls guide the walker's own route; everything blocks rays and causes collisions.
"""
import numpy as np


def _rect(x0, y0, x1, y1):
    """Axis-aligned rectangle from corners."""
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2, 0.0])


def _bound_r(r):
    return float(np.hypot(r[2], r[3]))


def _far_from(p, items, gap):
    """p = (x, y, bounding radius); items = list of (x, y, bounding radius)."""
    return all(np.hypot(p[0] - q[0], p[1] - q[1]) >= p[2] + q[2] + gap for q in items)


def room(rng, cfg):
    """Open room: mixed circles and boxes (the original task, plus furniture)."""
    size, boxes, circles, placed = cfg.arena_size, [], [], []
    n = int(rng.integers(cfg.n_obstacles_range[0], cfg.n_obstacles_range[1] + 1))
    for _ in range(n * 20):
        if len(placed) >= n:
            break
        c = rng.uniform(2.0, size - 2.0, size=2)
        if rng.uniform() < cfg.box_fraction:
            hw, hh = rng.uniform(*cfg.box_half_size_range, size=2)
            r = np.array([c[0], c[1], hw, hh, rng.uniform(0, np.pi)])
            item, add = (c[0], c[1], _bound_r(r)), boxes
        else:
            rad = rng.uniform(*cfg.obstacle_radius_range)
            r = np.array([c[0], c[1], rad])
            item, add = (c[0], c[1], rad), circles
        if _far_from(item, placed, 0.3):
            placed.append(item)
            add.append(r)
    return np.zeros((0, 5)), np.array(boxes).reshape(-1, 5), np.array(circles).reshape(-1, 3), False


def _strips(rng, size, n, width_range, min_block):
    """n corridor strips (lo, hi) across [0, size] with blocks >= min_block between them."""
    for _ in range(100):
        widths = rng.uniform(*width_range, size=n)
        free = size - widths.sum() - min_block * (n + 1)
        if free < 0:
            continue
        gaps = min_block + rng.dirichlet(np.ones(n + 1)) * free
        strips, x = [], 0.0
        for w, g in zip(widths, gaps[:-1]):
            x += g
            strips.append((x, x + w))
            x += w
        return strips
    return [(size / 2 - 1.0, size / 2 + 1.0)]


def _clutter_along(rng, cfg, lo, hi, along, horizontal, out_boxes, out_circles):
    """Bins/chairs against a corridor wall and the odd person, always leaving a gap."""
    width, used = hi - lo, []
    for _ in range(int(rng.integers(*cfg.corridor_clutter_range))):
        s = rng.uniform(along[0] + 1.0, along[1] - 1.0)
        if any(abs(s - u) < 1.6 for u in used):                  # never two items side by side
            continue
        used.append(s)
        if rng.uniform() < 0.25:                                  # a person
            rad = rng.uniform(0.25, 0.35)
            t = rng.uniform(lo + rad, hi - rad)
            if max(t - rad - lo, hi - t - rad) < cfg.min_gap_m:   # no way past
                continue
            out_circles.append(np.array([s, t, rad]) if horizontal else np.array([t, s, rad]))
            continue
        depth = min(rng.uniform(0.3, 0.7), width - cfg.min_gap_m)    # always leave a gap
        if depth < 0.2:
            continue
        length = rng.uniform(0.4, 1.4)
        side = rng.choice([lo, hi - depth])
        x0, x1, y0, y1 = s - length / 2, s + length / 2, side, side + depth
        out_boxes.append(_rect(x0, y0, x1, y1) if horizontal else _rect(y0, x0, y1, x1))


def corridors(rng, cfg):
    """City-block grid: corridors run wall to wall; the blocks between are walls."""
    size = cfg.arena_size
    nh, nv = (int(rng.integers(1, 4)) for _ in range(2))
    hs = _strips(rng, size, nh, cfg.corridor_width_range, cfg.min_block_m)
    vs = _strips(rng, size, nv, cfg.corridor_width_range, cfg.min_block_m)
    xs = [0.0] + [v for s in vs for v in s] + [size]
    ys = [0.0] + [v for s in hs for v in s] + [size]
    walls = [_rect(xs[i], ys[j], xs[i + 1], ys[j + 1])
             for i in range(0, len(xs), 2) for j in range(0, len(ys), 2)
             if xs[i + 1] - xs[i] > 0.05 and ys[j + 1] - ys[j] > 0.05]
    # knock out a block now and then: an open hall where corridors meet
    if len(walls) > 4 and rng.uniform() < 0.3:
        walls.pop(int(rng.integers(len(walls))))
    boxes, circles = [], []
    for lo, hi in hs:
        _clutter_along(rng, cfg, lo, hi, (0.0, size), True, boxes, circles)
    for lo, hi in vs:
        _clutter_along(rng, cfg, lo, hi, (0.0, size), False, boxes, circles)
    return np.array(walls), np.array(boxes).reshape(-1, 5), np.array(circles).reshape(-1, 3), True


def aisles(rng, cfg):
    """Rows of shelves or tables with aisles between, a clear aisle round the edge."""
    size, walls = cfg.arena_size, []
    aisle, depth = rng.uniform(*cfg.aisle_width_range), rng.uniform(0.6, 1.2)
    margin = rng.uniform(1.5, 2.5)
    y = margin
    while y + depth <= size - margin:
        x = margin
        while x < size - margin - 1.0:
            length = min(rng.uniform(3.0, 8.0), size - margin - x)
            walls.append(_rect(x, y, x + length, y + depth))
            x += length + rng.uniform(1.5, 2.5)                    # cross-aisle
        y += depth + aisle
    walls = np.array(walls).reshape(-1, 5)
    if rng.uniform() < 0.5:                                        # rows run the other way
        walls[:, [0, 1, 2, 3]] = walls[:, [1, 0, 3, 2]]
    boxes, circles = [], []
    for _ in range(int(rng.integers(*cfg.corridor_clutter_range))):
        c = rng.uniform(1.0, size - 1.0, size=2)
        if rect_clearance(c, walls) < 0.8:
            continue
        if rng.uniform() < 0.5:
            circles.append(np.array([c[0], c[1], rng.uniform(0.25, 0.4)]))
        else:
            hw, hh = rng.uniform(0.2, 0.4, size=2)
            boxes.append(np.array([c[0], c[1], hw, hh, rng.uniform(0, np.pi)]))
    return walls, np.array(boxes).reshape(-1, 5), np.array(circles).reshape(-1, 3), True


LAYOUTS = {"room": room, "corridors": corridors, "aisles": aisles}


# ---------------------------------------------------------------- geometry

def rays_rects(origin, dirs, rects):
    """Distance along each ray (rows of dirs) to the nearest rectangle; inf if none."""
    if len(rects) == 0:
        return np.full(len(dirs), np.inf)
    c, s = np.cos(rects[:, 4]), np.sin(rects[:, 4])             # (N,)
    d = origin - rects[:, :2]                                    # (N, 2)
    ox, oy = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    dx = dirs[:, None, 0] * c + dirs[:, None, 1] * s             # (R, N)
    dy = -dirs[:, None, 0] * s + dirs[:, None, 1] * c
    with np.errstate(divide="ignore", invalid="ignore"):
        tx1, tx2 = (-rects[:, 2] - ox) / dx, (rects[:, 2] - ox) / dx
        ty1, ty2 = (-rects[:, 3] - oy) / dy, (rects[:, 3] - oy) / dy
    t_in = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    t_out = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (t_out >= np.maximum(t_in, 0.0)) & np.isfinite(t_in)
    return np.where(hit, np.maximum(t_in, 0.0), np.inf).min(axis=1)


def rays_circles(origin, dirs, circles):
    if len(circles) == 0:
        return np.full(len(dirs), np.inf)
    oc = origin - circles[:, :2]                                 # (K, 2)
    b = dirs @ oc.T                                              # (R, K)
    c = (oc ** 2).sum(1) - circles[:, 2] ** 2
    disc = b ** 2 - c
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = np.where(-b - sq > 1e-6, -b - sq, -b + sq)
    return np.where((disc >= 0) & (t > 1e-6), t, np.inf).min(axis=1)


def rect_clearance(p, rects):
    """Distance from point p to the nearest rectangle (0 if inside); inf if none."""
    if len(rects) == 0:
        return np.inf
    c, s = np.cos(rects[:, 4]), np.sin(rects[:, 4])
    d = p - rects[:, :2]
    lx, ly = np.abs(d[:, 0] * c + d[:, 1] * s), np.abs(-d[:, 0] * s + d[:, 1] * c)
    return float(np.hypot(np.maximum(lx - rects[:, 2], 0), np.maximum(ly - rects[:, 3], 0)).min())


def corners(r):
    """The 4 corners of one rectangle row, for drawing."""
    c, s = np.cos(r[4]), np.sin(r[4])
    local = np.array([[-r[2], -r[3]], [r[2], -r[3]], [r[2], r[3]], [-r[2], r[3]]])
    return r[:2] + local @ np.array([[c, s], [-s, c]])
