"""
The whole chain in one call:

    place a fire -> render each camera's photo -> AI finds the fire
    -> each box becomes a wedge of bearings -> wedges cross at the fire
    -> walkways near it get expensive -> safe routes out

    result = run_all((423.1, 76.0), Settings())

The Streamlit app only calls run_all() and draws what comes back, so all of
this can be tested and demonstrated from the command line:

    python -m firepath.pipeline --fire 423 76
"""

import argparse
import math
import time
from dataclasses import dataclass, field, replace

from . import detect as det
from . import localize as loc
from .campus import GOAL, DangerSettings, build_campus
from .cameras import load_scenario
from .routing import alternatives, astar, plan_from_goal
from .simulate import DEFAULT_SPRITES, render_view

CHI2_95 = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49, 5: 11.07}


@dataclass
class Settings:
    scenario: str = "campus_watch"
    fire_width: float | None = None      # metres; None = the scenario's default
    sprite: str = DEFAULT_SPRITES[0]
    smoke: bool = True
    conf: float = 0.35                   # keep AI boxes at least this sure
    sigma_u: float = 6.0                 # how many pixels a box edge wobbles
    mode: str = "auto"                   # "live", "cached", "geometry" or "auto"
    block_radius: float = 25.0
    safe_radius: float = 200.0
    max_danger: float = 8.0
    n_routes: int = 3
    max_overlap: float = 0.60
    start: str = "DormA"


@dataclass
class CameraResult:
    cam: object
    image: object                    # the rendered photo (PIL)
    truth: object                    # FireTruth from the renderer, or None
    boxes: list                      # every AI box (all classes)
    fire_boxes: list                 # the ones used: 'fire', confident, not cut off
    cols: tuple | None = None        # (u_left, u_right) -- the bridge from AI to geometry
    wedge: tuple | None = None       # (theta_min, theta_max) in degrees
    theta: float | None = None       # centre bearing
    sigma: float | None = None       # bearing wobble (radians)
    note: str = ""


@dataclass
class Result:
    settings: Settings
    scenario: object
    fire_true: tuple
    fire_width: float
    source: str                      # "live", "cached" or "geometry"
    cams: list
    used: list = field(default_factory=list)       # CameraResults that gave a wedge
    estimate: tuple | None = None                   # the best single point
    cov: tuple | None = None
    chi2: float | None = None
    dof: int | None = None
    region: list = field(default_factory=list)     # polygon where the fire must be
    error_m: float | None = None
    graph: object = None
    routes: list = field(default_factory=list)
    n_candidates: int = 0
    no_fire_route: object = None
    dijkstra: object = None                         # same search, hint switched off
    plan: object = None                             # signposts from every place
    timings: dict = field(default_factory=dict)
    message: str = ""

    @property
    def located(self):
        return self.estimate is not None

    @property
    def region_ha(self):
        return loc.polygon_area(self.region) / 1e4 if len(self.region) >= 3 else 0.0

    @property
    def chi2_limit(self):
        return CHI2_95.get(self.dof)

    @property
    def start(self):
        return self.graph.index(self.settings.start)

    @property
    def goal(self):
        return self.graph.index(GOAL)


# ------------------------------------------------------------------ stages
def boxes_to_columns(boxes, cam, conf):
    """The bridge from AI to geometry: a box's left and right edges ARE the
    fire's left and right columns.  Several boxes -> take their union, so a
    fire seen as two blobs still gives one wedge."""
    keep = [b for b in boxes if b.label == "fire" and b.conf >= conf]
    if not keep:
        return [], None, "no fire seen"
    u_left, u_right = min(b.x1 for b in keep), max(b.x2 for b in keep)
    if u_left <= 2 or u_right >= cam.width - 2:
        return keep, None, "fire cut off at the edge of the photo -- width unknown"
    return keep, (u_left, u_right), ""


def columns_to_wedge(cols, cam, sigma_u):
    th = sorted(cam.pixel_to_bearing(u) for u in cols)
    uc = (cols[0] + cols[1]) / 2
    return (th[0], th[1]), (th[0] + th[1]) / 2, loc.bearing_sigma(uc, cam.cu, cam.f, sigma_u)


def _best_seed_pair(used):
    """Seed the region with the two wedges that cross most squarely."""
    best, pair = -1.0, (0, 1)
    for i in range(len(used)):
        for j in range(i + 1, len(used)):
            s = abs(math.sin(math.radians(used[i].theta - used[j].theta)))
            if s > best:
                best, pair = s, (i, j)
    return pair, best


def locate(used):
    """Wedges -> best point (IRLS), its covariance and the region they share."""
    obs = [(c.cam.pos, c.theta, c.sigma) for c in used]
    out = loc.localize(obs)
    for c, r in zip(used, out["ranges"]):
        # a crossing behind a camera is not a fire in front of it
        bearing = math.degrees(math.atan2(out["P"][1] - c.cam.y, out["P"][0] - c.cam.x))
        if abs((bearing - c.theta + 180) % 360 - 180) > 90:
            raise ValueError("the sight lines cross behind a camera")
    (i, j), _ = _best_seed_pair(used)
    wedges = [(c.cam.pos, c.wedge[0], c.wedge[1]) for c in used]
    seed = loc.wedge_quad(*wedges[i], *wedges[j])
    return out, loc.region(wedges, seed=seed)


def run_all(fire_xy, settings=None, model=None, observed=None):
    """Run the whole chain for a fire at ``fire_xy``.  Returns a Result.

    ``observed`` skips stages 1-2 and uses photos already taken and checked:
    ``(source, [(camera, image, truth, boxes), ...])``.  Live monitoring
    passes exactly what raised the alarm -- each camera at the heading it had
    when it saw the fire -- so the story shows what the cameras really saw.
    """
    s = settings or Settings()
    sc = load_scenario(s.scenario)
    width = s.fire_width or sc.fire_width
    t0 = time.perf_counter()
    T = {}

    if observed is not None:
        source, obs = observed
        sc = replace(sc, cameras=[o[0] for o in obs])
        shots = [(cam, img, truth) for cam, img, truth, _ in obs]
        all_boxes = [boxes for *_, boxes in obs]
        T["simulate"] = T["detect"] = 0.0
    else:
        # ---- which detector?  (decided first: offline, the fire snaps to the cache grid)
        source, model, fire_xy = resolve_detector(s, sc, width, fire_xy, model)

        # ---- 1. simulate
        shots = [(cam, *render_view(cam, fire_xy, width, s.sprite, s.smoke))
                 for cam in sc.cameras]
        T["simulate"] = time.perf_counter() - t0

        # ---- 2. detect
        t = time.perf_counter()
        names = [cam.name for cam in sc.cameras]
        if source == "live":
            all_boxes = [det.detect_live(img, model, conf=min(s.conf, 0.10))
                         for _, img, _ in shots]
        elif source == "cached":
            all_boxes = det.detect_cached(s.scenario, fire_xy, width, s.sprite, names)
        else:
            all_boxes = [det.detect_from_truth(tr, cam) for cam, _, tr in shots]
        T["detect"] = time.perf_counter() - t

    # ---- 3. locate
    t = time.perf_counter()
    cams = []
    for (cam, img, truth), boxes in zip(shots, all_boxes):
        fire_boxes, cols, note = boxes_to_columns(boxes, cam, s.conf)
        cr = CameraResult(cam, img, truth, boxes, fire_boxes, cols, note=note)
        if cols:
            cr.wedge, cr.theta, cr.sigma = columns_to_wedge(cols, cam, s.sigma_u)
        cams.append(cr)
    res = Result(s, sc, tuple(fire_xy), width, source, cams)
    res.used = [c for c in cams if c.wedge]

    if len(res.used) < 2:
        res.message = (f"Only {len(res.used)} camera(s) saw the fire -- need at least two "
                       "to cross sight lines.")
    elif _best_seed_pair(res.used)[1] < 0.05:
        res.message = "The cameras that saw the fire look along almost the same line."
    else:
        try:
            out, region = locate(res.used)
            res.estimate, res.cov = out["P"], out["cov"]
            res.chi2, res.dof = out["chi2"], out["dof"]
            res.region = region
            res.error_m = math.dist(res.estimate, fire_xy)
        except ValueError as e:
            res.message = f"Could not locate the fire: {e}."
    T["locate"] = time.perf_counter() - t

    # ---- 4. route -- around the region; if there is none, around the best point
    t = time.perf_counter()
    danger = DangerSettings(s.block_radius, s.safe_radius, s.max_danger)
    if len(res.region) >= 3:
        fire_poly = res.region
    elif res.estimate:
        fire_poly = [(res.estimate[0] + width / 2 * math.cos(a), res.estimate[1] + width / 2 * math.sin(a))
                     for a in (k * math.pi / 4 for k in range(8))]
    else:
        fire_poly = []
    g = build_campus(fire_poly, danger)
    res.graph = g
    res.routes, res.n_candidates = alternatives(g, res.start, res.goal, s.n_routes, s.max_overlap)
    res.dijkstra = astar(g, res.start, res.goal, use_hint=False)
    res.plan = plan_from_goal(g, res.goal)
    clean = build_campus((), danger)
    res.no_fire_route = astar(clean, res.start, res.goal)
    T["route"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    res.timings = T
    if not res.routes and not res.message:
        res.message = "No safe route exists from here!"
    if not fire_poly and res.used:
        res.message += ("  THE FIRE WAS SEEN BUT NOT LOCATED -- these routes do not "
                        "account for it.")
    return res


def resolve_detector(s, sc, width, fire_xy, model=None):
    """Pick the detector: live AI if we can, else the offline cache, else the
    renderer's own answer ("geometry" -- clearly labelled, never passed off as AI).
    Returns (source, model, fire_xy) -- offline, fire_xy snaps to the cache grid."""
    mode = s.mode
    if mode in ("live", "auto"):
        model = model or det.load_model(download=(mode == "live"))
        if model is not None:
            return "live", model, tuple(fire_xy)
        if mode == "live":
            raise RuntimeError("live mode asked for, but ultralytics/weights are missing")
    if mode in ("cached", "auto"):
        for xy in (tuple(fire_xy), det.snap_to_grid(fire_xy)):
            if det.is_cached(s.scenario, xy, width, s.sprite):
                return "cached", None, xy
        if mode == "cached":
            raise RuntimeError("no cached detections for this scenario/fire size/sprite")
    return "geometry", None, tuple(fire_xy)


# ------------------------------------------------------------------ CLI
def summary(res):
    g = res.graph
    lines = [f"Scenario  : {res.scenario.title}",
             f"Detection : {res.source.upper()}"]
    for c in res.cams:
        if c.cols:
            lines.append(f"  {c.cam.name}: fire cols {c.cols[0]:.0f}-{c.cols[1]:.0f}  "
                         f"conf {max(b.conf for b in c.fire_boxes):.2f}  "
                         f"wedge {c.wedge[0]:.2f}..{c.wedge[1]:.2f} deg")
        else:
            lines.append(f"  {c.cam.name}: {c.note or 'no fire seen'}")
    if res.located:
        lines += [f"Fire truth: ({res.fire_true[0]:.1f}, {res.fire_true[1]:.1f})",
                  f"Found     : ({res.estimate[0]:.1f}, {res.estimate[1]:.1f})   "
                  f"ERROR {res.error_m:.1f} m",
                  f"Region    : {res.region_ha:.4f} ha, {len(res.region)} corners"]
        if res.dof:
            lines.append(f"Agreement : chi2 {res.chi2:.2f} on {res.dof} dof "
                         f"(95% limit {res.chi2_limit})")
    if res.message:
        lines.append("NOTE      : " + res.message)
    for i, r in enumerate(res.routes):
        lines.append(f"Route {i + 1}   : {' -> '.join(g.name[v] for v in r.path)}")
        lines.append(f"            {r.metres:.0f} m walked, {r.cost:.0f} effective m, "
                     f"worst danger x{r.worst_danger:.2f}")
    if res.routes:
        lines.append(f"Search    : A* examined {res.routes[0].expanded}, "
                     f"Dijkstra {res.dijkstra.expanded} of {len(g.name)} places")
    lines.append("Timings   : " + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in res.timings.items()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Run the FirePath pipeline without the UI.")
    ap.add_argument("--fire", nargs=2, type=float, metavar=("X", "Y"), default=(423.1, 76.0))
    ap.add_argument("--scenario", default="campus_watch")
    ap.add_argument("--mode", default="auto", choices=["auto", "live", "cached", "geometry"])
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--start", default="DormA")
    ap.add_argument("--headless", metavar="DIR", help="also save the photos and map as PNGs here")
    a = ap.parse_args()
    res = run_all(tuple(a.fire), Settings(scenario=a.scenario, mode=a.mode,
                                          fire_width=a.width, start=a.start))
    print(summary(res))
    if a.headless:
        from .animate import save_headless
        for p in save_headless(res, a.headless):
            print("saved", p)


if __name__ == "__main__":
    main()
