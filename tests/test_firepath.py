import math
import sys

import pytest

from firepath import detect as det
from firepath.campus import GOAL, build_campus
from firepath.cameras import load_scenario
from firepath.geometry import dist, distance_to_fire
from firepath.localize import polygon_area
from firepath.pipeline import Settings, boxes_to_columns, columns_to_wedge, locate, run_all
from firepath.routing import alternatives, astar, plan_from_goal

# The fire polygon found by the camera guide, as used in campus_evacuation.cpp
GUIDE_FIRE = [(385.43, 34.93), (398.22, 36.09), (459.46, 92.70),
              (470.82, 127.81), (457.59, 124.21), (392.03, 68.53)]

FIRES = [(423.1, 76.0), (535, 345), (450, 215), (250, 60), (700, 100), (180, 250)]


def test_routing_matches_cpp():
    """Every number in docs/campus_evacuation/results.txt, reproduced."""
    assert polygon_area(GUIDE_FIRE) / 1e4 == pytest.approx(0.2676, abs=5e-5)
    g = build_campus(GUIDE_FIRE)
    s, t = g.index("DormA"), g.index(GOAL)

    routes, n_candidates = alternatives(g, s, t)
    assert [round(r.cost) for r in routes] == [1118, 1768, 2437]
    assert [round(r.metres) for r in routes] == [1118, 1113, 1106]
    assert [round(r.worst_danger, 2) for r in routes] == [1.00, 2.54, 3.55]
    assert [g.name[v] for v in routes[2].path] == \
        ["DormA", "WestLot", "Chapel", "SouthLot", "Field", "ASSEMBLY"]
    assert n_candidates == 5
    assert routes[0].expanded == 8

    closed = {(g.name[w.u], g.name[w.v]) for w in g.walkways if w.danger == math.inf}
    assert closed == {("Cafeteria", "Plaza"), ("Chapel", "Plaza")}

    plan = plan_from_goal(g, t)
    assert round(plan.cost[g.index("Plaza")]) == 1931
    assert round(plan.cost[g.index("Cafeteria")]) == 1793
    assert g.name[plan.next_hop[g.index("Cafeteria")]] == "Science"
    assert plan.cost[s] == pytest.approx(routes[0].cost)

    clean = build_campus()
    assert [clean.name[v] for v in astar(clean, s, t).path] == \
        ["DormA", "WestLot", "Chapel", "Plaza", "Theatre", "ASSEMBLY"]


@pytest.mark.parametrize("scenario", ["campus_watch", "guide_example"])
@pytest.mark.parametrize("fire", FIRES)
def test_roundtrip_geometry(scenario, fire):
    """world -> pixel -> bearing -> world, using the renderer's TRUE columns
    (no AI), must land back on the fire.  Isolates geometry bugs from AI bugs."""
    sc = load_scenario(scenario)

    class Seen:                                   # a minimal CameraResult
        def __init__(self, cam, cols):
            self.cam = cam
            self.wedge, self.theta, self.sigma = columns_to_wedge(cols, cam, 1.0)

    seen = [Seen(c, c.fire_columns(fire, sc.fire_width)) for c in sc.cameras
            if c.sees(fire, sc.fire_width)]
    if len(seen) < 2:
        pytest.skip("fewer than two cameras see this spot")
    out, region = locate(seen)
    assert dist(out["P"], fire) < 0.5
    assert len(region) >= 3


def test_admissible():
    """A*'s hint (straight-line distance) never exceeds the true cost to go."""
    for fire in ([], GUIDE_FIRE):
        g = build_campus(fire)
        goal = g.index(GOAL)
        plan = plan_from_goal(g, goal)
        for v in range(len(g.name)):
            assert dist(g.pos[v], g.pos[goal]) <= plan.cost[v] + 1e-9


@pytest.mark.parametrize("fire", FIRES)
def test_no_route_through_fire(fire):
    res = run_all(fire, Settings(mode="geometry"))
    assert res.located
    s = res.settings
    for r in res.routes:
        for a, b in zip(r.path, r.path[1:]):
            d = distance_to_fire(res.graph.pos[a], res.graph.pos[b], res.region)
            assert d > s.block_radius


def test_north_fire_flips_route_south():
    """The demo moment: a fire on the north path sends Dorm A south."""
    res = run_all((535, 345), Settings(mode="geometry"))
    names = [res.graph.name[v] for v in res.routes[0].path]
    assert "Science" not in names and "Chapel" in names


def test_bridge_union_and_edge_rule():
    cam = load_scenario("campus_watch").cameras[0]
    B = det.Box
    boxes = [B(100, 0, 150, 10, 0.9, "fire"), B(140, 0, 220, 10, 0.5, "fire"),
             B(0, 0, 900, 10, 0.9, "smoke"), B(300, 0, 400, 10, 0.1, "fire")]
    keep, cols, _ = boxes_to_columns(boxes, cam, 0.35)
    assert cols == (100, 220) and len(keep) == 2
    _, cols, note = boxes_to_columns([B(0, 0, 50, 10, 0.9, "fire")], cam, 0.35)
    assert cols is None and "edge" in note


def test_cached_fallback(monkeypatch):
    """With ultralytics hidden, the pipeline still completes -- and says how."""
    monkeypatch.setitem(sys.modules, "ultralytics", None)      # import now fails
    monkeypatch.setattr(det, "_model", None)
    res = run_all((423.1, 76.0), Settings(mode="auto"))
    assert res.source in ("cached", "geometry")
    assert res.located and res.routes
    if det.load_cache():                                         # cache built?
        assert res.source == "cached"


# ---------------------------------------------------------------- live monitoring
from firepath import monitor as live  # noqa: E402


def test_fire_plan_grows():
    plan = live.FirePlan(start=10, xy=(0, 0), sprite="x", w_start=6, w_max=40, grow=20)
    assert plan.width(9) is None
    assert plan.width(10) == 6
    assert plan.width(20) == pytest.approx(23)
    assert plan.width(99) == 40
    assert live.FirePlan(start=3, xy=(0, 0), sprite="x", grow=0).width(3) == 40


def test_patrol_sweeps_within_limits():
    sc = load_scenario("campus_watch")
    patrol = live.Patrol(sweep=30, period=20)
    for t in range(40):
        for home, cam in zip(sc.cameras, patrol.cameras_at(sc.cameras, t)):
            assert abs(cam.psi - home.psi) <= 30 + 1e-9
    assert [c.psi for c in live.Patrol(sweep=0).cameras_at(sc.cameras, 7)] == \
        [c.psi for c in sc.cameras]


def test_random_spots_are_seen_by_two_cameras():
    import random
    sc = load_scenario("campus_watch")
    patrol = live.Patrol()
    rng = random.Random(1)
    for _ in range(5):
        xy = live.random_spot(sc, patrol, 40, rng)
        assert len(live.visible_to(sc, patrol, xy, 40)) >= 2


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_monitor_to_alarm_to_routes(seed):
    """The whole live workflow without the AI: quiet, fire, alarm, located, routed."""
    import random
    sc = load_scenario("campus_watch")
    patrol = live.Patrol(30, 20)
    plan = live.plan_fire(sc, patrol, random.Random(seed), (3, 6), w_start=40, grow=0)
    mon = live.Monitor(sc, plan, patrol)
    while not mon.alarm and mon.t < 60:
        mon.tick(None)
        if mon.t < plan.start:
            assert not mon.recent()                    # no fire, nothing seen
    assert mon.alarm and mon.alarm["t"] >= plan.start
    cams = mon.alarm_cameras()
    assert any(c.psi != h.psi for c, h in zip(cams, sc.cameras))   # they really turned
    res = run_all(plan.xy, Settings(mode="geometry", fire_width=mon.alarm["width"],
                                    sprite=plan.sprite), observed=mon.observed())
    assert res.located and res.error_m < 0.5 and res.routes
    # the story uses exactly the photos that raised the alarm
    assert [c.image for c in res.cams] == [v.image for v in mon.alarm_views()]
    assert len(res.used) >= 2
