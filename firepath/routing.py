"""
Stage 4 -- safe walking routes: A*, alternative routes, and one backward
Dijkstra that puts a signpost at every building.

Ported from docs/campus_evacuation/campus_evacuation.cpp.  No dependencies.
"""

import heapq
import math
from dataclasses import dataclass, field

from .campus import INF
from .geometry import dist


@dataclass
class Route:
    path: list = field(default_factory=list)   # node indices, start -> goal
    cost: float = INF                          # effective metres
    metres: float = 0.0                        # real metres
    worst_danger: float = 1.0
    expanded: int = 0                          # how many places were examined
    visit_order: list = field(default_factory=list)   # for the search animation


def astar(g, start, goal, banned=(), use_hint=True):
    """A*: rank each place by f = g + h, where h is the straight-line distance
    still to go.  h never overestimates (every danger factor is >= 1), so A*
    still returns the cheapest route -- it just looks at fewer places.

    ``use_hint=False`` switches h off, which is exactly Dijkstra.
    """
    n = len(g.name)
    g_score = [INF] * n
    came_from = [-1] * n
    done = [False] * n
    out = Route()

    def h(v):
        return dist(g.pos[v], g.pos[goal]) if use_hint else 0.0

    g_score[start] = 0.0
    open_list = [(h(start), start)]
    while open_list:
        _, u = heapq.heappop(open_list)
        if done[u]:
            continue
        done[u] = True
        out.expanded += 1
        out.visit_order.append(u)
        if u == goal:
            break
        for e in g.adj[u]:
            if e.id in banned or e.cost == INF or done[e.to]:
                continue
            tentative = g_score[u] + e.cost
            if tentative < g_score[e.to] - 1e-9:
                g_score[e.to] = tentative
                came_from[e.to] = u
                heapq.heappush(open_list, (tentative + h(e.to), e.to))

    if g_score[goal] == INF:
        return out                                  # no way through

    v = goal
    while v != -1:                                  # follow the breadcrumbs back
        out.path.append(v)
        v = came_from[v]
    out.path.reverse()
    out.cost = g_score[goal]
    for e in _edges_of(g, out.path):
        out.metres += e.length
        out.worst_danger = max(out.worst_danger, e.danger)
    return out


def _edges_of(g, path):
    for a, b in zip(path, path[1:]):
        yield next(e for e in g.adj[a] if e.to == b)


def edge_ids_of(g, path):
    return [e.id for e in _edges_of(g, path)]


def overlap(g, a, b):
    """Fraction of route a's walking that route b also walks (0..1)."""
    ids_b = set(edge_ids_of(g, b.path))
    shared = sum(e.length for e in _edges_of(g, a.path) if e.id in ids_b)
    return shared / a.metres if a.metres > 0 else 0.0


def alternatives(g, start, goal, n_routes=3, max_overlap=0.60):
    """Take a route we already like, close ONE of its walkways, ask A* again.
    Keep the cheapest answer that is not just a small variation on a route we
    already have.  Returns (kept routes, number of distinct candidates)."""
    best = astar(g, start, goal)
    if not best.path:
        return [], 0
    kept, candidates, seen = [best], [], {tuple(best.path)}
    k = 0
    while k < len(kept) and len(kept) < n_routes:
        for wid in edge_ids_of(g, kept[k].path):
            alt = astar(g, start, goal, banned={wid})
            if not alt.path or tuple(alt.path) in seen:
                continue
            seen.add(tuple(alt.path))
            candidates.append(alt)
        candidates.sort(key=lambda r: r.cost)       # stable, like std::sort here
        for c, cand in enumerate(candidates):
            if any(overlap(g, cand, r) > max_overlap for r in kept):
                continue
            kept.append(candidates.pop(c))
            break
        k += 1
    return kept, len(kept) + len(candidates)


@dataclass
class EvacPlan:
    cost: list        # effective metres from each place to the goal
    next_hop: list    # the very next place to walk to (-1 = none)


def plan_from_goal(g, goal):
    """One Dijkstra spreading out from the assembly point.  Walkways cost the
    same both ways, so the cost it finds for a place IS that place's cost to
    reach the goal -- one pass answers every starting point at once."""
    n = len(g.name)
    cost, next_hop, done = [INF] * n, [-1] * n, [False] * n
    cost[goal] = 0.0
    open_list = [(0.0, goal)]
    while open_list:
        _, u = heapq.heappop(open_list)
        if done[u]:
            continue
        done[u] = True
        for e in g.adj[u]:
            if e.cost == INF:
                continue
            tentative = cost[u] + e.cost
            if tentative < cost[e.to] - 1e-9:
                cost[e.to] = tentative
                next_hop[e.to] = u                  # from e.to, step to u
                heapq.heappush(open_list, (tentative, e.to))
    return EvacPlan(cost, next_hop)


def follow_signs(plan, start):
    """Follow the next-hop signs from any place all the way to the goal."""
    if plan.cost[start] == INF:
        return []
    route, v = [start], start
    while plan.next_hop[v] != -1:
        v = plan.next_hop[v]
        route.append(v)
    return route


def route_names(g, route):
    return [g.name[i] for i in (route.path if isinstance(route, Route) else route)]


def fmt_cost(x):
    return "--" if x == INF or math.isinf(x) else f"{x:.0f}"
