"""
The campus as a graph -- places joined by walkways -- and how a fire changes it.

Ported from docs/campus_evacuation/campus_evacuation.cpp.  No dependencies.
"""

import math
from dataclasses import dataclass, field

from .geometry import dist, distance_to_fire

INF = math.inf

NODES = {
    "DormA": (80, 300),   "Library": (250, 330), "Science": (430, 350),
    "Sports": (640, 330), "NorthGate": (830, 300), "WestLot": (90, 150),
    "Union": (270, 170),  "Cafeteria": (450, 190), "Engineer": (640, 170),
    "Admin": (840, 160),  "Chapel": (250, 20),   "Plaza": (430, 30),
    "Theatre": (620, 30), "SouthLot": (250, -120), "Field": (470, -140),
    "ASSEMBLY": (850, -60),
}
WALKWAYS = [
    ("DormA", "Library"), ("Library", "Science"), ("Science", "Sports"),
    ("Sports", "NorthGate"), ("DormA", "WestLot"), ("Library", "Union"),
    ("Science", "Cafeteria"), ("Sports", "Engineer"), ("NorthGate", "Admin"),
    ("WestLot", "Union"), ("Union", "Cafeteria"), ("Cafeteria", "Engineer"),
    ("Engineer", "Admin"), ("WestLot", "Chapel"), ("Union", "Chapel"),
    ("Cafeteria", "Plaza"), ("Engineer", "Theatre"), ("Admin", "ASSEMBLY"),
    ("Chapel", "Plaza"), ("Plaza", "Theatre"), ("Chapel", "SouthLot"),
    ("Plaza", "Field"), ("Theatre", "ASSEMBLY"), ("SouthLot", "Field"),
    ("Field", "ASSEMBLY"),
]
GOAL = "ASSEMBLY"


@dataclass
class DangerSettings:
    block_radius: float = 25.0   # closer than this to the fire: walkway closed (m)
    safe_radius: float = 200.0   # further than this: fire ignored (m)
    max_danger: float = 8.0      # a walkway at the blocking edge feels this many times longer


def danger_factor(d, s=DangerSettings()):
    """Turn "how close is the fire" into "how many times worse is this walkway".
    Far away -> 1.0.  Very close -> INF (closed).  In between, a straight ramp."""
    if d <= s.block_radius:
        return INF
    if d >= s.safe_radius:
        return 1.0
    how_bad = (s.safe_radius - d) / (s.safe_radius - s.block_radius)   # 0..1
    return 1.0 + (s.max_danger - 1.0) * how_bad


@dataclass
class Edge:
    to: int
    length: float    # real walking metres
    danger: float    # multiplier from the fire
    cost: float      # length * danger ("effective metres")
    id: int          # so we can ban a specific walkway later


@dataclass
class Walkway:
    u: int
    v: int
    length: float
    to_fire: float
    danger: float

    @property
    def cost(self):
        return INF if self.danger == INF else self.length * self.danger


@dataclass
class Graph:
    name: list = field(default_factory=list)
    pos: list = field(default_factory=list)
    adj: list = field(default_factory=list)
    walkways: list = field(default_factory=list)

    def add_node(self, n, x, y):
        self.name.append(n)
        self.pos.append((x, y))
        self.adj.append([])
        return len(self.name) - 1

    def add_walkway(self, u, v, fire, s):
        length = dist(self.pos[u], self.pos[v])
        d = distance_to_fire(self.pos[u], self.pos[v], fire)
        dg = danger_factor(d, s)
        c = INF if dg == INF else length * dg
        wid = len(self.walkways)
        self.walkways.append(Walkway(u, v, length, d, dg))
        self.adj[u].append(Edge(v, length, dg, c, wid))
        self.adj[v].append(Edge(u, length, dg, c, wid))

    def index(self, n):
        return self.name.index(n)


def build_campus(fire=(), settings=DangerSettings(), nodes=NODES, walkways=WALKWAYS):
    """The campus graph with every walkway weighted by its distance to ``fire``
    (a polygon, list of (x, y); empty means no fire)."""
    g = Graph()
    for n, (x, y) in nodes.items():
        g.add_node(n, x, y)
    for a, b in walkways:
        g.add_walkway(g.index(a), g.index(b), list(fire), settings)
    return g
