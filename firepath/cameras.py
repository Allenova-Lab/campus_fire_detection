"""
Cameras: where each one stands, which way it looks, and the one equation
that links a map position to a photo column -- used in both directions.

    pixel  -> bearing   theta = psi - atan((u - cu) / f)       (the localiser)
    bearing -> pixel    u     = cu + f * tan(psi - theta)      (the simulator)

Bearings are in degrees, counter-clockwise from the +x (east) axis, exactly
as in the camera guide.  No dependencies.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from . import localize

SCENARIOS = Path(__file__).resolve().parent.parent / "data" / "scenarios"


def wrap180(a):
    """Angle in degrees wrapped into (-180, 180]."""
    return -((-a + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class Camera:
    name: str
    x: float
    y: float
    psi: float                 # heading: the bearing of the image centre (deg)
    hfov: float = 60.0         # horizontal field of view (deg)
    width: int = 1920          # image size (px)
    height: int = 1080
    mount_h: float = 12.0      # camera height above the ground (m)

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def f(self):
        return localize.focal_from_fov(self.width, self.hfov)

    @property
    def cu(self):
        return self.width / 2.0

    @property
    def cv(self):
        return self.height / 2.0

    # --- the localiser direction: photo -> map
    def pixel_to_bearing(self, u):
        return localize.pixel_to_bearing(u, self.psi, self.f, self.cu)

    # --- the simulator direction: map -> photo
    def bearing_to(self, p):
        return math.degrees(math.atan2(p[1] - self.y, p[0] - self.x))

    def world_to_pixel(self, p):
        """Image column of ground point p, or None if it is behind the camera."""
        off = wrap180(self.psi - self.bearing_to(p))
        if abs(off) >= 89.0:
            return None
        return self.cu + self.f * math.tan(math.radians(off))

    def ground_row(self, r):
        """Image row where the ground is, r metres away (flat ground, level camera)."""
        return self.cv + self.f * self.mount_h / r

    def fire_columns(self, p, width_m):
        """True left/right image columns of a fire of this width centred at p.

        The fire's two edges are the points width/2 either side of p, measured
        across the line of sight.  Returns None if either edge is behind us.
        """
        r = math.dist(p, self.pos)
        th = math.radians(self.bearing_to(p))
        nx, ny = -math.sin(th), math.cos(th)            # across the line of sight
        h = width_m / 2.0
        ua = self.world_to_pixel((p[0] + h * nx, p[1] + h * ny))
        ub = self.world_to_pixel((p[0] - h * nx, p[1] - h * ny))
        if ua is None or ub is None or r < 1.0:
            return None
        return min(ua, ub), max(ua, ub)

    def sees(self, p, width_m, min_px=0.0):
        """Is the whole fire inside the photo, and at least min_px wide?"""
        cols = self.fire_columns(p, width_m)
        return (cols is not None and cols[0] >= 0 and cols[1] <= self.width
                and cols[1] - cols[0] >= min_px)

    def fov_polygon(self, reach):
        """The camera's field of view on the map, as a triangle-ish fan."""
        pts = [self.pos]
        for k in range(9):
            b = math.radians(self.psi - self.hfov / 2 + self.hfov * k / 8)
            pts.append((self.x + reach * math.cos(b), self.y + reach * math.sin(b)))
        return pts


@dataclass
class Scenario:
    name: str
    title: str
    cameras: list
    fires: dict                 # preset name -> (x, y)
    fire_width: float = 40.0
    note: str = ""


def load_scenario(name):
    d = json.loads((SCENARIOS / f"{name}.json").read_text())
    d["cameras"] = [Camera(**c) for c in d["cameras"]]
    d["fires"] = {k: tuple(v) for k, v in d["fires"].items()}
    return Scenario(**d)


def list_scenarios():
    return sorted(p.stem for p in SCENARIOS.glob("*.json"))


def save_scenario(sc):
    d = asdict(sc)
    (SCENARIOS / f"{sc.name}.json").write_text(json.dumps(d, indent=2) + "\n")
