"""
Live monitoring -- the whole workflow, as it would really run.

The cameras sweep back and forth.  Once a second each one takes a photo and
the AI checks it.  Somewhere, at a time nobody announces, a fire starts --
small at first, then growing.  When two cameras have seen it, the alarm goes
and the pipeline takes over: locate the fire, reroute everyone.

A turning camera still gives exact bearings, because we know which way it
was pointing when it took each photo.  And since a fire stays put, sightings
from different seconds can be combined -- CAM 2 at 00:14 and CAM 4 at 00:16
cross just as well as if they had looked at the same moment.

    python -m firepath.monitor                     # a random fire, live AI
    python -m firepath.monitor --seed 4 --mode geometry
"""

import argparse
import math
import random
from dataclasses import dataclass, field, replace

from PIL import Image, ImageDraw

from . import detect as det
from .campus import NODES, WALKWAYS
from .pipeline import boxes_to_columns
from .simulate import DEFAULT_SPRITES, render_view

CAMPUS_X, CAMPUS_Y = (80, 850), (-140, 350)
MIN_PX_NO_AI = 60        # without the AI, pretend it sees fires at least this wide
COMPASS = ["east", "north-east", "north", "north-west",
           "west", "south-west", "south", "south-east"]


def compass(bearing_deg):
    return COMPASS[round((bearing_deg % 360) / 45) % 8]


def clock(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


# ---------------------------------------------------------------- the cameras
@dataclass
class Patrol:
    """Every camera sweeps its home heading +- ``sweep`` degrees, once every
    ``period`` seconds, each a little out of step with the next."""
    sweep: float = 30.0
    period: float = 20.0

    def heading(self, cam, t, k, n):
        if self.sweep <= 0 or self.period <= 0:
            return cam.psi
        return cam.psi + self.sweep * math.sin(2 * math.pi * (t / self.period + k / n))

    def cameras_at(self, cams, t):
        return [replace(c, psi=round(self.heading(c, t, k, len(cams)), 2))
                for k, c in enumerate(cams)]


# ---------------------------------------------------------------- the fire
@dataclass
class FirePlan:
    """When and where the fire starts, and how it grows (the system is not told)."""
    start: int                 # seconds after monitoring begins
    xy: tuple
    sprite: str
    w_start: float = 10.0      # width when it starts (m)
    w_max: float = 40.0        # width once fully grown (m)
    grow: float = 20.0         # seconds to grow to full size (0 = full size at once)

    def width(self, t):
        if t < self.start:
            return None
        if self.grow <= 0:
            return self.w_max
        return self.w_start + (self.w_max - self.w_start) * min(1.0, (t - self.start) / self.grow)


def visible_to(scenario, patrol, xy, width, min_px=100, samples=24):
    """Cameras that can see a fire this wide at xy at some point in their sweep."""
    out = []
    n = len(scenario.cameras)
    for k, cam in enumerate(scenario.cameras):
        for j in range(samples):
            t = j * patrol.period / samples if patrol.period else 0.0
            if replace(cam, psi=patrol.heading(cam, t, k, n)).sees(xy, width, min_px):
                out.append(cam.name)
                break
    return out


def random_spot(scenario, patrol, width, rng, tries=500):
    """A random campus spot that at least two cameras can see."""
    xy = None
    for _ in range(tries):
        xy = (round(rng.uniform(*CAMPUS_X), 1), round(rng.uniform(*CAMPUS_Y), 1))
        if len(visible_to(scenario, patrol, xy, width)) >= 2:
            break
    return xy


def plan_fire(scenario, patrol, rng, start_range=(5, 15), where=None,
              w_start=10.0, w_max=40.0, grow=20.0, sprite=None):
    return FirePlan(start=rng.randint(int(start_range[0]), int(start_range[1])),
                    xy=tuple(where) if where else random_spot(scenario, patrol, w_max, rng),
                    sprite=sprite or rng.choice(DEFAULT_SPRITES),
                    w_start=min(w_start, w_max), w_max=w_max, grow=grow)


# ---------------------------------------------------------------- watching
@dataclass
class View:
    """One camera, one second: what it saw and what the AI made of it."""
    cam: object
    image: object
    truth: object            # where the renderer put the fire (never shown to the AI)
    boxes: list
    fire_boxes: list
    cols: tuple | None
    note: str


@dataclass
class Sighting:
    t: int
    cam: object              # the camera AS IT WAS POINTING at that moment
    cols: tuple
    conf: float
    view: object = None      # the photo and the AI's boxes

    @property
    def bearings(self):
        return sorted(self.cam.pixel_to_bearing(u) for u in self.cols)

    @property
    def bearing(self):
        return sum(self.bearings) / 2


@dataclass
class Monitor:
    scenario: object
    plan: FirePlan
    patrol: Patrol = field(default_factory=Patrol)
    conf: float = 0.35
    smoke: bool = True
    memory: int = 6          # a sighting counts for this many seconds
    t: int = 0
    views: list = field(default_factory=list)
    sightings: dict = field(default_factory=dict)      # camera name -> latest Sighting
    log: list = field(default_factory=list)            # (t, icon, text)
    alarm: dict | None = None
    source: str = ""
    _unsure: set = field(default_factory=set)

    def say(self, icon, text):
        self.log.append((self.t, icon, text))

    def recent(self):
        return [s for s in self.sightings.values() if self.t - s.t <= self.memory]

    def status(self):
        """(level, text) for the banner: idle / clear / suspect / alarm."""
        if self.alarm:
            a = self.alarm
            return "alarm", (f"FIRE CONFIRMED at {clock(a['t'])} by {' + '.join(a['names'])} — "
                             f"{a['delay']} s after it started")
        rec = self.recent()
        if rec:
            return "suspect", (f"{rec[0].cam.name} sees fire — waiting for a second camera "
                               "to confirm it and cross sight lines")
        if self.t == 0:
            return "idle", "Monitoring is off"
        return "clear", f"All clear — {len(self.scenario.cameras)} cameras checked the campus"

    def tick(self, model=None):
        """One second passes: every camera turns, takes a photo, the AI checks it."""
        self.t += 1
        t = self.t
        cams = self.patrol.cameras_at(self.scenario.cameras, t)
        w = self.plan.width(t)
        if w is not None and self.plan.width(t - 1) is None:
            self.say("🔥", f"A fire starts somewhere on campus ({w:.0f} m wide). "
                           "Nobody has told the system.")
        shots = [render_view(c, self.plan.xy if w else None, w or 0.0, self.plan.sprite,
                             self.smoke, seed=t) for c in cams]
        if model is not None:
            self.source = "live"
            boxes = det.detect_live_batch([img for img, _ in shots], model,
                                          conf=min(self.conf, 0.10))
        else:
            self.source = "geometry"
            boxes = [det.detect_from_truth(tr, c) if tr and tr.width_px >= MIN_PX_NO_AI else []
                     for c, (_, tr) in zip(cams, shots)]

        self.views = []
        for c, (img, tr), bx in zip(cams, shots, boxes):
            fb, cols, note = boxes_to_columns(bx, c, self.conf)
            view = View(c, img, tr, bx, fb, cols, note)
            self.views.append(view)
            if cols:
                conf = max(b.conf for b in fb)
                old = self.sightings.get(c.name)
                s = Sighting(t, c, cols, conf, view)
                self.sightings[c.name] = s
                if old is None or t - old.t > self.memory:
                    self.say("👁", f"{c.name} sees fire to the {compass(s.bearing)} "
                                  f"({conf:.0%} sure).")
            elif fb:
                self.say("✂", f"{c.name} sees fire at the edge of its photo — can't tell "
                              "its width yet.") if c.name not in self._unsure else None
                self._unsure.add(c.name)
            else:
                weak = [b for b in bx if b.label == "fire"]
                if weak and c.name not in self._unsure:
                    self._unsure.add(c.name)
                    self.say("🤔", f"{c.name}: maybe fire ({max(b.conf for b in weak):.0%}) — "
                                  "not sure enough yet.")

        # the alarm: two cameras whose sight lines cross properly
        rec = sorted(self.recent(), key=lambda s: -s.conf)
        for i in range(len(rec)):
            for j in range(i + 1, len(rec)):
                if abs(math.sin(math.radians(rec[i].bearing - rec[j].bearing))) >= 0.2:
                    names = sorted(s.cam.name for s in rec)
                    self.alarm = dict(t=t, xy=self.plan.xy, width=w, names=names,
                                      delay=t - self.plan.start if w else None)
                    self.say("🚨", f"FIRE CONFIRMED by {' + '.join(names)}, "
                                  f"{self.alarm['delay']} s after it started. "
                                  "Locating it and working out the routes.")
                    return self.views
        return self.views

    def alarm_views(self):
        """For each camera: its photo from when it last saw the fire, or -- if
        it didn't -- its latest photo.  The evidence behind the alarm."""
        out = []
        for v in self.views:
            s = self.sightings.get(v.cam.name)
            out.append(s.view if s and self.t - s.t <= self.memory else v)
        return out

    def alarm_cameras(self):
        """Each camera at the heading it had in its alarm_views photo."""
        return [v.cam for v in self.alarm_views()]

    def observed(self):
        """The alarm evidence in the form run_all(observed=...) takes."""
        return self.source, [(v.cam, v.image, v.truth, v.boxes) for v in self.alarm_views()]


# ---------------------------------------------------------------- the live map
# Drawn as a picture (not a Plotly chart) so it can change every second
# without the page rebuilding a chart -- which would flash.
X0, X1, Y0, Y1 = -200, 1000, -300, 520
SURFACE, INK, INK2, CAMERA = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
GRID, WALK, FIRE = "#e1e0d9", "#c3c2b7", "#eb6834"


def live_map(mon, reveal=True, width=960):
    """The control-room view: cameras and where they are looking right now,
    recent fire sightings as orange sight lines, and (if revealed to the
    audience) where the fire really is."""
    ss = 2                                               # draw big, shrink: smooth lines
    sc = width * ss / (X1 - X0)
    W, H = width * ss, round((Y1 - Y0) * sc)

    def px(p):
        return ((p[0] - X0) * sc, (Y1 - p[1]) * sc)

    img = Image.new("RGBA", (W, H), SURFACE)
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d, o = ImageDraw.Draw(img), ImageDraw.Draw(over)
    for x in range(-200, 1001, 200):
        d.line([px((x, Y0)), px((x, Y1))], fill=GRID, width=ss)
    for y in range(-200, 501, 200):
        d.line([px((X0, y)), px((X1, y))], fill=GRID, width=ss)
    for a, b in WALKWAYS:
        d.line([px(NODES[a]), px(NODES[b])], fill=WALK, width=4 * ss)

    cams = [v.cam for v in mon.views] or mon.scenario.cameras
    for c in cams:                                       # where each camera looks now
        o.polygon([px(p) for p in c.fov_polygon(700)], fill=(42, 120, 214, 28),
                  outline=(42, 120, 214, 110))
    for s in mon.recent():                               # fire sightings
        age = mon.t - s.t
        alpha = int(170 * (1 - age / (mon.memory + 1)))
        reach = 1400
        pts = [s.cam.pos] + [(s.cam.x + reach * math.cos(math.radians(a)),
                              s.cam.y + reach * math.sin(math.radians(a))) for a in s.bearings]
        o.polygon([px(p) for p in pts], fill=(235, 104, 52, alpha // 2),
                  outline=(217, 89, 38, alpha))
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)

    from .animate import _font
    f_small, f_big = _font(12 * ss), _font(16 * ss)
    for n, p in NODES.items():
        x, y = px(p)
        r = 5 * ss
        if n == "ASSEMBLY":
            d.rectangle([x - r * 1.5, y - r * 1.5, x + r * 1.5, y + r * 1.5], fill="#0ca30c",
                        outline=INK, width=ss)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=SURFACE, outline=INK, width=ss)
        d.text((x, y - 9 * ss), n, fill=INK2, font=f_small, anchor="ms")
    for c in cams:
        x, y = px(c.pos)
        r = 8 * ss
        d.polygon([(x, y - r), (x - r, y + r * 0.8), (x + r, y + r * 0.8)], fill=CAMERA,
                  outline=SURFACE)
        d.text((x, y + r + 12 * ss), c.name, fill=CAMERA, font=f_small, anchor="ms")

    w = mon.plan.width(mon.t) if mon.t else None
    if reveal and w:
        x, y = px(mon.plan.xy)
        r = max(6, w / 2 * sc)
        d.ellipse([x - r, y - r, x + r, y + r], fill=FIRE, outline=INK, width=ss)
        d.text((x + r + 6 * ss, y), "fire (the system can't see this)", fill=FIRE,
               font=f_small, anchor="lm")
    d.text((14 * ss, 12 * ss), f"⏱ {clock(mon.t)}" if mon.t else "⏱ --:--", fill=INK,
           font=f_big)
    return img.convert("RGB").resize((width, round(H / ss)), Image.LANCZOS)


# ---------------------------------------------------------------- CLI
def main():
    from .cameras import load_scenario
    from .pipeline import Settings, run_all, summary

    ap = argparse.ArgumentParser(description="Simulate live monitoring until a fire is found.")
    ap.add_argument("--scenario", default="campus_watch")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mode", default="live", choices=["live", "geometry"])
    ap.add_argument("--start", nargs=2, type=int, default=(5, 15), metavar=("EARLIEST", "LATEST"))
    ap.add_argument("--sweep", type=float, default=30.0)
    ap.add_argument("--period", type=float, default=20.0)
    ap.add_argument("--max-seconds", type=int, default=120)
    a = ap.parse_args()

    sc = load_scenario(a.scenario)
    rng = random.Random(a.seed)
    patrol = Patrol(a.sweep, a.period)
    plan = plan_fire(sc, patrol, rng, tuple(a.start))
    model = det.load_model() if a.mode == "live" else None
    mon = Monitor(sc, plan, patrol)
    seen = 0
    while not mon.alarm and mon.t < a.max_seconds:
        mon.tick(model)
        for t, icon, text in mon.log[seen:]:
            print(f"{clock(t)}  {icon}  {text}")
        seen = len(mon.log)
    if not mon.alarm:
        print("no alarm -- the fire was never confirmed")
        return
    print(f"\n(the fire really was at {plan.xy}, started at {clock(plan.start)})\n")
    res = run_all(plan.xy, Settings(scenario=a.scenario, fire_width=mon.alarm["width"],
                                    sprite=plan.sprite, conf=mon.conf, mode=a.mode),
                  observed=mon.observed())
    print(summary(res))


if __name__ == "__main__":
    main()
