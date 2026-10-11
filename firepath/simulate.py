"""
Stage 1 -- make the photo each camera would take of the fire.

The scene is drawn (sky, tree line, grass, the campus buildings where they
really are), but the fire is a *real photograph* of flames pasted in.  The AI
was trained on real fire, so a drawn orange blob would fool nobody; a real
flame cut-out, placed where the geometry says, is what it sees.

Where to paste it is the localiser run backwards (see cameras.py):

    theta = atan2(fire_y - cam_y, fire_x - cam_x)     bearing to the fire
    u     = cu + f * tan(psi - theta)                 its image column
    w_px  = f * fire_width / r                        its width in pixels
"""

import functools
import io
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .campus import NODES

SPRITE_DIR = Path(__file__).resolve().parent.parent / "assets" / "fire_sprites"
# The three cut-outs the AI finds most reliably at small sizes (Phase 0).
DEFAULT_SPRITES = ("holika_opendraft", "campfire_hameltion", "flame_waleedtariq")

BUILDING_W, BUILDING_H = 45.0, 16.0     # metres
SKIP_BUILDINGS = {"WestLot", "SouthLot", "Field", "ASSEMBLY", "Plaza"}   # open spaces


@dataclass
class FireTruth:
    """What the renderer knows and the AI has to discover."""
    u_left: float
    u_right: float
    v_top: float
    v_base: float
    range_m: float
    in_frame: bool       # whole fire inside the photo?

    @property
    def width_px(self):
        return self.u_right - self.u_left


def list_sprites():
    return sorted(p.stem for p in SPRITE_DIR.glob("*.png"))


@functools.lru_cache(maxsize=None)
def load_sprite(name):
    return Image.open(SPRITE_DIR / f"{name}.png").convert("RGBA")


# ---------------------------------------------------------------- scenery
# The tree line's height is a fixed wobbly function of compass bearing (a sum
# of sine waves), so every camera sees the same hills in the same directions.
_TREE_K = np.arange(1, 120)
_TREE_RNG = np.random.default_rng(7)
_TREE_AMP = _TREE_RNG.standard_normal(len(_TREE_K)) / _TREE_K ** 0.7
_TREE_PHASE = _TREE_RNG.uniform(0, 2 * np.pi, len(_TREE_K))
_TREE_SD = math.sqrt((_TREE_AMP ** 2).sum() / 2)


def _treeline(cam):
    """Tree-line height (in standard units) for every image column."""
    u = np.arange(cam.width)
    theta = math.radians(cam.psi) - np.arctan((u - cam.cu) / cam.f)   # bearing of each column
    h = (np.sin(np.outer(theta, _TREE_K) + _TREE_PHASE) * _TREE_AMP).sum(axis=1) / _TREE_SD
    return np.clip(h, -1.5, 3.0)


@functools.lru_cache(maxsize=64)
def _background(cam):
    """Everything except the fire.  Cached per camera pose."""
    W, H = cam.width, cam.height
    hz = int(round(cam.cv))                          # level camera: horizon mid-frame
    rng = np.random.default_rng(zlib.crc32(cam.name.encode()))  # same scene every run

    img = np.zeros((H, W, 3), np.float32)
    t = np.linspace(0, 1, hz)[:, None, None]
    img[:hz] = (1 - t) * np.array([95, 140, 200]) + t * np.array([190, 210, 225])  # sky
    t = np.linspace(0, 1, H - hz)[:, None, None]
    img[hz:] = (1 - t) * np.array([120, 135, 95]) + t * np.array([70, 100, 55])     # grass
    img[hz:] += rng.normal(0, 6, (H - hz, W, 1))                                    # texture

    # distant tree line hugging the horizon -- fixed to the world, so it
    # slides past when the camera turns
    ridge = hz - 12 - 7 * _treeline(cam)
    rows = np.arange(H)[:, None]
    trees = (rows >= ridge[None, :]) & (rows < hz)
    img[trees] = np.array([55, 75, 50]) + rng.normal(0, 5, (trees.sum(), 1))

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    # campus buildings, far ones first so near ones cover them
    shade = random.Random(cam.name)
    blds = []
    for name, p in NODES.items():
        if name in SKIP_BUILDINGS:
            continue
        r = math.dist(p, cam.pos)
        u = cam.world_to_pixel(p)
        if u is None or r < 30:
            continue
        blds.append((r, u, name))
    for r, u, name in sorted(blds, reverse=True):
        w = cam.f * BUILDING_W / r
        h = cam.f * BUILDING_H / r
        base = cam.ground_row(r)
        x0, x1, y0 = u - w / 2, u + w / 2, base - h
        if x1 < 0 or x0 > W:
            continue
        g = shade.randint(150, 200)
        wall = (g, g - 8, g - 20)                     # pale concrete / sandstone
        draw.rectangle([x0, y0, x1, base], fill=wall)
        draw.rectangle([x0, y0 - 0.06 * h, x1, y0], fill=(95, 95, 100))    # roof edge
        if w > 40:                                    # windows when near enough
            nx, ny = max(2, int(w / 18)), 3
            for i in range(nx):
                for j in range(ny):
                    wx = x0 + (i + 0.25) * w / nx
                    wy = y0 + (j + 0.25) * h / (ny + 0.3)
                    draw.rectangle([wx, wy, wx + 0.5 * w / nx, wy + 0.45 * h / (ny + 0.3)],
                                   fill=(70, 85, 105))
    return pil


# ---------------------------------------------------------------- the fire
def fire_truth(cam, fire_xy, fire_width, sprite="holika_opendraft"):
    """Where the fire lands in this camera's photo -- or None if behind it."""
    cols = cam.fire_columns(fire_xy, fire_width)
    if cols is None:
        return None
    s = load_sprite(sprite)
    r = math.dist(fire_xy, cam.pos)
    w = cols[1] - cols[0]
    v_base = cam.ground_row(r)
    v_top = v_base - w * s.height / s.width
    in_frame = cols[0] >= 0 and cols[1] <= cam.width and v_top >= 0
    return FireTruth(cols[0], cols[1], v_top, v_base, r, in_frame)


def _smoke(size, rng):
    """A soft grey plume: blurred overlapping puffs drifting up and sideways."""
    w, h = size
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    drift = rng.uniform(-0.25, 0.25)
    for k in range(14):
        t = k / 13
        cx = w / 2 + drift * w * t + rng.uniform(-0.05, 0.05) * w
        cy = h * (1 - t)
        rad = w * (0.12 + 0.18 * t)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=int(150 * (1 - 0.6 * t)))
    return layer.filter(ImageFilter.GaussianBlur(max(2, w / 25)))


def render_view(cam, fire_xy=None, fire_width=40.0, sprite="holika_opendraft",
                smoke=True, seed=0):
    """The photo this camera takes.  Returns (PIL image, FireTruth or None)."""
    img = _background(cam).copy()
    truth = fire_truth(cam, fire_xy, fire_width, sprite) if fire_xy else None
    if truth and truth.u_right > 0 and truth.u_left < cam.width and truth.width_px >= 2:
        w = max(2, int(round(truth.width_px)))
        s = load_sprite(sprite)
        h = max(2, int(round(w * s.height / s.width)))
        x0, y0 = int(round(truth.u_left)), int(round(truth.v_base)) - h
        if smoke:
            rng = random.Random(seed)
            sw, sh = int(w * 1.6), int(h * 1.8)
            plume = _smoke((sw, sh), rng)
            grey = Image.new("RGB", (sw, sh), (105, 105, 110))
            img.paste(grey, (x0 + w // 2 - sw // 2, y0 - sh + int(0.35 * h)), plume)
        img.paste(s.resize((w, h), Image.LANCZOS), (x0, y0), s.resize((w, h), Image.LANCZOS))

    # make it look like a camera took it: a little sensor noise, JPEG'd
    arr = np.asarray(img).astype(np.int16)
    arr += np.random.default_rng(seed).integers(-4, 5, arr.shape, dtype=np.int16)
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=88)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), truth
