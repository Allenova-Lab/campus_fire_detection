"""
Precompute the AI's detections so the app works with no PyTorch and no wifi.

Runs the real YOLO model on every camera's photo for a fire at every point of
a 20 m grid over the campus (plus each scenario's preset fires), and saves
every box to data/cache/detections.json.  Offline, the app snaps the fire to
the nearest grid point and replays these -- still genuine AI output.

    python -m scripts.build_cache                 # all scenarios
    python -m scripts.build_cache campus_watch
"""

import json
import sys
import time

from firepath import detect as det
from firepath.cameras import list_scenarios, load_scenario
from firepath.simulate import DEFAULT_SPRITES, render_view

X_RANGE = (0, 920)          # campus plus a margin (m)
Y_RANGE = (-220, 440)
KEEP_CONF = 0.05            # store weak boxes too; the app filters by its slider


def grid():
    s = det.CACHE_STEP
    xs = [X_RANGE[0] + i * s for i in range(int((X_RANGE[1] - X_RANGE[0]) / s) + 1)]
    ys = [Y_RANGE[0] + j * s for j in range(int((Y_RANGE[1] - Y_RANGE[0]) / s) + 1)]
    return [(x, y) for x in xs for y in ys]


def main(names):
    model = det.load_model()
    if model is None:
        raise SystemExit("needs ultralytics and the weights")
    cache = det.load_cache()
    sprite = DEFAULT_SPRITES[0]
    for name in names:
        sc = load_scenario(name)
        fires = list(sc.fires.values()) + grid()
        t0 = time.time()
        for k, xy in enumerate(fires):
            shots = [render_view(cam, xy, sc.fire_width, sprite) for cam in sc.cameras]
            boxes = [det.detect_live(img, model, conf=KEEP_CONF) for img, _ in shots]
            cache[det._key(name, xy, sc.fire_width, sprite)] = det.cache_entry(
                [[b._replace(conf=round(b.conf, 4), x1=round(b.x1, 1), y1=round(b.y1, 1),
                             x2=round(b.x2, 1), y2=round(b.y2, 1)) for b in bs] for bs in boxes],
                [c.name for c in sc.cameras])
            if k % 100 == 0:
                print(f"  {name}: {k}/{len(fires)}  ({time.time() - t0:.0f} s)", flush=True)
        print(f"{name}: {len(fires)} fire positions cached")
    det.CACHE.parent.mkdir(parents=True, exist_ok=True)
    det.CACHE.write_text(json.dumps(cache, separators=(",", ":")))
    print(f"wrote {det.CACHE} ({det.CACHE.stat().st_size / 1e6:.1f} MB, {len(cache)} entries)")


if __name__ == "__main__":
    main(sys.argv[1:] or list_scenarios())
