"""
Stage 2 -- find the fire in a photo.

Live mode runs the YOLO26 fire model from HuggingFace
(SalahALHaismawi/yolov26-fire-detection).  If ultralytics or the weights are
missing -- a school laptop without PyTorch, no wifi at the venue -- we fall
back to detections saved earlier by scripts/build_cache.py, and say so.
"""

import json
import threading
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
CACHE = ROOT / "data" / "cache" / "detections.json"
HF_REPO = "SalahALHaismawi/yolov26-fire-detection"

Box = namedtuple("Box", "x1 y1 x2 y2 conf label")

_model = None
_device = None
# Streamlit runs every browser tab (and the live monitor) on its own thread,
# but they share one model -- and a YOLO model must not be used by two
# threads at once (it can crash the whole process).  One at a time.
_lock = threading.Lock()


def _pick_device(model):
    """GPU if it actually works, else CPU.  (Some CUDA installs load fine
    but fail on the first convolution, so we try one tiny prediction.)"""
    try:
        import numpy as np
        import torch
        if torch.cuda.is_available():
            model.predict(np.zeros((64, 64, 3), np.uint8), device=0, verbose=False)
            return 0
    except Exception:
        pass
    return "cpu"


def load_model(download=True):
    """The YOLO model, or None if it cannot be loaded here."""
    with _lock:
        return _load(download)


def _load(download):
    global _model, _device
    if _model is not None:
        return _model
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    path = WEIGHTS
    if not path.exists():
        if not download:
            return None
        try:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(HF_REPO, "best.pt", local_dir=WEIGHTS.parent))
        except Exception:
            return None
    _device = _pick_device(YOLO(str(path)))
    model = YOLO(str(path))           # fresh copy, so the test run leaves no trace
    import numpy as np                # warm it up here, under the lock, so no two
    model.predict(np.zeros((64, 64, 3), np.uint8), device=_device, verbose=False)
    _model = model                    # threads ever race to set it up
    return _model


def detect_live(image, model, conf=0.35, imgsz=1280):
    """Run YOLO on one PIL image and return every box as a Box."""
    with _lock:
        res = model.predict(image, imgsz=imgsz, conf=conf, device=_device,
                            verbose=False)[0]
    return [Box(*map(float, xyxy), float(c), model.names[int(k)])
            for xyxy, c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf, res.boxes.cls)]



def detect_live_batch(images, model, conf=0.35, imgsz=1280):
    """detect_live for several photos in one go (faster: one trip to the GPU)."""
    with _lock:
        results = model.predict(list(images), imgsz=imgsz, conf=conf, device=_device,
                                verbose=False)
    return [[Box(*map(float, xyxy), float(c), model.names[int(k)])
             for xyxy, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf, r.boxes.cls)]
            for r in results]

# ---------------------------------------------------------------- offline cache
# scripts/build_cache.py runs the real model on a grid of fire positions and
# saves every box.  Offline, the fire snaps to the nearest grid point and we
# replay what the AI really said there.
CACHE_STEP = 20.0          # metres between cached fire positions

_cache = None


def _key(scenario, fire_xy, width, sprite):
    return f"{scenario}|{sprite}|{width:g}|{fire_xy[0]:.1f},{fire_xy[1]:.1f}"


def load_cache():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    return _cache


def snap_to_grid(fire_xy):
    return tuple(round(v / CACHE_STEP) * CACHE_STEP for v in fire_xy)


def detect_cached(scenario, fire_xy, width, sprite, cam_names):
    """Saved boxes for this exact fire, or None if it was never cached."""
    entry = load_cache().get(_key(scenario, fire_xy, width, sprite))
    if entry is None:
        return None
    return [[Box(*b) for b in entry.get(n, [])] for n in cam_names]


def is_cached(scenario, fire_xy, width, sprite):
    return _key(scenario, fire_xy, width, sprite) in load_cache()


def cache_entry(boxes_per_cam, cam_names):
    return {n: [list(b) for b in boxes] for n, boxes in zip(cam_names, boxes_per_cam)}


# ---------------------------------------------------------------- no AI at all
def detect_from_truth(truth, cam):
    """Last resort without the AI or a cache: the renderer's own answer.
    Perfect by construction -- the app labels it so it never passes as AI."""
    if truth is None or truth.u_right <= 0 or truth.u_left >= cam.width:
        return []
    return [Box(max(0.0, truth.u_left), max(0.0, truth.v_top),
                min(float(cam.width), truth.u_right), truth.v_base, 1.0, "fire")]
