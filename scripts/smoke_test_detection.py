"""
Phase 0 -- GO / NO-GO: will the AI see a pasted-in flame at all?

Pastes every flame sprite onto a plain daytime background at several sizes,
runs the YOLO fire model, and prints the best "fire" confidence for each.

    python -m scripts.smoke_test_detection

GO     : confidence > 0.5 on most sprites at realistic sizes (>= 60 px wide)
NO-GO  : nothing detects -> use real photographs for the detection stage
"""

from pathlib import Path

from PIL import Image, ImageDraw

from firepath import detect
from firepath.detect import load_model

ROOT = Path(__file__).resolve().parent.parent
SPRITES = sorted((ROOT / "assets" / "fire_sprites").glob("*.png"))
WIDTHS = [30, 45, 60, 100, 200, 400]     # apparent fire width in pixels
IMGSZ = [640, 1280]                      # YOLO input size
W, H = 1920, 1080


def plain_background():
    img = Image.new("RGB", (W, H), (150, 190, 230))            # sky
    ImageDraw.Draw(img).rectangle([0, H // 2, W, H], fill=(95, 125, 80))  # grass
    return img


def composite(sprite, width):
    img = plain_background()
    s = sprite.resize((width, max(1, round(width * sprite.height / sprite.width))),
                      Image.LANCZOS)
    img.paste(s, (W // 2 - s.width // 2, H // 2 + 20 - s.height), s)  # base on horizon
    return img


def main():
    model = load_model()
    if model is None:
        raise SystemExit("ultralytics or the weights are missing -- cannot run Phase 0")

    passed = total = 0
    for imgsz in IMGSZ:
        print(f"\nYOLO imgsz = {imgsz}     best 'fire' confidence (- = not detected)")
        print(f"  {'sprite':22s}" + "".join(f"{w:>7d}px" for w in WIDTHS))
        for path in SPRITES:
            sprite = Image.open(path).convert("RGBA")
            row = []
            for w in WIDTHS:
                res = model.predict(composite(sprite, w), imgsz=imgsz, conf=0.05, device=detect._device,
                                    verbose=False)[0]
                confs = [float(c) for c, k in zip(res.boxes.conf, res.boxes.cls)
                         if model.names[int(k)] == "fire"]
                best = max(confs, default=0.0)
                row.append(best)
                if w >= 60:
                    total += 1
                    passed += best > 0.5
            print(f"  {path.stem:22s}" +
                  "".join(f"{c:9.2f}" if c else f"{'-':>9s}" for c in row))

    rate = passed / total
    print(f"\n{passed}/{total} sprite/size/imgsz combos at >= 60 px scored > 0.5 "
          f"({100 * rate:.0f}%)")
    print("GO" if rate >= 0.5 else "NO-GO")


if __name__ == "__main__":
    main()
