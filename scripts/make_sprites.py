"""
Build the flame cut-outs in assets/fire_sprites/ from real photographs.

Every source is a photo of fire at night, so the flames are the only bright
thing in the picture.  That makes the cut-out easy: brightness becomes
transparency.  Bright flame -> solid, dark background -> see-through.

    python -m scripts.make_sprites

Needs internet once.  The PNGs are committed, so the app never needs this.
"""

import io
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "assets" / "fire_sprites"

# (file name, Wikimedia Commons title, author, licence)
SOURCES = [
    ("campfire_goforth", "Campfire - Flickr - kestingoforth.png",
     "Kestin Goforth", "CC BY 2.0"),
    ("campfire_hameltion", "Campfire September 2017 (cropped).jpg",
     "Hameltion", "CC BY-SA 4.0"),
    ("flame_waleedtariq", "Rare view of flame.jpg",
     "Waleedtariq711", "CC BY-SA 4.0"),
    ("holika_opendraft", "Flames of Holika dahan.jpg",
     "The open draft", "CC BY-SA 4.0"),
    ("osterfeuer_geoprofi", "2019 Osterfeuer Karow bei Nacht.jpg",
     "Geoprofi Lars", "CC BY-SA 4.0"),
    ("bonfire_hilton", "Guy Fawkes bonfire, Carshalton Park (geograph 2681900).jpg",
     "Christopher Hilton", "CC BY-SA 2.0"),
]

USER_AGENT = "FirePathSchoolProject/0.1 (educational; python urllib)"
DOWNLOAD_WIDTH = 1200
MAX_SIDE = 700          # sprites are scaled down in the scene anyway
DARK, BRIGHT = 45, 170  # brightness ramp: below DARK invisible, above BRIGHT solid


def download(title):
    url = ("https://commons.wikimedia.org/wiki/Special:FilePath/"
           + urllib.parse.quote(title) + f"?width={DOWNLOAD_WIDTH}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def cut_out(photo):
    """Turn a night-time fire photo into an RGBA flame sprite."""
    rgb = np.asarray(photo).astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Flames are bright AND warm (more red than blue).  Using max(r, g)
    # instead of plain brightness keeps deep-orange tongues solid.
    bright = np.maximum(r, g)
    alpha = np.clip((bright - DARK) / (BRIGHT - DARK), 0.0, 1.0)
    # Drop blue-ish lights, but keep the white-hot core: a pixel counts if
    # it is warm (red beats blue) or so bright that blue is high too.
    warm = np.clip((r - b) / 40.0, 0.0, 1.0)
    white_hot = np.clip((b - 140.0) / 60.0, 0.0, 1.0)
    alpha *= np.maximum(warm, white_hot)

    a_img = Image.fromarray((alpha * 255).astype(np.uint8))
    a_img = a_img.filter(ImageFilter.GaussianBlur(1.2))   # soften jaggies
    sprite = photo.convert("RGBA")
    sprite.putalpha(a_img)

    # crop to the flame, dropping stray sparks: keep rows/cols that hold
    # a real amount of flame
    a = np.asarray(a_img) > 40
    rows = np.where(a.sum(axis=1) > 0.01 * a.shape[1])[0]
    cols = np.where(a.sum(axis=0) > 0.01 * a.shape[0])[0]
    sprite = sprite.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
    sprite.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return sprite


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    credits = ["# Flame sprite credits", "",
               "Cut out from photographs on Wikimedia Commons by "
               "`scripts/make_sprites.py` (brightness used as transparency).", "",
               "| sprite | source | author | licence |", "|---|---|---|---|"]
    for name, title, author, licence in SOURCES:
        print(f"  {name:22s} <- {title}")
        sprite = cut_out(download(title))
        sprite.save(OUT / f"{name}.png")
        page = "https://commons.wikimedia.org/wiki/File:" + urllib.parse.quote(
            title.replace(" ", "_"))
        credits.append(f"| `{name}.png` | [{title}]({page}) | {author} | {licence} |")
        print(f"  {'':22s}    {sprite.width}x{sprite.height}")
    (OUT / "CREDITS.md").write_text("\n".join(credits) + "\n")
    print(f"wrote {len(SOURCES)} sprites to {OUT}")


if __name__ == "__main__":
    main()
