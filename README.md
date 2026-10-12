# Everblazing

Drop a fire on a campus map. Five cameras "photograph" it, an AI finds the fire
in each photo, the cameras' sight lines cross where the fire is, and the app
works out the safest walking routes to the assembly point. Every step is shown,
animated, and measured.

**Everything is Python.** No HTML, CSS or JavaScript to write.

```
place a fire ─▸ ① Simulate ─▸ ② Detect ─▸ ③ Locate ─▸ ④ Route
                camera photos   YOLO26 AI    sight lines   A* around
                with a real     finds the    cross at the  the danger
                flame pasted in fire         fire
```

## Run it

```bash
conda activate torch                      # or any Python 3.11+ environment
pip install -r requirements.txt
streamlit run app.py                      # the app
python -m firepath.pipeline --fire 423 76 # the same chain, no UI, prints the numbers
python -m pytest -q                       # the checks
```

Then click anywhere on the map to move the fire. Each new fire plays the
story (about 6 seconds) with a caption for each stage:

1. **① Simulate**: the camera photos appear.
2. **② Detect**: the AI's boxes appear on them.
3. **③ Locate**: the wedges sweep out from the cameras and cross at the fire.
4. **④ Route**: the walkways recolour and the escape routes draw themselves.

**▶ RUN PIPELINE** replays the story, and **▶ walk the routes** sets walkers
off along each route. Try the north path near Science: the recommended route
flips from north to south.

### 📡 Live monitoring mode

Switch **Mode** to *Live monitoring* and press **▶ START MONITORING**:

1. Every camera **sweeps** back and forth (±30° by default). Once a second
   each one turns, takes a photo, and the AI checks it. You see this on a
   live map (each camera's view cone moving), in all five camera feeds, and
   in an event log. -- test ---
2. At a **random moment** (5–15 s by default), a fire starts at a random
   spot that two cameras can see. It starts small (6 m) and grows. Nobody
   tells the system. The audience can see it on the live map (you can turn
   this off).
3. Each time a camera's AI spots the fire, an orange **sight line** appears.
   When **two cameras'** sight lines cross, the alarm goes: "FIRE CONFIRMED
   at 00:17 by CAM 1 + CAM 2, 3 s after it started".
4. The story then plays: locate the fire, then reroute everyone. It uses
   each camera **at the heading it had when it saw the fire**, so a turning
   camera gives bearings as exact as a fixed one. A fire doesn't move, so
   sightings from different seconds can be combined.

The same thing runs without the UI, printing the event log and the result:

```bash
python -m firepath.monitor --seed 3              # live AI
python -m firepath.monitor --seed 3 --mode geometry
```

**No PyTorch or no wifi?** The app still works. It replays AI detections saved
in `data/cache/detections.json` (the fire snaps to the nearest 20 m cached
spot). The results panel always says which mode is running:

| Mode | What finds the fire |
|---|---|
| 🟢 LIVE AI | the YOLO26 model runs on each photo (needs `ultralytics`) |
| 🟡 OFFLINE | the model's saved answers for that fire position |
| ⚪ GEOMETRY ONLY | no AI: the simulator's own boxes (used for testing, labelled clearly) |

## How it works

**① Simulate: the localiser run backwards.** The camera guide turns a photo
column into a bearing: `θ = ψ − atan((u − cu) / f)`. Turned around, the same
equation says where the fire lands in the photo: `u = cu + f · tan(ψ − θ)`. It
appears `f · W / r` pixels wide. The scene is drawn, but the fire is a **real
photograph of flames**, because the AI learned from real fire and would ignore
a drawn orange blob. (`firepath/simulate.py`)

**② Detect.** [`SalahALHaismawi/yolov26-fire-detection`](https://huggingface.co/SalahALHaismawi/yolov26-fire-detection)
(YOLO26-S, MIT licence) puts a box round the fire. The box's left and right
edges are the fire's columns: `u_left, u_right = box.x1, box.x2`. That one line
joins the AI to the geometry. (`firepath/detect.py`, `firepath/pipeline.py`)

**③ Locate.** Each camera's two columns become two bearings, which make a
wedge. Where the wedges overlap is the fire region; weighted least squares
gives the best single point. (`firepath/localize.py`, copied unchanged from the
camera guide)

**④ Route.** Walkways near the fire feel longer (up to ×8), and those within
25 m are closed. A* finds the cheapest way out. Closing one walkway at a time
gives genuinely different alternatives, and one backward Dijkstra from the
assembly point puts a signpost at every building. (`firepath/campus.py`,
`firepath/routing.py`, a Python port of the evacuation guide's C++ program that
reproduces its numbers exactly)

## How accurate is it?

The photos are simulated, so we know exactly where every fire really was. On
20 random fires, run through the live AI (`python -m scripts.accuracy_study`):

- located **20 / 20**
- median error **1.1 m**, worst 2.7 m
- the true fire was inside the reported region every time

Full table: [docs/accuracy_study.md](docs/accuracy_study.md).

## What's where

```
app.py                     Streamlit UI (map, photos, results, the maths)
firepath/
  cameras.py               camera poses; world_to_pixel() and pixel_to_bearing()
  simulate.py              render_view(): scene + real flame cut-out
  detect.py                YOLO live / offline cache / geometry fallback
  localize.py              bearings, wedges, triangulation, region (from the guide)
  geometry.py, campus.py   the 16-place campus graph and danger weighting
  routing.py               A*, alternatives, backward Dijkstra
  pipeline.py              run_all(): the whole chain -> one Result
  monitor.py               live mode: sweeping cameras, hidden fire, alarm, live map
  animate.py               Plotly map, animation frames, headless export
scripts/
  smoke_test_detection.py  Phase 0: does the AI see a pasted-in flame? (GO)
  make_sprites.py          cuts the flames out of Wikimedia Commons photos
  build_cache.py           saves the AI's detections for offline mode
  accuracy_study.py        writes docs/accuracy_study.md
data/scenarios/            camera rigs + preset fires (JSON)
assets/fire_sprites/       6 flame cut-outs; sources and licences in CREDITS.md
docs/                      the original guides (PDF) and the accuracy study
tests/                     pytest: C++ numbers, round trip, safety rules, fallback
```

`python -m firepath.pipeline --fire 535 345 --headless out/` also saves every
camera photo with its boxes, a map PNG, and a self-playing `animation.html` for
the written report.

## Honest notes for the report

- **The camera images are simulated.** That's a strength: it is the only way
  to state a measured accuracy.
- The simulated ground is flat and the buildings never hide the fire. A real
  fire behind a building would be missed by that camera.
- The AI finds flames reliably once they are **about 100 px wide** or more in
  a 1920 px photo. Below about 60 px it starts missing them. The camera rig was
  placed so that about 94% of the campus is seen by two cameras at that size.
- Two of the six flame photos (`bonfire_hilton`, `campfire_goforth`) are
  harder for the AI. The default flame is one of the three reliable ones.
