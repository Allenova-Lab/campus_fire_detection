"""
FirePath -- see it, find it, escape it.

    streamlit run app.py

Two ways to use it:

* Place a fire -- click the map (or pick a preset).  Each camera "photographs"
  it, the AI finds the fire in every photo, the sight lines cross where the
  fire is, and the walking routes out are recomputed around it.
* Live monitoring -- the cameras sweep and check the campus every second; a
  fire starts at a random time and place; when two cameras have seen it the
  alarm goes and the same chain runs.
"""

import faulthandler
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import plotly.offline
import streamlit as st

from firepath import animate as anim
from firepath import detect as det
from firepath import monitor as live
from firepath.campus import GOAL, NODES
from firepath.cameras import list_scenarios, load_scenario
from firepath.pipeline import Settings, run_all
from firepath.simulate import DEFAULT_SPRITES, list_sprites

st.set_page_config(page_title="Everblazing", page_icon="🔥", layout="wide")
faulthandler.enable()     # if Python ever crashes hard, print where every thread was

STAGES = ["① Simulate", "② Detect", "③ Locate", "④ Route"]
MAP_H = 560               # map height (px) -- the story and the clickable map match
LOAD_S = 0.3              # seconds the story page takes to start

# The story page loads Plotly from this app (static/ is served at app/static/),
# so it works without internet and the browser keeps it cached between runs.
_PLOTLY_FILE = Path(__file__).parent / "static" / "plotly.min.js"
if not _PLOTLY_FILE.exists():
    _PLOTLY_FILE.parent.mkdir(exist_ok=True)
    _PLOTLY_FILE.write_text(plotly.offline.get_plotlyjs(), encoding="utf-8")
PLOTLY_JS = "app/static/plotly.min.js"


@st.cache_resource(show_spinner="Loading the fire-detection AI…")
def get_model():
    return det.load_model(download=False)


# ------------------------------------------------------------------ sidebar
ss = st.session_state
with st.sidebar:
    app_mode = st.radio("Mode", ["🎯 Place a fire", "📡 Live monitoring"], horizontal=True,
                        help="Live monitoring: the cameras check the campus every second and a "
                             "fire starts at a random time and place.")
    is_live = app_mode.startswith("📡")
    st.header("Settings")
    scen_names = list_scenarios()
    scen_name = st.selectbox("Scenario", scen_names,
                             index=scen_names.index("campus_watch"),
                             format_func=lambda n: load_scenario(n).title)
    sc = load_scenario(scen_name)
    if ss.get("scenario") != scen_name:                # new scenario: reset
        ss.scenario = scen_name
        ss.fire = next(iter(sc.fires.values()))
        ss.result = None
    st.caption(sc.note)

    if is_live:
        st.subheader("🔥 The hidden fire")
        start_range = st.slider("Starts after (seconds)", 2, 60, (5, 15),
                                help="A random moment in this window.")
        where = st.selectbox("Where", ["Random spot"] + list(sc.fires),
                             help="Random spots are always ones two cameras can see.")
        w_start = st.slider("Size when it starts (m)", 4, 40, 6)
        fire_width = st.slider("Full size (m)", 20, 80, int(sc.fire_width), 5)
        grow = st.slider("Seconds to reach full size", 0, 60, 20)
        sprite_pick = st.selectbox("Flame photo", ["Random"] + list_sprites())
        sprite = DEFAULT_SPRITES[0] if sprite_pick == "Random" else sprite_pick
        reveal = st.toggle("Show the audience where the fire is", True)
    else:
        st.subheader("🔥 Fire")
        preset = st.selectbox("Preset", ["(click the map)"] + list(sc.fires))
        if preset != "(click the map)" and ss.get("preset") != preset:
            ss.fire = sc.fires[preset]
            ss.result = None
        ss.preset = preset
        fire_width = st.slider("Fire width (m)", 20, 80, int(sc.fire_width), 5,
                               help="Wider fires look bigger in the photos, so the AI finds "
                                    "them from further away.")
        sprite = st.selectbox("Flame photo", list_sprites(),
                              index=list_sprites().index(DEFAULT_SPRITES[0]))
    smoke = st.checkbox("Add smoke", True)

    st.subheader("📷 Cameras")
    conf = st.slider("AI confidence needed", 0.10, 0.90, 0.35, 0.05)
    if is_live:
        sweep = st.slider("Each camera sweeps ± (degrees)", 0, 60, 30, 5,
                          help="0 = fixed cameras.")
        period = st.slider("Seconds per sweep", 8, 60, 20, 2)
    mode_label = st.radio("Detection", ["Auto", "Live AI", "Offline cache", "Geometry only"],
                          help="Auto: live AI if PyTorch is installed, else the offline cache, "
                               "else the renderer's own answer.  (Live monitoring needs the AI "
                               "or falls back to geometry only.)")
    mode = {"Auto": "auto", "Live AI": "live", "Offline cache": "cached",
            "Geometry only": "geometry"}[mode_label]

    st.subheader("🚧 Danger")
    block = st.slider("Closed within (m)", 5, 80, 25, 5)
    safe = st.slider("Ignored beyond (m)", 100, 400, 200, 10)
    worst = st.slider("Worst slow-down (×)", 2.0, 20.0, 8.0, 0.5)

    st.subheader("🚶 Start")
    places = [n for n in NODES if n != GOAL]
    start = st.selectbox("Walking from", places, index=places.index("DormA"))

    animate_on = st.toggle("Play the story on each new fire", True)
    show_maths = st.toggle("Show the maths", False)
    if st.button("↻ Reset", width="stretch"):
        for k in ("fire", "result", "last_click", "preset", "monitor", "live_running"):
            ss.pop(k, None)
        st.rerun()

settings = Settings(scenario=scen_name, fire_width=float(fire_width), sprite=sprite,
                    smoke=smoke, conf=conf, mode=mode, block_radius=float(block),
                    safe_radius=float(safe), max_danger=float(worst), start=start)
ss.setdefault("fire", next(iter(sc.fires.values())))

# a click on the map arrives here, on the rerun it triggers
event = ss.get("map")
if event and event.get("selection", {}).get("points"):
    p = event["selection"]["points"][0]
    click = (round(float(p["x"]), 1), round(float(p["y"]), 1))
    if click != ss.get("last_click"):
        ss.last_click = click
        ss.fire = click
        ss.result = None

# ------------------------------------------------------------------ header
left, right = st.columns([5, 1.3])
left.title("Everblazing")
if is_live:
    mon = ss.get("monitor")
    if ss.get("live_running"):
        if right.button("■ STOP", width="stretch"):
            ss.live_running = False
            st.rerun()
        right.caption("Checking every second…")
    else:
        if right.button("▶ START MONITORING", type="primary", width="stretch"):
            patrol = live.Patrol(float(sweep), float(period))
            plan = live.plan_fire(sc, patrol, random.Random(),
                                  start_range, None if where == "Random spot" else sc.fires[where],
                                  w_start=float(w_start), w_max=float(fire_width), grow=float(grow),
                                  sprite=None if sprite_pick == "Random" else sprite)
            ss.monitor = live.Monitor(sc, plan, patrol, conf=conf, smoke=smoke)
            ss.live_running = True
            st.rerun()
        right.caption("The cameras will check the campus every second.")
    run_clicked = bool(mon and mon.alarm) and right.button("↻ replay the story", width="stretch")
else:
    run_clicked = right.button("▶ RUN PIPELINE", type="primary", width="stretch")
    right.caption(f"Fire at ({ss.fire[0]:.0f}, {ss.fire[1]:.0f}) m — click the map to move it")


# ------------------------------------------------------------------ live panel
BANNER = {"idle": st.info, "clear": st.success, "suspect": st.warning, "alarm": st.error}
ICON = {"idle": "⏸", "clear": "🟢", "suspect": "🟡", "alarm": "🚨"}


def draw_live(mon):
    """The control room: status, live map, every camera's feed, the event log."""
    level, text = mon.status()
    BANNER[level](f"{ICON[level]} **⏱ {live.clock(mon.t)}** · {text}")
    # after the alarm this folds away, so the story below is on screen
    body = (st.expander("📡 What the cameras saw (the live map and every feed)", expanded=False)
            if mon.alarm else st.container())
    with body:
        left_col, right_col = st.columns([1.15, 1])
        left_col.image(live.live_map(mon, reveal=reveal), width="stretch", output_format="PNG")
        feeds = right_col.columns(2)
        for k, v in enumerate(mon.views):
            with feeds[k % 2]:
                img = anim.annotate_photo(v, mon.conf)
                st.image(img.resize((480, 270)), width="stretch", output_format="JPEG")
                seen = (f"🔥 fire {max(b.conf for b in v.fire_boxes):.2f}" if v.cols
                        else ("✂ fire at the edge" if v.fire_boxes else "clear"))
                st.caption(f"**{v.cam.name}** · facing {live.compass(v.cam.psi)} "
                           f"({v.cam.psi % 360:.0f}°) · {seen}")
    with st.container(height=190 if not mon.alarm else 110):
        for t, icon, line in reversed(mon.log[-40:]):
            st.markdown(f"`{live.clock(t)}` {icon} {line}")
        if not mon.log:
            st.caption("Nothing yet.")


def live_panel():
    mon = ss.get("monitor")
    if mon is None:
        st.info("⏸ Press **▶ START MONITORING**. The cameras start sweeping and checking the "
                "campus every second; a fire will start at a random moment "
                f"{start_range[0]}–{start_range[1]} s later.")
        idle = live.Monitor(sc, live.FirePlan(start=10**9, xy=(0, 0), sprite=sprite))
        st.image(live.live_map(idle, reveal=False), width=720, output_format="PNG")
        return
    if ss.get("live_running") and not mon.alarm:
        ai = get_model() if mode in ("auto", "live") else None
        mon.tick(ai)
        if mon.alarm:                       # hand over to the pipeline and its story
            ss.live_running = False
            st.rerun(scope="app")
    draw_live(mon)


if is_live:
    # only this panel reruns every second (a Streamlit fragment); the rest of
    # the page stays still
    st.fragment(run_every=1.0 if ss.get("live_running") else None)(live_panel)()
    mon = ss.get("monitor")
    if not (mon and mon.alarm):
        st.stop()

# ------------------------------------------------------------------ run
# Re-run whenever the fire or a setting changes -- BEFORE drawing anything
# below, so the old map and photos stay on screen until the new ones replace
# them.  A new fire (or RUN) also plays the story; a slider nudge just updates.
if is_live:
    # the fire the monitor confirmed, seen by each camera at the heading it
    # had when it spotted it
    fire_xy, observed = mon.alarm["xy"], mon.observed()
    settings = replace(settings, fire_width=mon.alarm["width"], sprite=mon.plan.sprite,
                       conf=mon.conf,
                       mode="geometry" if mon.source == "geometry" else mode)
    token = ("live", id(mon), mon.alarm["t"])
else:
    fire_xy, observed, token = ss.fire, None, ("manual", tuple(ss.fire))
run_key = (settings, token)
play = False
if run_clicked or ss.get("result") is None or ss.get("run_key") != run_key:
    play = animate_on and (run_clicked or ss.get("result") is None
                           or ss.get("run_key", (None, None))[1] != token)
    try:
        model = get_model() if settings.mode in ("auto", "live") else None
        with right, st.spinner("Running…"):
            ss.result = run_all(fire_xy, settings, model=model, observed=observed)
        ss.run_key = run_key
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
res = ss.result


def narration(res, stage):
    """What is happening at each stage, in words, with this run's numbers."""
    g = res.graph
    if stage == 1:
        return (f"**① Simulate** — {len(res.cams)} cameras photograph the campus. "
                "The fire in each photo is a real flame photograph, pasted exactly where "
                "the geometry says that camera would see it.")
    if stage == 2:
        seen = [f"{c.cam.name} ({max(b.conf for b in c.fire_boxes):.0%} sure)"
                for c in res.cams if c.cols]
        return ("**② Detect** — the AI (YOLO26) looks for fire in every photo: "
                + (f"found it in {', '.join(seen)}." if seen else "found no usable fire."))
    if stage == 3:
        if not res.located:
            return f"**③ Locate** — {res.message}"
        return (f"**③ Locate** — each fire box becomes a wedge of directions from its camera; "
                f"where the wedges cross is the fire: found at ({res.estimate[0]:.0f}, "
                f"{res.estimate[1]:.0f}), **{res.error_m:.1f} m** from where we put it.")
    if not res.routes:
        return "**④ Route** — no safe route out from here!"
    if not res.located and res.used:
        return ("**④ Route** — ⚠️ **the fire was seen but could not be located, so these "
                "routes do NOT avoid it.** Wait for a second camera or evacuate by the "
                "signposts.")
    r = res.routes[0]
    closed = sum(w.danger == math.inf for w in g.walkways)
    return (f"**④ Route** — walkways near the fire get slower (darker) and {closed} are "
            f"closed (dotted). A* finds the best way out from {res.settings.start}: "
            f"{' → '.join(g.name[v] for v in r.path)} ({r.metres:.0f} m), "
            f"plus {len(res.routes) - 1} backup route(s).")


def show_strip(slots, stage):
    for i, (slot, name) in enumerate(zip(slots, STAGES), start=1):
        done = i < stage or stage > 4
        slot.progress(1.0 if done else (0.5 if i == stage else 0.0),
                      text=f"**{name}**" + (" ✓" if done else (" …" if i == stage else "")))


def photo(cr, boxes):
    """A camera photo with (or without) the AI's boxes, shrunk for the page."""
    img = anim.annotate_photo(cr, res.settings.conf, show_boxes=boxes)
    return img.resize((640, round(640 * img.height / img.width)))


def show_photos(slots, res, images, boxes):
    for slot, cr, img in zip(slots, res.cams, images):
        with slot.container():
            st.image(img, width="stretch", output_format="JPEG")
            if not boxes:
                st.caption(f"**{cr.cam.name}** · photo taken")
            elif cr.cols:
                best = max(b.conf for b in cr.fire_boxes)
                st.caption(f"**{cr.cam.name}** · fire {best:.2f} · "
                           f"cols {cr.cols[0]:.0f}–{cr.cols[1]:.0f}")
            else:
                st.caption(f"**{cr.cam.name}** · {cr.note}")


def show_map(where):
    """The clickable map -- a click on it reruns the app with the fire moved there."""
    fig = anim.final_figure(res, clickable=True, height=MAP_H)
    # a click "selects" a point, and Plotly would dim everything else -- don't
    fig.update_traces(selected=dict(marker=dict(opacity=1)),
                      unselected=dict(marker=dict(opacity=1)), selector=dict(type="scatter"))
    # the last trace is the invisible click grid: keep it invisible
    fig.data[-1].update(unselected=dict(marker=dict(opacity=0.01)),
                        selected=dict(marker=dict(opacity=0.01)))
    # theme=None: draw exactly as designed, so it matches the story page pixel for pixel
    where.plotly_chart(fig, key="map", on_select="rerun", selection_mode="points",
                       width="stretch", theme=None, config={"displayModeBar": False})


# While the story plays, stage bars, caption, map and photos are placeholders
# that the timeline at the end fills in.  Everything they need is prepared
# first, so no placeholder ever sits empty.  Without the story, everything is
# drawn straight away, in place, so nothing on screen blinks.
boxed = [photo(cr, True) for cr in res.cams]
if play:
    story = anim.story_states(res)
    page = anim.story_html(res, PLOTLY_JS, height=MAP_H)
    raw = [photo(cr, False) for cr in res.cams]
    strip = [c.empty() for c in st.columns(4)]
    narr = st.empty()
    map_slot = st.empty()
    photo_slots = [c.empty() for c in st.columns(len(res.cams))]
    show_strip(strip, 1)
    narr.markdown(narration(res, 1))
    map_slot.iframe(page, height=MAP_H)            # our own page, made by Plotly
    t0 = time.time() + LOAD_S                  # the page needs a moment to start
    show_photos(photo_slots, res, raw, boxes=False)
else:
    show_strip([c for c in st.columns(4)], 5)
    st.markdown(narration(res, 4) if res.routes else narration(res, 3))
    show_map(st)
    show_photos([c.empty() for c in st.columns(len(res.cams))], res, boxed, boxes=True)

if res.source == "cached" and tuple(res.fire_true) != tuple(ss.fire):
    st.info(f"Offline mode: the fire snapped to the nearest cached spot "
            f"({res.fire_true[0]:.0f}, {res.fire_true[1]:.0f}).")
results_slot = st.container()

# ------------------------------------------------------------------ results
with results_slot:
    src = {"live": "🟢 LIVE AI (YOLO26)", "cached": "🟡 OFFLINE — replaying saved AI detections",
           "geometry": "⚪ GEOMETRY ONLY — no AI; boxes come from the simulator"}[res.source]
    st.markdown(f"**Detection mode:** {src}")
    if is_live:
        a, plan = mon.alarm, mon.plan
        st.markdown(f"**Live monitoring:** the fire started at `{live.clock(plan.start)}` and was "
                    f"confirmed at `{live.clock(a['t'])}` by {' + '.join(a['names'])} — "
                    f"**{a['delay']} s to detect**, while the cameras were sweeping.")
    if res.message:
        st.warning(res.message)
    if res.located:
        m = st.columns(4)
        m[0].metric("Fire found at", f"({res.estimate[0]:.1f}, {res.estimate[1]:.1f})",
                    help=f"We placed it at ({res.fire_true[0]:.1f}, {res.fire_true[1]:.1f})")
        m[1].metric("Error", f"{res.error_m:.1f} m",
                    delta="✓ within 5 m" if res.error_m < 5 else "✗ over 5 m",
                    delta_color="normal" if res.error_m < 5 else "inverse")
        m[2].metric("Fire region", f"{res.region_ha:.2f} ha", f"{len(res.used)} cameras crossed",
                    delta_color="off")
        if res.dof:
            ok = res.chi2 < res.chi2_limit
            m[3].metric("Cameras agree?", f"{res.chi2:.1f}",
                        delta=f"{'✓' if ok else '✗'} limit {res.chi2_limit}",
                        delta_color="normal" if ok else "inverse",
                        help="χ² test: how well the sight lines agree on one point.")
        else:
            m[3].metric("Cameras agree?", "—", "needs 3+ cameras", delta_color="off")

    st.markdown("#### Ways out")
    g = res.graph
    if not res.routes:
        st.error("No safe route from here — stay put and signal for help.")
    base = res.routes[0].cost if res.routes else 1
    rows = []
    for i, r in enumerate(res.routes):
        rows.append({
            "": anim.ROUTE_ICONS[i],
            "Route": f"Route {i + 1}",
            "Advice": anim.ROUTE_LABELS[i],
            "Path": " → ".join(g.name[v] for v in r.path),
            "Walk (m)": round(r.metres),
            "Feels like (m)": round(r.cost),
            "Worst danger": f"×{r.worst_danger:.1f}",
            "Extra effort": "—" if i == 0 else f"+{100 * (r.cost / base - 1):.0f}%",
        })
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    nf = res.no_fire_route
    st.caption(f"With no fire, the best way from {res.settings.start} would be "
               f"{' → '.join(g.name[v] for v in nf.path)} ({nf.metres:.0f} m).  "
               "Press **▶ walk the routes** on the map to set the walkers off, or "
               + ("**↻ replay the story** " if is_live else "**▶ RUN PIPELINE** ")
               + "to replay the whole story.")
    t = res.timings
    st.caption("Timings: " + " · ".join(f"{k} {v * 1000:.0f} ms" for k, v in t.items()))

    with st.expander("🔍 How did A* search?  (and why not just Dijkstra?)"):
        st.plotly_chart(anim.search_figure(res), width="stretch",
                        config={"displayModeBar": False})
        if res.routes:
            st.markdown(
                f"A* examined **{res.routes[0].expanded}** places; Dijkstra examined "
                f"**{res.dijkstra.expanded}**.  Both find the same cheapest route "
                f"({res.routes[0].cost:.0f} vs {res.dijkstra.cost:.0f} effective m) — A* just "
                "ignores places that point away from the goal, because its hint (the "
                "straight-line distance still to go) never overestimates.")

    with st.expander("🪧 Signposts from every place (one search from the assembly point)"):
        sign_rows = []
        for v in sorted(range(len(g.name)), key=lambda v: -res.plan.cost[v]):
            if g.name[v] == GOAL:
                continue
            c = res.plan.cost[v]
            sign_rows.append({"You are at": g.name[v],
                              "Go to": "—" if c == math.inf else g.name[res.plan.next_hop[v]],
                              "Effort (m)": "NO SAFE ROUTE" if c == math.inf else f"{c:.0f}"})
        st.dataframe(sign_rows, hide_index=True, width="stretch")

# ------------------------------------------------------------------ the maths
if show_maths:
    st.divider()
    st.subheader("The maths, with this run's numbers")
    st.markdown("**1 · Simulate — the localiser run backwards.** For each camera, the bearing "
                "to the fire becomes a pixel column:")
    st.latex(r"\theta=\operatorname{atan2}(y_f-y_c,\;x_f-x_c),\quad "
             r"u=c_u+f\tan(\psi-\theta),\quad w_{px}=\frac{f\,W}{r}")
    tbl = []
    for cr in res.cams:
        tr = cr.truth
        tbl.append({"camera": cr.cam.name, "ψ (deg)": cr.cam.psi, "f (px)": round(cr.cam.f, 1),
                    "range r (m)": round(tr.range_m) if tr else "—",
                    "true cols": f"{tr.u_left:.0f}–{tr.u_right:.0f}" if tr else "behind",
                    "in frame": "yes" if tr and tr.in_frame else "no"})
    st.dataframe(tbl, hide_index=True, width="stretch")
    st.markdown("**2 · Detect — the bridge.** The AI's box edges *are* the fire's columns: "
                "`u_left, u_right = box.x1, box.x2` (several boxes → their union).")
    st.markdown("**3 · Locate — the same equation forwards.** Each column becomes a bearing; "
                "the two bearings of a box make a wedge:")
    st.latex(r"\theta=\psi-\arctan\frac{u-c_u}{f}")
    tbl = []
    for cr in res.used:
        tbl.append({"camera": cr.cam.name, "cols": f"{cr.cols[0]:.1f}–{cr.cols[1]:.1f}",
                    "wedge (deg)": f"{cr.wedge[0]:.3f} … {cr.wedge[1]:.3f}",
                    "centre θ (deg)": round(cr.theta, 3),
                    "σθ (deg)": round(math.degrees(cr.sigma), 4)})
    if tbl:
        st.dataframe(tbl, hide_index=True, width="stretch")
    if res.located:
        st.markdown(f"Weighted least squares over the sight lines gives "
                    f"**({res.estimate[0]:.2f}, {res.estimate[1]:.2f})**; clipping the wedges "
                    f"against each other leaves a {len(res.region)}-corner region of "
                    f"**{res.region_ha * 1e4:.0f} m²**.")
    st.markdown("**4 · Route — danger makes walkways longer.** A walkway d metres from the "
                "fire costs")
    st.latex(r"\text{cost}=\text{length}\times\Big(1+(k_{\max}-1)\,"
             r"\frac{R_{\text{safe}}-d}{R_{\text{safe}}-R_{\text{block}}}\Big)")
    st.markdown(f"with R_block = {block} m (closed), R_safe = {safe} m (ignored), "
                f"k_max = {worst}.  A* then ranks places by *f = g + h*.")


# ------------------------------------------------------------------ the story
# Played last, so the results above are already up to date while it runs.
# The map part is a self-playing copy of the map (a Streamlit chart cannot
# start an animation by itself); the stage bars, caption and photos below
# are stepped in time with it; then the clickable map takes over, showing
# exactly the picture the story ended on.
if play:
    starts = anim.stage_starts(story)
    for stage in (2, 3, 4):
        time.sleep(max(0.0, t0 + starts[stage] - time.time()))
        show_strip(strip, stage)
        narr.markdown(narration(res, stage))
        if stage == 2:
            show_photos(photo_slots, res, boxed, boxes=True)
    time.sleep(max(0.0, t0 + len(story) * anim.FRAME_MS / 1000 - time.time()))
    show_strip(strip, 5)
    show_map(map_slot)
