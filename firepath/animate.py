"""
Pictures of a pipeline Result: the campus map (Plotly), its animation frames,
the A*-vs-Dijkstra search reveal, annotated camera photos, and a headless
export (PNGs + an animated HTML) for the written report.

Every map picture is built by one function, map_figure(), from a MapState that
says how far each animation has got -- so the live app, the exported HTML and
the PNG all draw exactly the same thing.
"""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from PIL import ImageDraw, ImageFont

from .campus import INF, NODES
from .localize import polygon_centroid

# ---------------------------------------------------------------- colours
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, WALKWAY = "#e1e0d9", "#c3c2b7"
CAMERA = "#2a78d6"
WEDGE_FILL = "rgba(42,120,214,0.16)"
FOV_FILL = "rgba(42,120,214,0.025)"
FIRE = "#eb6834"
REGION_LINE, REGION_FILL = "#6e2a10", "rgba(217,89,38,{a})"
# routes are a status (best / fallback / last resort) -> status colours + labels
ROUTE_COLORS = ["#0ca30c", "#fab219", "#d03b3b"]
ROUTE_LABELS = ["RECOMMENDED", "fallback", "last resort"]
ROUTE_ICONS = ["🟢", "🟡", "🔴"]
# walkway danger is a magnitude -> one hue (orange), light to dark
DANGER_RAMP = ["#f4c9b3", "#ec9c78", "#d95926", "#a8421a", "#6e2a10"]

X_RANGE, Y_RANGE = (-200, 1000), (-300, 520)
MAX_CAMS = 6
MAX_ROUTES = 3
# "draw nothing" for a trace.  A gap rather than an empty list: when the map
# animates in the browser, a trace animated to [] keeps its old line on screen.
NOTHING = [None]
CLICK_STEP = 10          # metres between invisible click targets


def danger_color(d):
    if d == INF:
        return INK
    if d <= 1.001:
        return WALKWAY
    k = min(len(DANGER_RAMP) - 1, int((d - 1.0) / 7.0 * len(DANGER_RAMP)))
    return DANGER_RAMP[k]


# ---------------------------------------------------------------- state
@dataclass
class MapState:
    """How far each animation has got.  0 = not started, 1 = finished."""
    fire: bool = True
    pulse: float = 0.0                              # 0..1 around the breathing cycle
    wedges: list = field(default_factory=lambda: [1.0] * MAX_CAMS)
    region: float = 1.0
    estimate: bool = True
    danger: bool = True
    routes: list = field(default_factory=lambda: [1.0] * MAX_ROUTES)
    walkers: list | None = None                     # fraction along each route


BLANK = MapState(fire=False, wedges=[0.0] * MAX_CAMS, region=0.0, estimate=False,
                 danger=False, routes=[0.0] * MAX_ROUTES)


# ---------------------------------------------------------------- polylines
def _offset_path(pts, d):
    """Shift a polyline sideways by d metres, so parallel routes don't hide
    each other where they share a walkway."""
    if d == 0 or len(pts) < 2:
        return list(pts)
    out = []
    for i, p in enumerate(pts):
        nx = ny = 0.0
        segs = ([(pts[i - 1], p)] if i > 0 else []) + ([(p, pts[i + 1])] if i + 1 < len(pts) else [])
        for a, b in segs:                     # average the normals of both neighbours
            L = math.dist(a, b) or 1.0
            nx += -(b[1] - a[1]) / L
            ny += (b[0] - a[0]) / L
        L = math.hypot(nx, ny) or 1.0
        out.append((p[0] + d * nx / L, p[1] + d * ny / L))
    return out


def _slice(pts, frac):
    """The first `frac` of a polyline (by length), and the point where it ends."""
    if frac <= 0 or len(pts) < 2:
        return [pts[0]] if pts else [], pts[0] if pts else None
    segs = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    want = min(1.0, frac) * sum(segs)
    out = [pts[0]]
    for (a, b), L in zip(zip(pts, pts[1:]), segs):
        if want <= L:
            t = want / L if L else 0.0
            end = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            return out + [end], end
        out.append(b)
        want -= L
    return out, pts[-1]


def route_points(res, i):
    return _offset_path([res.graph.pos[v] for v in res.routes[i].path], (1 - i) * 7.0)


def _xy(pts, close=False):
    pts = list(pts) + ([pts[0]] if close and pts else [])
    return [p[0] for p in pts], [p[1] for p in pts]


# ---------------------------------------------------------------- the map
def map_traces(scenario, res=None, state=MapState(), clickable=False):
    """The campus map's traces, as plain dicts (quick to build, which matters
    because every animation frame is a whole map).  ``res`` may be None."""
    traces = []

    def add(**kw):
        traces.append(dict(type="scatter", **kw))

    g = res.graph if res else None
    fire_xy = res.fire_true if res else None

    # camera fields of view (faint), so you can see where the fire can be seen
    fx, fy = [], []
    for cam in scenario.cameras:
        x, y = _xy(cam.fov_polygon(700), close=True)
        fx += x + [None]
        fy += y + [None]
    add(x=fx, y=fy, fill="toself", fillcolor=FOV_FILL, mode="lines",
                    line=dict(width=0.5, color="rgba(42,120,214,0.25)"),
                    hoverinfo="skip", name="camera view", showlegend=False)

    # walkways -- recoloured by danger once the fire is known
    from .campus import WALKWAYS
    for k, (a, b) in enumerate(WALKWAYS):
        pa, pb = NODES[a], NODES[b]
        if g is not None and state.danger:
            w = g.walkways[k]
            col = danger_color(w.danger)
            dash = "dot" if w.danger == INF else "solid"
            tip = (f"{a}–{b}<br>{w.length:.0f} m · fire {w.to_fire:.0f} m away<br>" +
                   ("<b>CLOSED</b>" if w.danger == INF else
                    f"danger ×{w.danger:.2f} → feels like {w.cost:.0f} m"))
        else:
            col, dash = WALKWAY, "solid"
            tip = f"{a}–{b}<br>{math.dist(pa, pb):.0f} m"
        add(x=[pa[0], (pa[0] + pb[0]) / 2, pb[0]], y=[pa[1], (pa[1] + pb[1]) / 2, pb[1]],
                        mode="lines", line=dict(color=col, width=4, dash=dash),
                        hovertemplate=tip + "<extra></extra>", showlegend=False)

    # camera wedges
    first = next((i for i, c in enumerate(res.cams) if c.wedge), None) if res else None
    for i in range(MAX_CAMS):
        x = y = NOTHING
        if res and i < len(res.cams) and res.cams[i].wedge and state.wedges[i] > 0:
            c = res.cams[i]
            p = min(1.0, state.wedges[i])
            reach = math.dist(c.cam.pos, res.estimate or fire_xy) + 150
            reach *= min(1.0, 0.25 + p)
            half = (c.wedge[1] - c.wedge[0]) / 2 * p
            a0, a1 = math.radians(c.theta - half), math.radians(c.theta + half)
            x, y = _xy([c.cam.pos, (c.cam.x + reach * math.cos(a0), c.cam.y + reach * math.sin(a0)),
                        (c.cam.x + reach * math.cos(a1), c.cam.y + reach * math.sin(a1))], close=True)
        add(x=x, y=y, fill="toself", fillcolor=WEDGE_FILL, mode="lines",
                        line=dict(color=CAMERA, width=1), hoverinfo="skip",
                        name="sight wedge", legendgroup="wedge", showlegend=(i == first))

    # region where the fire must be
    x = y = NOTHING
    if res and len(res.region) >= 3 and state.region > 0:
        cx, cy = polygon_centroid(res.region)
        s = 0.3 + 0.7 * min(1.0, state.region)
        x, y = _xy([(cx + s * (px - cx), cy + s * (py - cy)) for px, py in res.region], close=True)
    add(x=x, y=y, fill="toself",
                    fillcolor=REGION_FILL.format(a=0.55 * min(1.0, state.region)),
                    mode="lines", line=dict(color=REGION_LINE, width=2),
                    name=f"fire region ({res.region_ha:.2f} ha)" if res and res.region else "fire region",
                    hovertemplate="where the fire must be<extra></extra>",
                    showlegend=bool(res and len(res.region) >= 3))

    # routes
    for i in range(MAX_ROUTES):
        x = y = NOTHING
        if res and i < len(res.routes) and state.routes[i] > 0:
            pts, _ = _slice(route_points(res, i), state.routes[i])
            x, y = _xy(pts)
        r = res.routes[i] if res and i < len(res.routes) else None
        add(x=x, y=y, mode="lines", line=dict(color=ROUTE_COLORS[i], width=5),
                        name=(f"Route {i + 1} · {ROUTE_LABELS[i]}" if r else f"Route {i + 1}"),
                        hovertemplate=(f"Route {i + 1}: {r.metres:.0f} m, feels like {r.cost:.0f} m"
                                       f"<extra></extra>" if r else None),
                        showlegend=r is not None)

    # places
    signs = {}
    if res and res.plan:
        for v, n in enumerate(g.name):
            if res.plan.cost[v] == INF:
                signs[n] = "NO SAFE ROUTE"
            elif res.plan.next_hop[v] != -1:
                signs[n] = f"sign: go to {g.name[res.plan.next_hop[v]]} ({res.plan.cost[v]:.0f} m effort)"
    start = res.settings.start if res else None
    names = list(NODES)
    add(
        x=[NODES[n][0] for n in names], y=[NODES[n][1] for n in names],
        mode="markers+text", text=names, textposition="top center",
        textfont=dict(size=11, color=INK2),
        marker=dict(size=[16 if n in ("ASSEMBLY", start) else 10 for n in names],
                    color=[ROUTE_COLORS[0] if n == "ASSEMBLY" else (INK if n == start else SURFACE)
                           for n in names],
                    symbol=["square" if n == "ASSEMBLY" else "circle" for n in names],
                    line=dict(color=INK, width=1.5)),
        customdata=[signs.get(n, "") for n in names],
        hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>", showlegend=False)

    # cameras
    add(x=[c.x for c in scenario.cameras], y=[c.y for c in scenario.cameras],
                    mode="markers+text", text=[c.name for c in scenario.cameras],
                    textposition="bottom center", textfont=dict(size=11, color=CAMERA),
                    marker=dict(symbol="triangle-up", size=14, color=CAMERA,
                                line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{text}<extra></extra>", name="camera")

    # true fire (we placed it) and the estimate (the system found it)
    x = y = NOTHING
    if fire_xy and state.fire:
        x, y = [fire_xy[0]], [fire_xy[1]]
    size = 20 + 8 * (0.5 + 0.5 * math.sin(2 * math.pi * state.pulse))
    add(x=x, y=y, mode="markers", name="fire (placed)",
                    marker=dict(symbol="star", size=size, color=FIRE, line=dict(color=INK, width=1)),
                    hovertemplate="fire placed at (%{x:.1f}, %{y:.1f})<extra></extra>",
                    showlegend=fire_xy is not None)
    x = y = NOTHING
    if res and res.estimate and state.estimate:
        x, y = [res.estimate[0]], [res.estimate[1]]
    add(x=x, y=y, mode="markers", name="fire (found)",
                    marker=dict(symbol="x-thin", size=16, color=INK, line=dict(color=INK, width=3)),
                    hovertemplate="found at (%{x:.1f}, %{y:.1f})<extra></extra>",
                    showlegend=bool(res and res.estimate))

    # walkers -- one dot per route
    for i in range(MAX_ROUTES):
        x = y = NOTHING
        if state.walkers and res and i < len(res.routes):
            _, p = _slice(route_points(res, i), state.walkers[i])
            x, y = [p[0]], [p[1]]
        add(x=x, y=y, mode="markers", hoverinfo="skip", showlegend=False,
                        marker=dict(size=13, color=ROUTE_COLORS[i], line=dict(color=INK, width=1.5)))

    # invisible click targets over the whole map (last, so it is on top)
    if clickable:
        gx, gy = [], []
        for x in range(X_RANGE[0] + 50, X_RANGE[1] - 49, CLICK_STEP):
            for y in range(Y_RANGE[0] + 50, Y_RANGE[1] - 49, CLICK_STEP):
                gx.append(x)
                gy.append(y)
        add(x=gx, y=gy, mode="markers", name="click", showlegend=False,
                        marker=dict(size=CLICK_STEP, color="rgba(0,0,0,0)", opacity=0.01),
                        hovertemplate="put the fire at (%{x}, %{y})<extra></extra>")

    return traces


def map_layout(height=560):
    return dict(
        height=height, margin=dict(l=50, r=10, t=40, b=30),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2),
        xaxis=dict(range=X_RANGE, showgrid=True, gridcolor=GRIDLINE, zeroline=False,
                   ticksuffix=" m", tickfont=dict(color=MUTED, size=10), fixedrange=True),
        yaxis=dict(range=Y_RANGE, showgrid=True, gridcolor=GRIDLINE, zeroline=False,
                   scaleanchor="x", scaleratio=1, tickfont=dict(color=MUTED, size=10),
                   fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.0, bgcolor=SURFACE,
                    font=dict(size=11)),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(color=INK)),
        dragmode=False, clickmode="event+select", uirevision="map",
    )


def map_figure(scenario, res=None, state=MapState(), clickable=False, height=560):
    """The campus map as a Plotly figure."""
    return go.Figure(data=map_traces(scenario, res, state, clickable), layout=map_layout(height))


# ---------------------------------------------------------------- sequences
def stage_sequence(res):
    """The run, as a list of (stage number, MapState).  ~40 states, played at
    ~10 per second, so the whole story takes about 4 seconds."""
    seq = []
    s = replace(BLANK, fire=True)
    seq.append((1, s))
    # 3. wedges sweep open, staggered camera by camera; then the region pops in
    used = [i for i, c in enumerate(res.cams) if c.wedge]
    steps = 6 + 2 * max(0, len(used) - 1)
    for k in range(1, steps + 1):
        w = [0.0] * MAX_CAMS
        for order, i in enumerate(used):
            w[i] = max(0.0, min(1.0, (k - 2 * order) / 6))
        s = replace(s, wedges=w, pulse=k / 10)
        seq.append((3, s))
    if res.located:
        for k in range(1, 6):
            s = replace(s, region=k / 5, pulse=s.pulse + 0.1)
            seq.append((3, s))
        s = replace(s, estimate=True)
        seq.append((3, s))
    # 4. walkways recolour; each route draws itself, staggered
    s = replace(s, danger=True)
    seq.append((4, s))
    n = len(res.routes)
    for k in range(1, 13 + 3 * max(0, n - 1)):
        s = replace(s, routes=[max(0.0, min(1.0, (k - 3 * i) / 12)) for i in range(MAX_ROUTES)],
                    pulse=s.pulse + 0.1)
        seq.append((4, s))
    return seq


def walker_frames(res, n=60, laps=3):
    """MapStates for walkers looping along each route while the fire breathes.
    Every walker walks at the same speed, so the long route takes longest."""
    if not res.routes:
        return []
    longest = max(r.metres for r in res.routes)
    out = []
    for k in range(n * laps):
        t = k / n
        walk = [((t * longest / r.metres) % 1.0) if r.metres else 0.0 for r in res.routes]
        walk += [0.0] * (MAX_ROUTES - len(walk))
        out.append(MapState(walkers=walk, pulse=t * 2))
    return out


def _frames(res, states, prefix, traces=None):
    """Plotly frames (plain dicts) for a list of MapStates.  They run in the
    browser, inside the one chart.  ``traces`` limits each frame to the traces
    that move."""
    out = []
    for k, st in enumerate(states):
        data = map_traces(res.scenario, res, st)
        idx = traces if traces is not None else list(range(len(data)))
        out.append(dict(name=f"{prefix}{k}", traces=idx, data=[data[i] for i in idx]))
    return out


def _play_args(names, ms):
    return [names, dict(frame=dict(duration=ms, redraw=False), transition=dict(duration=0),
                        fromcurrent=False, mode="immediate")]


def dynamic_traces(scenario):
    """Indices of the fire marker and the walker dots in map_traces' output."""
    n = len(map_traces(scenario))
    return [n - MAX_ROUTES - 2] + list(range(n - MAX_ROUTES, n))


def final_figure(res, clickable=True, height=560):
    """The finished map, with a button that sets the walkers off.  (The full
    story plays as its own page -- see story_html -- because Streamlit rebuilds
    a chart from scratch whenever its figure changes: animating by sending new
    figures makes the map flash, and a chart cannot start an animation itself.)"""
    fig = map_figure(res.scenario, res, MapState(), clickable=clickable, height=height)
    walk = _frames(res, walker_frames(res, laps=2), "w", traces=dynamic_traces(res.scenario))
    if walk:
        fig.frames = [go.Frame(**f) for f in walk]
        fig.update_layout(updatemenus=[dict(
            type="buttons", x=0.99, y=0.99, xanchor="right", yanchor="top", direction="left",
            bgcolor=SURFACE, bordercolor=GRIDLINE, font=dict(color=INK),
            buttons=[dict(label="▶ walk the routes", method="animate",
                          args=_play_args([f["name"] for f in walk], 60)),
                     dict(label="■", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode="immediate")])])])
    return fig


FRAME_MS = 100            # one story frame
HOLD_FRAMES = 20          # the map waits while the photos and AI boxes appear


def story_states(res):
    """The whole run as (stage, MapState) frames: the map holds still while
    the photos are taken (1) and the AI finds the fire (2), then the wedges
    sweep and cross (3) and the routes draw themselves (4)."""
    seq = stage_sequence(res)
    first = seq[0][1]
    hold = [(1, first)] * (HOLD_FRAMES // 2) + [(2, first)] * (HOLD_FRAMES - HOLD_FRAMES // 2)
    return hold + seq[1:] + [(4, seq[-1][1])] * 5


def stage_starts(states):
    """Seconds into the story at which each stage begins."""
    starts = {}
    for k, (stage, _) in enumerate(states):
        starts.setdefault(stage, k * FRAME_MS / 1000)
    return starts


def story_figure(res, height=560):
    """The whole story as one self-playing figure (a plain dict)."""
    states = [st for _, st in story_states(res)]
    return dict(data=map_traces(res.scenario, res, states[0]), layout=map_layout(height),
                frames=_frames(res, states, "s"))


def story_html(res, plotly_js, height=560):
    """A self-playing copy of the map as a small web page, written by Plotly.
    It starts by itself -- which a Streamlit chart cannot -- and ends on
    exactly the picture the clickable map shows next."""
    html = pio.to_html(story_figure(res, height), validate=False, include_plotlyjs=plotly_js,
                       full_html=True, auto_play=True,
                       animation_opts=dict(frame=dict(duration=FRAME_MS, redraw=False),
                                           transition=dict(duration=0), mode="immediate"),
                       default_width="100%", default_height=f"{height}px",
                       config={"displayModeBar": False})
    return html.replace("<body>", '<body style="margin:0">', 1)


def animated_figure(res, height=620):
    """The whole story as a self-playing figure (for the HTML export)."""
    return go.Figure(story_figure(res, height), skip_invalid=True)


# ---------------------------------------------------------------- search reveal
def search_figure(res, height=380):
    """Animation 7: A* and Dijkstra side by side, lighting up each place as
    it is examined.  A* is pulled toward the goal by its hint; Dijkstra spreads
    out in every direction."""
    from plotly.subplots import make_subplots
    g = res.graph
    runs = [("A*", res.routes[0] if res.routes else None), ("Dijkstra", res.dijkstra)]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.04,
                        subplot_titles=[f"{n}: examined {r.expanded if r else 0} of {len(g.name)} places"
                                        for n, r in runs])
    names = g.name
    xs, ys = [p[0] for p in g.pos], [p[1] for p in g.pos]

    def colours(r, k):
        seen = set(r.visit_order[:k]) if r else set()
        return [CAMERA if v in seen else SURFACE for v in range(len(names))]

    for col in (1, 2):
        for w in g.walkways:
            (xa, ya), (xb, yb) = g.pos[w.u], g.pos[w.v]
            fig.add_scatter(x=[xa, xb], y=[ya, yb], mode="lines", row=1, col=col,
                            line=dict(color=danger_color(w.danger), width=2,
                                      dash="dot" if w.danger == INF else "solid"),
                            hoverinfo="skip", showlegend=False)
    dyn = []
    for col, (_, r) in enumerate(runs, start=1):
        fig.add_scatter(x=xs, y=ys, mode="markers+text", text=names, textposition="top center",
                        textfont=dict(size=9, color=INK2), row=1, col=col,
                        marker=dict(size=12, color=colours(r, 0), line=dict(color=INK, width=1)),
                        hoverinfo="skip", showlegend=False)
        dyn.append(len(fig.data) - 1)
    steps = max(len(r.visit_order) if r else 0 for _, r in runs)
    fig.frames = [go.Frame(name=str(k), traces=dyn,
                           data=[go.Scatter(marker=dict(size=12, color=colours(r, k),
                                                        line=dict(color=INK, width=1)))
                                 for _, r in runs])
                  for k in range(steps + 1)]
    for ax in ("xaxis", "xaxis2"):
        fig.layout[ax].update(visible=False, range=(40, 890), fixedrange=True)
    for ax, anchor in (("yaxis", "x"), ("yaxis2", "x2")):
        fig.layout[ax].update(visible=False, range=(-180, 390), scaleanchor=anchor, fixedrange=True)
    fig.update_layout(
        height=height, margin=dict(l=5, r=5, t=40, b=5), plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2),
        updatemenus=[dict(type="buttons", x=0.5, y=-0.02, xanchor="center", yanchor="top",
                          bgcolor=SURFACE, bordercolor=GRIDLINE, font=dict(color=INK),
                          buttons=[dict(label="▶ watch them search", method="animate",
                                        args=[None, dict(frame=dict(duration=450, redraw=False),
                                                         transition=dict(duration=0),
                                                         fromcurrent=False)])])])
    return fig


# ---------------------------------------------------------------- photos
def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate_photo(cr, conf, show_boxes=True):
    """The camera photo with the AI's boxes drawn on it (fire boxes used for the
    wedge in orange, everything else thin grey)."""
    img = cr.image.copy()
    if not show_boxes:
        return img
    d = ImageDraw.Draw(img)
    f = _font(max(18, img.width // 60))
    used = set(cr.fire_boxes)
    for b in cr.boxes:
        if b.conf < conf:
            continue
        on = b in used
        col = FIRE if on else "#c3c2b7"
        d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=col, width=6 if on else 3)
        tag = f"{b.label} {b.conf:.2f}"
        tw = d.textlength(tag, font=f)
        ty = max(0, b.y1 - f.size - 10)
        d.rectangle([b.x1, ty, b.x1 + tw + 12, ty + f.size + 8], fill=col)
        d.text((b.x1 + 6, ty + 3), tag, fill=INK, font=f)
    if cr.cols:                                     # the two columns that become bearings
        for u in cr.cols:
            d.line([u, 0, u, img.height], fill=FIRE, width=2)
    return img


# ---------------------------------------------------------------- headless
def save_headless(res, out_dir):
    """Photos with boxes (PNG), a static map (PNG, matplotlib) and the full
    animation (self-contained HTML).  Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in res.cams:
        p = out / f"photo_{c.cam.name.replace(' ', '').lower()}.png"
        annotate_photo(c, res.settings.conf).save(p)
        paths.append(p)
    p = out / "map.png"
    _map_png(res, p)
    paths.append(p)
    p = out / "animation.html"
    animated_figure(res).write_html(p, include_plotlyjs=True, auto_play=True)
    paths.append(p)
    return paths


def _map_png(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res.graph
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=130)
    ax.set_facecolor(SURFACE)
    for cam in res.scenario.cameras:
        x, y = _xy(cam.fov_polygon(700), close=True)
        ax.fill(x, y, color=CAMERA, alpha=0.025, lw=0)
    for w in g.walkways:
        (xa, ya), (xb, yb) = g.pos[w.u], g.pos[w.v]
        ax.plot([xa, xb], [ya, yb], color=danger_color(w.danger), lw=3,
                ls=":" if w.danger == INF else "-", zorder=1)
    for c in res.cams:
        if c.wedge:
            reach = math.dist(c.cam.pos, res.estimate or res.fire_true) + 150
            pts = [c.cam.pos] + [(c.cam.x + reach * math.cos(math.radians(a)),
                                  c.cam.y + reach * math.sin(math.radians(a))) for a in c.wedge]
            ax.fill(*_xy(pts, True), color=CAMERA, alpha=0.16, lw=0.8, ec=CAMERA)
    if len(res.region) >= 3:
        ax.fill(*_xy(res.region, True), color="#d95926", alpha=0.55, ec=REGION_LINE, lw=1.5,
                label=f"fire region ({res.region_ha:.2f} ha)")
    for i, r in enumerate(res.routes):
        ax.plot(*_xy(route_points(res, i)), color=ROUTE_COLORS[i], lw=4, zorder=3,
                label=f"Route {i + 1} · {ROUTE_LABELS[i]} · {r.metres:.0f} m, x{r.worst_danger:.1f}")
    for n, (x, y) in NODES.items():
        ax.plot(x, y, "o", ms=6, mfc=SURFACE, mec=INK, zorder=4)
        ax.annotate(n, (x, y), xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    ax.plot([c.x for c in res.scenario.cameras], [c.y for c in res.scenario.cameras], "^",
            color=CAMERA, ms=10, label="camera", zorder=5)
    ax.plot(*res.fire_true, "*", color=FIRE, ms=18, mec=INK, label="fire (placed)", zorder=6)
    if res.estimate:
        ax.plot(*res.estimate, "x", color=INK, ms=11, mew=2.5,
                label=f"fire (found) — error {res.error_m:.1f} m", zorder=7)
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_aspect("equal")
    ax.grid(color=GRIDLINE, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(WALKWAY)
    ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRIDLINE)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
