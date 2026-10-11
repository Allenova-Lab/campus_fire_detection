"""
Stage 3 -- from fire columns in each photo to a fire position on the map.

Copied unchanged from docs/camera_fire_localization/fire_localization_2d.py
(the reference code for "Camera-Based Fire Localization in 2D").  Pure Python
3 standard library -- no dependencies.  Section and equation numbers in the
docstrings refer to docs/camera_fire_localization/fire_localization_2d.pdf.

Run directly to reproduce every number in the guide's worked examples:

    python -m firepath.localize
"""

import math
from itertools import combinations

# ---------------------------------------------------------------------------
# Part I -- photo pixel to world ray
# ---------------------------------------------------------------------------


def focal_from_fov(width_px, hfov_deg):
    """Eq. (3): focal length in pixels from image width and horizontal FOV."""
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def undistort_x(x_dist, y_dist, k1=0.0, k2=0.0, p1=0.0, p2=0.0, iters=8):
    """Invert the Brown-Conrady model, Eq. (4), by fixed-point iteration.

    Takes and returns *normalized* coordinates ((u - cu) / f).
    """
    x, y = x_dist, y_dist
    for _ in range(iters):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (x_dist - dx) / radial
        y = (y_dist - dy) / radial
    return x, y


def pixel_to_bearing(u, psi_deg, f, cu, k=None, v=None, cv=None, pitch_deg=0.0):
    """Eq. (8): image column -> world bearing in degrees (CCW from +x).

    Pass ``k=(k1, k2, p1, p2)`` to undistort first, and ``v``/``cv``/
    ``pitch_deg`` to use the exact pitched-camera form, Eq. (14).
    """
    xn = (u - cu) / f
    yn = 0.0 if (v is None or cv is None) else (v - cv) / f
    if k:
        xn, yn = undistort_x(xn, yn, *k)
    if pitch_deg == 0.0:
        alpha = math.atan(xn)
    else:
        ph = math.radians(pitch_deg)
        alpha = math.atan2(xn, yn * math.sin(ph) + math.cos(ph))
    return psi_deg - math.degrees(alpha)


def bearing_sigma(u, cu, f, sigma_u):
    """Eq. (12): bearing standard deviation (radians) from pixel noise."""
    return f / (f * f + (u - cu) ** 2) * sigma_u


def direction(theta_deg):
    """Eq. (1): unit vector along a bearing."""
    t = math.radians(theta_deg)
    return (math.cos(t), math.sin(t))


def normal(theta_deg):
    """Eq. (2): unit normal to a bearing."""
    t = math.radians(theta_deg)
    return (-math.sin(t), math.cos(t))


def ground_range_flat(v_base, cv, fv, h, pitch_deg):
    """Eq. (13): single-camera range under the flat-ground assumption."""
    eps = math.radians(pitch_deg) + math.atan((v_base - cv) / fv)
    if eps <= 0:
        raise ValueError("ray points at or above the horizon; no ground intersection")
    return h / math.tan(eps)


# ---------------------------------------------------------------------------
# Part II -- two-camera intersection
# ---------------------------------------------------------------------------


def intersect(C1, th1, C2, th2):
    """Eqs. (16)-(18): closed-form ray-ray intersection.

    Returns ``(P, r1, r2)``. Raises if the rays are (near) parallel.
    ``r1 > 0 and r2 > 0`` must be checked by the caller: a negative range
    means the crossing lies behind that camera.
    """
    gamma = math.radians(th2 - th1)
    s = math.sin(gamma)
    if abs(s) < 1e-9:
        raise ValueError("rays are parallel; intersection undefined")
    dx, dy = C2[0] - C1[0], C2[1] - C1[1]
    r1 = (dx * math.sin(math.radians(th2)) - dy * math.cos(math.radians(th2))) / s
    r2 = (dx * math.sin(math.radians(th1)) - dy * math.cos(math.radians(th1))) / s
    d1 = direction(th1)
    return (C1[0] + r1 * d1[0], C1[1] + r1 * d1[1]), r1, r2


def intersect_cov(th1, r1, s1, th2, r2, s2):
    """Eq. (23): 2x2 covariance of the two-camera point estimate.

    ``s1``, ``s2`` are bearing standard deviations in radians.
    """
    n1, n2 = normal(th1), normal(th2)
    det = n1[0] * n2[1] - n1[1] * n2[0]
    Ai = ((n2[1] / det, -n1[1] / det), (-n2[0] / det, n1[0] / det))
    v1, v2 = (r1 * s1) ** 2, (r2 * s2) ** 2
    sxx = Ai[0][0] ** 2 * v1 + Ai[0][1] ** 2 * v2
    syy = Ai[1][0] ** 2 * v1 + Ai[1][1] ** 2 * v2
    sxy = Ai[0][0] * Ai[1][0] * v1 + Ai[0][1] * Ai[1][1] * v2
    return ((sxx, sxy), (sxy, syy))


def ellipse(cov, conf=0.95):
    """Confidence-ellipse semi-axes (m) and orientation (deg) from a 2x2 cov."""
    (a, b), (_, c) = cov
    tr, det = a + c, a * c - b * b
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    scale = math.sqrt(-2.0 * math.log(1.0 - conf))  # chi2_2 quantile
    ang = math.degrees(math.atan2(l1 - a, b)) if abs(b) > 1e-15 else 0.0
    return scale * math.sqrt(l1), scale * math.sqrt(max(l2, 0.0)), ang


def wedge_quad(C1, th1_min, th1_max, C2, th2_min, th2_max):
    """Eq. (24): the four corners of wedge-1 cap wedge-2, ordered CCW."""
    pts = [
        intersect(C1, a, C2, b)[0]
        for a in (th1_min, th1_max)
        for b in (th2_min, th2_max)
    ]
    gx = sum(p[0] for p in pts) / len(pts)
    gy = sum(p[1] for p in pts) / len(pts)
    return sorted(pts, key=lambda p: math.atan2(p[1] - gy, p[0] - gx))


# ---------------------------------------------------------------------------
# Polygon helpers -- Eqs. (25), (26)
# ---------------------------------------------------------------------------


def polygon_area(poly):
    """Eq. (25): shoelace area."""
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
                   for i in range(n))) / 2.0


def polygon_centroid(poly):
    """Eq. (26): area centroid."""
    n = len(poly)
    cross = [poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
             for i in range(n)]
    a2 = sum(cross)
    gx = sum((poly[i][0] + poly[(i + 1) % n][0]) * cross[i] for i in range(n)) / (3 * a2)
    gy = sum((poly[i][1] + poly[(i + 1) % n][1]) * cross[i] for i in range(n)) / (3 * a2)
    return gx, gy


def clip_halfplane(poly, n, C, keep_positive):
    """Eq. (31): Sutherland-Hodgman clip of a convex polygon by one half-plane.

    Keeps the side where ``n . (P - C) >= 0`` if ``keep_positive`` else ``<= 0``.
    """
    out, m = [], len(poly)
    if m == 0:
        return out
    for i in range(m):
        A, B = poly[i], poly[(i + 1) % m]
        ga = n[0] * (A[0] - C[0]) + n[1] * (A[1] - C[1])
        gb = n[0] * (B[0] - C[0]) + n[1] * (B[1] - C[1])
        if not keep_positive:
            ga, gb = -ga, -gb
        if ga >= 0:
            out.append(A)
        if (ga >= 0) != (gb >= 0):
            t = ga / (ga - gb)
            out.append((A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])))
    return out


# ---------------------------------------------------------------------------
# Part III -- N cameras
# ---------------------------------------------------------------------------


def _solve_wls(rows, bs, w):
    """Eqs. (28)-(30): weighted least squares via the 2x2 normal equations."""
    a11 = sum(w[i] * rows[i][0] ** 2 for i in range(len(w)))
    a12 = sum(w[i] * rows[i][0] * rows[i][1] for i in range(len(w)))
    a22 = sum(w[i] * rows[i][1] ** 2 for i in range(len(w)))
    b1 = sum(w[i] * rows[i][0] * bs[i] for i in range(len(w)))
    b2 = sum(w[i] * rows[i][1] * bs[i] for i in range(len(w)))
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        raise ValueError("normal matrix singular: cameras collinear with target")
    P = ((a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det)
    cov = ((a22 / det, -a12 / det), (-a12 / det, a11 / det))
    return P, cov, (a11, a12, a22)


def localize(cams, iters=3, huber=1.5):
    """Algorithm 1: IRLS multi-camera localization.

    ``cams`` is a list of ``(C, theta_deg, sigma_theta_rad)``.
    Returns a dict with the point, covariance, chi-square, dof and weights.
    """
    rows = [normal(th) for _, th, _ in cams]
    bs = [n[0] * C[0] + n[1] * C[1] for n, (C, _, _) in zip(rows, cams)]
    sig = [s for _, _, s in cams]

    P, cov, _ = _solve_wls(rows, bs, [1.0] * len(cams))  # unweighted start
    w = [1.0] * len(cams)
    for _ in range(iters):
        w = []
        for i, (C, _, _) in enumerate(cams):
            r = math.dist(P, C)
            sr = r * sig[i]
            wi = 1.0 / (sr * sr)
            rho = rows[i][0] * P[0] + rows[i][1] * P[1] - bs[i]
            if huber and abs(rho) > huber * sr:          # Huber down-weight
                wi *= huber * sr / abs(rho)
            w.append(wi)
        P, cov, _ = _solve_wls(rows, bs, w)

    resid = [rows[i][0] * P[0] + rows[i][1] * P[1] - bs[i] for i in range(len(cams))]
    chi2 = sum(w[i] * resid[i] ** 2 for i in range(len(cams)))
    return {"P": P, "cov": cov, "residuals": resid, "weights": w,
            "chi2": chi2, "dof": len(cams) - 2,
            "ranges": [math.dist(P, C) for C, _, _ in cams]}


def region(wedges, seed=None):
    """Algorithm 2: convex region from N wedges by successive clipping.

    ``wedges`` is a list of ``(C, theta_min_deg, theta_max_deg)``.
    """
    if seed is None:
        (C1, a1, b1), (C2, a2, b2) = wedges[0], wedges[1]
        seed = wedge_quad(C1, a1, b1, C2, a2, b2)
    Q = seed
    for C, tmin, tmax in wedges:
        Q = clip_halfplane(Q, normal(tmin), C, True)
        Q = clip_halfplane(Q, normal(tmax), C, False)
        if not Q:
            return []
    return Q


# ---------------------------------------------------------------------------
# Worked example from the document
# ---------------------------------------------------------------------------

def _demo():
    W, HFOV, CU, SIGMA_U = 1920, 60.0, 960.0, 1.5
    f = focal_from_fov(W, HFOV)

    site = {
        1: {"C": (0.0, 0.0),      "psi": 20.0,  "u": (1100.0, 1400.0)},
        2: {"C": (300.0, -400.0), "psi": 70.0,  "u": (700.0, 900.0)},
        3: {"C": (900.0, 500.0),  "psi": 220.0, "u": (880.0, 950.0)},
    }

    print("=" * 72)
    print("PART I  -- pixel to bearing")
    print("=" * 72)
    print(f"focal length  f = {f:.4f} px\n")
    for i, c in site.items():
        th = sorted(pixel_to_bearing(u, c["psi"], f, CU) for u in c["u"])
        c["wedge"] = th
        c["theta"] = (th[0] + th[1]) / 2
        uc = sum(c["u"]) / 2
        c["sigma"] = bearing_sigma(uc, CU, f, SIGMA_U)
        for u in c["u"]:
            print(f"  cam{i}  u={u:7.1f}  xn={(u - CU) / f:+.6f}  "
                  f"alpha={math.degrees(math.atan((u - CU) / f)):+8.4f} deg  "
                  f"theta={pixel_to_bearing(u, c['psi'], f, CU):9.4f} deg")
        print(f"  cam{i}  wedge=[{th[0]:.4f}, {th[1]:.4f}] deg  "
              f"width={th[1] - th[0]:.4f} deg  centre={c['theta']:.4f} deg")
        print(f"  cam{i}  sigma_theta={c['sigma']:.6e} rad "
              f"= {math.degrees(c['sigma']):.5f} deg\n")

    print("=" * 72)
    print("PART II -- cameras 1 and 2")
    print("=" * 72)
    c1, c2 = site[1], site[2]
    P, r1, r2 = intersect(c1["C"], c1["theta"], c2["C"], c2["theta"])
    gamma = c2["theta"] - c1["theta"]
    print(f"  gamma = {gamma:.4f} deg,  sin(gamma) = {math.sin(math.radians(gamma)):.6f}")
    print(f"  r1 = {r1:.2f} m,  r2 = {r2:.2f} m   (both > 0: fire is in front)")
    print(f"  P  = ({P[0]:.2f}, {P[1]:.2f}) m")
    back = intersect(c2["C"], c2["theta"], c1["C"], c1["theta"])[0]
    print(f"  cross-check from cam2: ({back[0]:.2f}, {back[1]:.2f}) m\n")

    quad = wedge_quad(c1["C"], c1["wedge"][0], c1["wedge"][1],
                      c2["C"], c2["wedge"][0], c2["wedge"][1])
    print("  quadrilateral corners (CCW):")
    for p in quad:
        print(f"    ({p[0]:8.2f}, {p[1]:8.2f})")
    aq = polygon_area(quad)
    gq = polygon_centroid(quad)
    print(f"  area = {aq:.1f} m^2 = {aq / 1e4:.4f} ha")
    print(f"  centroid = ({gq[0]:.2f}, {gq[1]:.2f}) m\n")

    cov2 = intersect_cov(c1["theta"], r1, c1["sigma"], c2["theta"], r2, c2["sigma"])
    print(f"  sigma_perp: r1*s1 = {r1 * c1['sigma']:.3f} m, "
          f"r2*s2 = {r2 * c2['sigma']:.3f} m")
    print(f"  Sigma = [[{cov2[0][0]:.5f}, {cov2[0][1]:.5f}], "
          f"[{cov2[1][0]:.5f}, {cov2[1][1]:.5f}]] m^2")
    print(f"  sigma_x = {math.sqrt(cov2[0][0]):.3f} m, "
          f"sigma_y = {math.sqrt(cov2[1][1]):.3f} m")
    a, b, ang = ellipse(cov2, 0.3935)   # 1-sigma for 2 dof
    print(f"  1-sigma semi-axes: {a:.3f} m, {b:.3f} m at {ang:.2f} deg\n")

    print("=" * 72)
    print("PART III -- all three cameras")
    print("=" * 72)
    cams = [(site[i]["C"], site[i]["theta"], site[i]["sigma"]) for i in (1, 2, 3)]
    for i, (C, th, _) in zip((1, 2, 3), cams):
        n = normal(th)
        print(f"  cam{i}  n=({n[0]:+.6f}, {n[1]:+.6f})  "
              f"b={n[0] * C[0] + n[1] * C[1]:+.4f}")
    out = localize(cams)
    print(f"\n  ranges  = {[f'{r:.2f}' for r in out['ranges']]}")
    print(f"  weights = {[f'{w:.3f}' for w in out['weights']]} m^-2")
    print(f"  P_hat   = ({out['P'][0]:.2f}, {out['P'][1]:.2f}) m")
    print(f"  residuals = {[f'{r:+.3f}' for r in out['residuals']]} m")
    cv = out["cov"]
    print(f"  Sigma = [[{cv[0][0]:.5f}, {cv[0][1]:.5f}], "
          f"[{cv[1][0]:.5f}, {cv[1][1]:.5f}]] m^2")
    print(f"  sigma_x = {math.sqrt(cv[0][0]):.3f} m, "
          f"sigma_y = {math.sqrt(cv[1][1]):.3f} m")
    a, b, ang = ellipse(cv, 0.3935)
    print(f"  1-sigma semi-axes: {a:.3f} m, {b:.3f} m at {ang:.2f} deg")
    crit = {1: 3.84, 2: 5.99, 3: 7.81, 4: 9.49}
    print(f"  chi2 = {out['chi2']:.2f} on {out['dof']} dof "
          f"(95% critical {crit.get(out['dof'], '?')}) -> "
          f"{'PASS' if out['chi2'] < crit.get(out['dof'], 1e9) else 'FAIL'}\n")

    print("  pairwise cross-check:")
    for i, j in combinations((1, 2, 3), 2):
        Pp, ra, rb = intersect(site[i]["C"], site[i]["theta"],
                               site[j]["C"], site[j]["theta"])
        g = site[j]["theta"] - site[i]["theta"]
        print(f"    ({i},{j})  |sin gamma| = {abs(math.sin(math.radians(g))):.3f}   "
              f"P = ({Pp[0]:8.2f}, {Pp[1]:8.2f})")

    wedges = [(site[i]["C"], site[i]["wedge"][0], site[i]["wedge"][1])
              for i in (1, 2, 3)]
    R = region(wedges)
    print(f"\n  region has {len(R)} vertices:")
    for p in R:
        print(f"    ({p[0]:8.2f}, {p[1]:8.2f})")
    ar = polygon_area(R)
    gr = polygon_centroid(R)
    print(f"  area = {ar:.1f} m^2 = {ar / 1e4:.4f} ha "
          f"({100 * (1 - ar / aq):.1f}% smaller than the 2-camera quad)")
    print(f"  centroid = ({gr[0]:.2f}, {gr[1]:.2f}) m")

    inside = all(
        clip_halfplane([out["P"]], normal(w[1]), w[0], True) and
        clip_halfplane([out["P"]], normal(w[2]), w[0], False)
        for w in wedges
    )
    print(f"  P_hat inside the region: {inside}")


if __name__ == "__main__":
    _demo()
