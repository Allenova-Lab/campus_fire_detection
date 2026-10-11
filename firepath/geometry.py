"""
2-D distance helpers for "how close does this walkway get to the fire?".

Ported from docs/campus_evacuation/campus_evacuation.cpp.  Points are plain
(x, y) tuples in metres.  No dependencies.
"""

import math


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_to_segment(p, a, b):
    """Shortest distance from point p to the straight segment ab.

    t says how far along the segment the closest spot is (0 at a, 1 at b).
    If the closest spot on the infinite line is off the segment, clamping t
    back into [0, 1] lands us on whichever endpoint is nearer.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:                       # a and b are the same point
        return dist(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2
    t = max(0.0, min(1.0, t))              # clamp onto the segment
    return dist(p, (a[0] + t * dx, a[1] + t * dy))


def _side(p, q, r):
    v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return 1 if v > 1e-9 else (-1 if v < -1e-9 else 0)


def segments_cross(a, b, c, d):
    """Do segments ab and cd cross each other?"""
    return (_side(a, b, c) * _side(a, b, d) < 0 and
            _side(c, d, a) * _side(c, d, b) < 0)


def segment_to_segment(a, b, c, d):
    """Shortest distance between two segments (0 if they cross)."""
    if segments_cross(a, b, c, d):
        return 0.0
    return min(point_to_segment(a, c, d), point_to_segment(b, c, d),
               point_to_segment(c, a, b), point_to_segment(d, a, b))


def point_in_polygon(p, poly):
    """Ray casting: count boundary crossings on a ray heading east from p.
    An odd count means inside."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        if (poly[i][1] > p[1]) != (poly[j][1] > p[1]):
            x_cross = poly[i][0] + (p[1] - poly[i][1]) * \
                (poly[j][0] - poly[i][0]) / (poly[j][1] - poly[i][1])
            if p[0] < x_cross:
                inside = not inside
        j = i
    return inside


def distance_to_fire(a, b, fire):
    """How close does the walkway a--b get to the fire polygon?  0 if it touches."""
    if not fire:
        return math.inf
    if point_in_polygon(a, fire) or point_in_polygon(b, fire):
        return 0.0
    best = math.inf
    n = len(fire)
    for i in range(n):
        best = min(best, segment_to_segment(a, b, fire[i], fire[(i + 1) % n]))
        if best == 0.0:
            break
    return best
