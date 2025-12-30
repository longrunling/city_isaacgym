"""Geometry helpers for simple_roadgen

Provides functions to compute distance from a 2D point to a line segment
and to find the nearest edge (road segment) in a graph of `Vertex` nodes.

Typical usage:
    from simple_roadgen.geometry import nearest_edge
    dist, (a,b), closest_point, t = nearest_edge((x,y), global_object)

Return values:
    dist: float distance
    (a,b): tuple of `Vertex` endpoints of the nearest segment
    closest_point: numpy array of coordinates of the projection
    t: parameter along segment (0..1) corresponding to the closest point
"""
import numpy as np
from typing import Tuple, Any


def point_segment_distance(point, a, b) -> Tuple[float, float, np.ndarray]:
    """Compute distance from `point` to segment [a,b].

    Parameters
    ----------
    point : array-like (2,)
    a : array-like (2,) segment start
    b : array-like (2,) segment end

    Returns
    -------
    dist : float
        Euclidean distance from point to the segment.
    t : float
        Unclamped projection parameter along the segment (then clamped to [0,1]).
    closest : np.ndarray
        The closest point on the segment to `point`.
    """
    p = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0.0:
        # a and b are the same point
        closest = a.copy()
        t = 0.0
        dist = np.linalg.norm(p - closest)
        return dist, t, closest

    t_unclamped = np.dot(p - a, ab) / ab2
    t = max(0.0, min(1.0, t_unclamped))
    closest = a + t * ab
    dist = np.linalg.norm(p - closest)
    return dist, t, closest


def nearest_edge(point, g) -> Tuple[float, Tuple[Any, Any], np.ndarray, float]:
    """Find nearest edge in graph `g` to `point`.

    Parameters
    ----------
    point : array-like (2,)
        Query point.
    g : SimpleGlobal-like object
        Object that exposes `vertex_list`, each vertex has `.coords` and
        `.neighbours` (list of vertices).

    Returns
    -------
    best_dist : float
    (a,b) : tuple
        The endpoints (`Vertex` objects) of the nearest segment.
    closest_point : np.ndarray
    t : float
        Parameter along the segment (0..1) of the closest point.

    If graph has no edges, returns (inf, (None,None), None, None).
    """
    best = (np.inf, (None, None), None, None)
    seen = set()
    for a in g.vertex_list:
        for b in a.neighbours:
            # avoid checking the same undirected edge twice
            eid = tuple(sorted((id(a), id(b))))
            if eid in seen:
                continue
            seen.add(eid)
            dist, t, closest = point_segment_distance(point, a.coords, b.coords)
            if dist < best[0]:
                best = (dist, (a, b), closest, t)
    return best
