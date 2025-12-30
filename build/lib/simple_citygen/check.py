import numpy as np
from scipy.spatial import cKDTree
try:
    from .vertex import Vertex
except Exception:
    from vertex import Vertex

class SimpleGlobal:
    def __init__(self):
        self.vertex_list = []
        self.coord_list = []
        self.tree = None


def rebuild_tree(g):
    if g.coord_list:
        g.tree = cKDTree(g.coord_list)
    else:
        g.tree = None


def get_intersection(a, ab, c, cd):
    """Return (t, u) where a + t*ab intersects c + u*cd. If parallel, return (inf, inf)."""
    try:
        sol = np.linalg.solve(np.array([ab, -cd]).T, c - a)
        return sol[0], sol[1]
    except np.linalg.LinAlgError:
        return np.inf, np.inf


def check(suggested, neighbour, g, params):
    """
    Simplified check:
    - Bounds
    - If nearest existing vertex closer than min_distance: connect to it instead
    - Else check for intersections with nearby edges and cut/insert intersection
    - Else add suggested vertex
    Returns: (added_vertex or None)
    """
    border = params.get("border", (15, 15))
    maxLength = params.get("max_segment_length", 1.6)
    min_distance = params.get("min_node_distance", 0.6)

    x, y = suggested.coords
    if abs(x) > border[0] - maxLength or abs(y) > border[1] - maxLength:
        return None

    # find nearby vertices
    if g.tree is None:
        dists = []
        near = []
    else:
        dists, idxs = g.tree.query(suggested.coords, k=8, distance_upper_bound=maxLength)
        # normalize outputs
        if np.isscalar(idxs):
            idxs = [idxs]
            dists = [dists]
        near = [g.vertex_list[i] for i in idxs if i < len(g.vertex_list)]

    if len(dists) > 0 and dists[0] < min_distance:
        nearest = near[0]
        if nearest not in neighbour.neighbours:
            # connect neighbour to nearest
            neighbour.connection(nearest)
            return None

    # check intersections with nearby edges
    best_t = np.inf
    hit_pair = None
    for k in near:
        for n in k.neighbours:
            if n in near:
                t, u = get_intersection(neighbour.coords, suggested.coords - neighbour.coords, k.coords, n.coords - k.coords)
                if 1e-6 < t < 1.5 and 1e-6 < u < 1.0 and t < best_t:
                    best_t = t
                    hit_pair = (k, n)
    if hit_pair is not None:
        # insert intersection vertex
        newcoords = neighbour.coords + best_t * (suggested.coords - neighbour.coords)
        newv = Vertex(newcoords)
        # remove old connection between hit_pair
        a, b = hit_pair
        if b in a.neighbours:
            a.neighbours.remove(b)
        if a in b.neighbours:
            b.neighbours.remove(a)
        # connect intersection
        newv.connection(neighbour)
        a.connection(newv)
        b.connection(newv)
        g.vertex_list.append(newv)
        g.coord_list.append(newv.coords)
        rebuild_tree(g)
        return newv

    # ok to add
    suggested.connection(neighbour)
    g.vertex_list.append(suggested)
    g.coord_list.append(suggested.coords)
    rebuild_tree(g)
    return suggested
