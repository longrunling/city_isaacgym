import numpy as np
import random
try:
    from .vertex import Vertex
except Exception:
    from vertex import Vertex


def organic(vertex, b, params):
    """
    Simplified organic growth rule.
    - Uses previous segment direction to propose forward/left/right candidates.
    - Returns list of Vertex objects (candidates).
    """
    # canonical parameter names (no legacy aliases)
    pForward = params.get("organic_prob_forward", 100)
    pTurn = params.get("organic_prob_turn", 7)
    lMin = params.get("organic_length_min", 0.8)
    lMax = params.get("organic_length_max", 1.6)

    suggested = []

    if not vertex.neighbours:
        return suggested

    prev = np.array(vertex.coords - vertex.neighbours[-1].coords)
    norm = np.linalg.norm(prev)
    if norm == 0:
        return suggested
    prev = prev / norm

    # forward
    v = random.uniform(lMin, lMax) * prev
    if random.randint(0, 100) <= pForward:
        suggested.append(Vertex(vertex.coords + v))

    # right (rotate by random angle between -120 and -60)
    angle = random.uniform(-120, -60)
    theta = np.deg2rad(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    v = random.uniform(lMin, lMax) * (rot @ prev)
    if random.randint(0, 100) <= b * pTurn:
        suggested.append(Vertex(vertex.coords + v))

    # left (rotate by random angle between 60 and 120)
    angle = random.uniform(60, 120)
    theta = np.deg2rad(angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    v = random.uniform(lMin, lMax) * (rot @ prev)
    if random.randint(0, 100) <= b * pTurn:
        suggested.append(Vertex(vertex.coords + v))

    return suggested
