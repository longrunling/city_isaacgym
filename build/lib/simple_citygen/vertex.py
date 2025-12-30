import numpy as np


class Vertex:
    def __init__(self, coords):
        self.coords = np.array(coords, dtype=float)
        self.neighbours = []
        self.minor_road = False
        self.seed = False

    def connection(self, other):
        if other not in self.neighbours:
            self.neighbours.append(other)
        if self not in other.neighbours:
            other.neighbours.append(self)

    def __repr__(self):
        return f"Vertex({self.coords.tolist()})"