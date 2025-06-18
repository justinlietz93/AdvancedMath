import numpy as np
from collections import defaultdict

class SpatialHashGrid:
    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def _hash(self, point: np.ndarray) -> tuple:
        return tuple(np.floor(point / self.cell_size).astype(int))

    def insert(self, point: np.ndarray, obj: any):
        self.grid[self._hash(point)].append(obj)

    def query(self, point: np.ndarray, radius: float) -> list:
        min_cell = self._hash(point - radius)
        max_cell = self._hash(point + radius)
        
        results = []
        for i in range(min_cell[0], max_cell[0] + 1):
            for j in range(min_cell[1], max_cell[1] + 1):
                cell_content = self.grid.get((i, j), [])
                results.extend(cell_content)
                
        return results

    def get_collisions(self, point: np.ndarray) -> list:
        return self.grid.get(self._hash(point), [])
