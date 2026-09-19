"""Robot-centre free space shared by reset sampling and asset acceptance.

The grid backend's physical obstacles are CLOSED occupied-cell rectangles, not
their centre points. The map boundary is also an obstacle. No map is exposed to
the learned policy by these utilities.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .maps import GridMap


class ClearanceSpace:
    def __init__(self, grid: GridMap, clearance: float = .15):
        self.grid = grid
        self.clearance = float(clearance)
        rows, columns = np.nonzero(grid.occupied)
        self.lower = np.column_stack((columns, rows)) * grid.cell_size
        self.upper = self.lower + grid.cell_size
        self.valid = np.zeros_like(grid.occupied, dtype=bool)
        centres = grid.free_centers()
        safe = centres[self.clearances(centres) >= clearance - 1e-7]
        for x, y in safe:
            self.valid[int(y / grid.cell_size), int(x / grid.cell_size)] = True
        self.centres = safe
        self.labels = np.full(grid.occupied.shape, -1, dtype=np.int32)
        label = 0
        for row, col in zip(*np.nonzero(self.valid)):
            if self.labels[row, col] >= 0:
                continue
            queue = deque([(int(row), int(col))])
            self.labels[row, col] = label
            while queue:
                for other in self.neighbours(queue.popleft()):
                    if self.labels[other] < 0:
                        self.labels[other] = label
                        queue.append(other)
            label += 1

    def clearances(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        result = []
        for offset in range(0, len(points), 128):
            batch = points[offset:offset + 128]
            boundary = np.minimum.reduce((batch[:, 0], batch[:, 1], self.grid.width - batch[:, 0], self.grid.height - batch[:, 1]))
            if len(self.lower):
                delta = np.maximum(np.maximum(self.lower[None] - batch[:, None], batch[:, None] - self.upper[None]), 0)
                boundary = np.minimum(boundary, np.sqrt((delta ** 2).sum(-1)).min(-1))
            result.extend(boundary)
        return np.asarray(result)

    def is_valid(self, point) -> bool:
        point = np.asarray(point)
        return point.shape == (2,) and bool(np.isfinite(point).all()) and bool(self.clearances([point])[0] >= self.clearance - 1e-7)

    def cell(self, point) -> tuple[int, int]:
        x, y = point
        return int(y / self.grid.cell_size), int(x / self.grid.cell_size)

    def neighbours(self, cell):
        row, col = cell
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < self.valid.shape[0] and 0 <= c < self.valid.shape[1] and self.valid[r, c]:
                yield r, c

    def connected(self, start, goal) -> bool:
        if not self.is_valid(start) or not self.is_valid(goal):
            return False
        first, last = self.cell(start), self.cell(goal)
        return bool(self.labels[first] >= 0 and self.labels[first] == self.labels[last])

    def segment_valid(self, start, goal) -> bool:
        # Exact segment-to-axis-aligned-box distance in 2D: an intersection
        # gives zero; otherwise the minimum is attained at an endpoint or corner.
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not self.is_valid(start) or not self.is_valid(goal):
            return False
        delta = goal - start
        length2 = float(delta @ delta)
        if length2 < 1e-15 or not len(self.lower):
            return True
        lower_t = np.full((len(self.lower), 2), -np.inf)
        upper_t = np.full((len(self.lower), 2), np.inf)
        for axis in range(2):
            if abs(delta[axis]) < 1e-12:
                outside = (start[axis] < self.lower[:, axis]) | (start[axis] > self.upper[:, axis])
                lower_t[outside, axis] = np.inf
                upper_t[outside, axis] = -np.inf
            else:
                values = np.stack(((self.lower[:, axis] - start[axis]) / delta[axis], (self.upper[:, axis] - start[axis]) / delta[axis]))
                lower_t[:, axis], upper_t[:, axis] = values.min(0), values.max(0)
        if np.any(np.maximum(lower_t.max(1), 0) <= np.minimum(upper_t.min(1), 1)):
            return False
        corners = np.concatenate((self.lower, self.upper, np.column_stack((self.lower[:, 0], self.upper[:, 1])), np.column_stack((self.upper[:, 0], self.lower[:, 1]))))
        t = np.clip(((corners - start) @ delta) / length2, 0, 1)
        distances = np.linalg.norm(corners - (start + t[:, None] * delta), axis=1)
        return bool(distances.min() >= self.clearance - 1e-7)

    def path(self, start, goal, *, smooth: bool = True) -> np.ndarray:
        if not self.connected(start, goal):
            raise ValueError("Start/goal are not connected in robot-clearance free space")
        first, last = self.cell(start), self.cell(goal)
        queue, parents = deque([first]), {first: None}
        while queue and last not in parents:
            current = queue.popleft()
            for other in self.neighbours(current):
                if other not in parents:
                    parents[other] = current
                    queue.append(other)
        cells, current = [], last
        while current is not None:
            cells.append(current)
            current = parents[current]
        points = [np.asarray(start, float)] + [np.asarray(self.grid.cell_center(*cell)) for cell in reversed(cells)] + [np.asarray(goal, float)]
        points = [point for i, point in enumerate(points) if i == 0 or np.linalg.norm(point - points[i - 1]) > 1e-7]
        if smooth:
            reduced, index = [points[0]], 0
            while index < len(points) - 1:
                next_index = len(points) - 1
                while next_index > index + 1 and not self.segment_valid(points[index], points[next_index]):
                    next_index -= 1
                reduced.append(points[next_index])
                index = next_index
            points = reduced
        if any(not self.segment_valid(a, b) for a, b in zip(points, points[1:])):
            raise ValueError("Endpoint-to-grid connection violates clearance")
        return np.asarray(points)


def path_length(path) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())
