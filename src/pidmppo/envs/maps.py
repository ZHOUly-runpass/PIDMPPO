from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import math
from pathlib import Path

import numpy as np


MAPS: dict[str, tuple[str, ...]] = {
    "map1": (
        "#############",
        "#S..........#",
        "#..##.......#",
        "#..##..###..#",
        "#.......#...#",
        "#.####...#..#",
        "#..........G#",
        "#############",
    ),
    "map2": (
        "#############",
        "#S....#....G#",
        "#.....#.....#",
        "#.###...###.#",
        "#.....#.....#",
        "#.....#.....#",
        "#...........#",
        "#############",
    ),
    "map3": (
        "############",
        "#S.........#",
        "#.######...#",
        "#......#...#",
        "#..##..#...#",
        "#..##......#",
        "#.........G#",
        "############",
    ),
    "map4": (
        "############",
        "#S...#.....#",
        "#....#..##.#",
        "#.##....##.#",
        "#.##.##....#",
        "#....##....#",
        "#.........G#",
        "############",
    ),
}


@dataclass(frozen=True)
class GridMap:
    occupied: np.ndarray
    starts: tuple[tuple[float, float], ...]
    goals: tuple[tuple[float, float], ...]
    cell_size: float = 0.50
    segments: tuple[tuple[float, float, float, float], ...] = ()
    wall_thickness: float = 0.0
    source: str = "ascii"

    @property
    def width(self) -> float:
        return self.occupied.shape[1] * self.cell_size

    @property
    def height(self) -> float:
        return self.occupied.shape[0] * self.cell_size

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    def cell_center(self, row: int, column: int) -> tuple[float, float]:
        return ((column + 0.5) * self.cell_size, (row + 0.5) * self.cell_size)

    def is_occupied(self, x: float, y: float) -> bool:
        column = int(np.floor(x / self.cell_size))
        row = int(np.floor(y / self.cell_size))
        if row < 0 or column < 0 or row >= self.occupied.shape[0] or column >= self.occupied.shape[1]:
            return True
        return bool(self.occupied[row, column])

    def free_centers(self) -> np.ndarray:
        rows, columns = np.nonzero(~self.occupied)
        return np.asarray([self.cell_center(int(r), int(c)) for r, c in zip(rows, columns)], dtype=np.float32)


def _parse_lines(lines: tuple[str, ...], name: str, cell_size: float) -> GridMap:
    widths = {len(line) for line in lines}
    if len(widths) != 1:
        raise ValueError(f"Map {name!r} is not rectangular")
    occupied = np.zeros((len(lines), len(lines[0])), dtype=bool)
    starts: list[tuple[float, float]] = []
    goals: list[tuple[float, float]] = []
    for row, line in enumerate(lines):
        for column, symbol in enumerate(line):
            occupied[row, column] = symbol == "#"
            point = ((column + 0.5) * cell_size, (row + 0.5) * cell_size)
            if symbol == "S":
                starts.append(point)
            elif symbol == "G":
                goals.append(point)
    if not starts or not goals:
        raise ValueError(f"Map {name!r} requires at least one S and one G")
    return GridMap(occupied, tuple(starts), tuple(goals), cell_size)


def _point_segment_distance(
    x: float, y: float, segment: tuple[float, float, float, float]
) -> float:
    x1, y1, x2, y2 = segment
    dx, dy = x2 - x1, y2 - y1
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return float(np.hypot(x - x1, y - y1))
    projection = np.clip(((x - x1) * dx + (y - y1) * dy) / denominator, 0.0, 1.0)
    return float(np.hypot(x - (x1 + projection * dx), y - (y1 + projection * dy)))


@lru_cache(maxsize=4)
def _load_paper_map(name: str) -> GridMap:
    resource = files("pidmppo").joinpath("assets/maps/paper_maps.json")
    raw = json.loads(resource.read_text(encoding="utf-8"))
    try:
        specification = raw["maps"][name]
    except KeyError as error:
        raise ValueError(f"Unknown manuscript map {name!r}") from error

    cell_size = float(raw["resolution"])
    wall_thickness = float(raw["wall_thickness"])
    width, height = (float(value) for value in specification["size"])
    columns = math.ceil(width / cell_size)
    rows = math.ceil(height / cell_size)
    segments = tuple(tuple(float(value) for value in segment) for segment in specification["segments"])
    occupied = np.zeros((rows, columns), dtype=bool)

    # Rasterisation is deliberately conservative: any cell touched by a wall is
    # occupied. PyBullet uses the original metric segments rather than this mask.
    clearance = wall_thickness / 2.0 + cell_size * np.sqrt(2.0) / 2.0
    for row in range(rows):
        for column in range(columns):
            x = (column + 0.5) * cell_size
            y = (row + 0.5) * cell_size
            occupied[row, column] = any(
                _point_segment_distance(x, y, segment) <= clearance for segment in segments
            )

    start = tuple(float(value) for value in specification["start"])
    goal = tuple(float(value) for value in specification["goal"])
    grid_map = GridMap(
        occupied=occupied,
        starts=(start,),
        goals=(goal,),
        cell_size=cell_size,
        segments=segments,
        wall_thickness=wall_thickness,
        source="manuscript_figure_reconstruction",
    )
    if grid_map.is_occupied(*start) or grid_map.is_occupied(*goal):
        raise ValueError(f"Reconstructed {name} has an occupied start or goal")
    return grid_map


def load_map(name: str, cell_size: float = 0.50, map_file: str | Path | None = None) -> GridMap:
    if map_file is not None:
        path = Path(map_file)
        if path.suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            return GridMap(np.asarray(raw["occupied"], dtype=bool), tuple(map(tuple, raw["starts"])),
                           tuple(map(tuple, raw["goals"])), float(raw["cell_size"]), source=raw.get("source", "v2_asset"))
        lines = tuple(line.rstrip("\n\r") for line in path.read_text(encoding="utf-8").splitlines())
        lines = tuple(line for line in lines if line)
        return _parse_lines(lines, str(path), cell_size)
    if name in MAPS:
        return _load_paper_map(name)
    try:
        lines = MAPS[name]
    except KeyError as error:
        raise ValueError(f"Unknown map {name!r}; choose one of {sorted(MAPS)}") from error
    return _parse_lines(lines, name, cell_size)


def generate_random_map(
    *,
    width: int = 24,
    height: int = 24,
    obstacle_density: float = 0.18,
    trap_geometry: str = "none",
    seed: int = 0,
    cell_size: float = 0.50,
) -> GridMap:
    """Create a reproducible connected map for zero-shot evaluation.

    ``trap_geometry`` may be ``none``, ``u``, ``c`` or ``concave``. The
    generator retries until the selected start and goal are connected.
    """

    if not 0.0 <= obstacle_density <= 0.45:
        raise ValueError("obstacle_density must be in [0, 0.45]")
    if trap_geometry not in {"none", "u", "c", "concave"}:
        raise ValueError(f"Unsupported trap geometry: {trap_geometry}")
    rng = np.random.default_rng(seed)
    for _ in range(200):
        occupied = rng.random((height, width)) < obstacle_density
        occupied[[0, -1], :] = True
        occupied[:, [0, -1]] = True
        start_cell = (height - 2, 1)
        goal_cell = (1, width - 2)
        occupied[start_cell] = False
        occupied[goal_cell] = False
        _draw_trap(occupied, trap_geometry)
        occupied[start_cell] = False
        occupied[goal_cell] = False
        if _connected(occupied, start_cell, goal_cell):
            temporary = GridMap(occupied, (), (), cell_size)
            return GridMap(
                occupied,
                (temporary.cell_center(*start_cell),),
                (temporary.cell_center(*goal_cell),),
                cell_size,
            )
    raise RuntimeError("Could not generate a connected map; reduce obstacle_density")


def generate_corridor_map(
    *,
    width: int = 26,
    height: int = 16,
    obstacle_segments: int = 8,
    seed: int = 0,
    cell_size: float = 0.25,
) -> GridMap:
    """Generate a connected corridor-like randomized Map 1."""

    rng = np.random.default_rng(seed)
    for _ in range(200):
        occupied = np.zeros((height, width), dtype=bool)
        occupied[[0, -1], :] = True
        occupied[:, [0, -1]] = True
        for _ in range(obstacle_segments):
            horizontal = bool(rng.integers(0, 2))
            length = int(rng.integers(3, max(4, min(width, height) // 2)))
            row = int(rng.integers(2, height - 2))
            column = int(rng.integers(2, width - 2))
            if horizontal:
                occupied[row, column : min(width - 1, column + length)] = True
            else:
                occupied[row : min(height - 1, row + length), column] = True
        start_cell = (height - 2, 2)
        goal_cell = (1, width - 3)
        occupied[start_cell] = False
        occupied[goal_cell] = False
        reachable = _reachable_cells(occupied, start_cell)
        free_count = int((~occupied).sum())
        if goal_cell in reachable and len(reachable) >= int(0.95 * free_count):
            temporary = GridMap(occupied, (), (), cell_size)
            return GridMap(
                occupied,
                (temporary.cell_center(*start_cell),),
                (temporary.cell_center(*goal_cell),),
                cell_size,
            )
    raise RuntimeError("Could not generate a connected corridor map")


def _draw_trap(occupied: np.ndarray, geometry: str) -> None:
    if geometry == "none":
        return
    height, width = occupied.shape
    center_r, center_c = height // 2, width // 2
    radius = max(2, min(height, width) // 6)
    occupied[center_r - radius : center_r + radius + 1, center_c - radius] = True
    occupied[center_r - radius : center_r + radius + 1, center_c + radius] = True
    if geometry in {"u", "concave"}:
        occupied[center_r + radius, center_c - radius : center_c + radius + 1] = True
    if geometry in {"c", "concave"}:
        occupied[center_r - radius, center_c - radius : center_c + radius + 1] = True
        opening = slice(center_r - 1, center_r + 2)
        occupied[opening, center_c + radius] = False


def _connected(occupied: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    return goal in _reachable_cells(occupied, start)


def _reachable_cells(occupied: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    frontier = [start]
    visited = {start}
    while frontier:
        row, column = frontier.pop()
        for next_cell in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            r, c = next_cell
            if 0 <= r < occupied.shape[0] and 0 <= c < occupied.shape[1]:
                if not occupied[r, c] and next_cell not in visited:
                    visited.add(next_cell)
                    frontier.append(next_cell)
    return visited
