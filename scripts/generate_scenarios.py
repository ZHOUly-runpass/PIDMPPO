from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from pidmppo.envs import load_map


DIFFICULTY = {"map1": "medium", "map2": "hard", "map3": "medium", "map4": "hard"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fixed, reproducible start-goal manifest")
    parser.add_argument("--maps", nargs="+", default=["map1", "map2", "map3", "map4"])
    parser.add_argument("--per-map", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--minimum-distance", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("configs/scenarios.json"))
    return parser.parse_args()


def connected(occupied: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        row, column = current
        for cell in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            r, c = cell
            if 0 <= r < occupied.shape[0] and 0 <= c < occupied.shape[1]:
                if not occupied[r, c] and cell not in visited:
                    visited.add(cell)
                    queue.append(cell)
    return False


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    scenarios = []
    map_geometry = {}
    for map_name in args.maps:
        grid = load_map(map_name)
        map_geometry[map_name] = {
            "source": grid.source,
            "size_m": [grid.width, grid.height],
            "resolution_m": grid.cell_size,
            "wall_thickness_m": grid.wall_thickness,
        }
        free_cells = [tuple(map(int, cell)) for cell in np.argwhere(~grid.occupied)]
        accepted: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        attempts = 0
        while len(accepted) < args.per_map and attempts < 20_000:
            attempts += 1
            indices = rng.choice(len(free_cells), size=2, replace=False)
            start_cell, goal_cell = free_cells[int(indices[0])], free_cells[int(indices[1])]
            start = grid.cell_center(*start_cell)
            goal = grid.cell_center(*goal_cell)
            pair = (start_cell, goal_cell)
            if pair in accepted or np.linalg.norm(np.subtract(goal, start)) < args.minimum_distance:
                continue
            if connected(grid.occupied, start_cell, goal_cell):
                accepted.add(pair)
        if len(accepted) != args.per_map:
            raise RuntimeError(f"Could only generate {len(accepted)} scenarios for {map_name}")
        for index, (start_cell, goal_cell) in enumerate(sorted(accepted), start=1):
            start = grid.cell_center(*start_cell)
            goal = grid.cell_center(*goal_cell)
            theta = float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
            scenarios.append(
                {
                    "id": f"{map_name}_{index:03d}",
                    "map": map_name,
                    "start": list(start),
                    "goal": list(goal),
                    "theta": theta,
                    "difficulty": DIFFICULTY.get(map_name, "unclassified"),
                }
            )
    payload = {
        "version": 1,
        "coordinate_unit": "metre",
        "generator_seed": args.seed,
        "scenarios_per_map": args.per_map,
        "map_geometry": map_geometry,
        "scenarios": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"scenarios={len(scenarios)} output={args.output}")


if __name__ == "__main__":
    main()
