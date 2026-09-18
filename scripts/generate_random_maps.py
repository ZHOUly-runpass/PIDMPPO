from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pidmppo.envs import generate_random_map


DIFFICULTIES = {
    "easy": 0.10,
    "medium": 0.18,
    "hard": 0.28,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible zero-shot navigation maps")
    parser.add_argument("--output", type=Path, default=Path("generated_maps"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=24)
    return parser.parse_args()


def serialize(grid) -> list[str]:
    characters = np.full(grid.occupied.shape, ".", dtype="<U1")
    characters[grid.occupied] = "#"
    for point, marker in ((grid.starts[0], "S"), (grid.goals[0], "G")):
        column = int(point[0] // grid.cell_size)
        row = int(point[1] // grid.cell_size)
        characters[row, column] = marker
    return ["".join(row) for row in characters]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "maps": []}
    scenarios = {"version": 1, "coordinate_unit": "metre", "scenarios": []}
    for difficulty, density in DIFFICULTIES.items():
        for geometry in ("none", "u", "c", "concave"):
            for seed in args.seeds:
                grid = generate_random_map(
                    width=args.width,
                    height=args.height,
                    obstacle_density=density,
                    trap_geometry=geometry,
                    seed=seed,
                )
                name = f"{difficulty}_{geometry}_{seed}"
                path = args.output / f"{name}.txt"
                path.write_text("\n".join(serialize(grid)) + "\n", encoding="utf-8")
                manifest["maps"].append(
                    {
                        "id": name,
                        "path": str(path),
                        "difficulty": difficulty,
                        "trap_geometry": geometry,
                        "density": density,
                        "seed": seed,
                    }
                )
                start, goal = grid.starts[0], grid.goals[0]
                scenarios["scenarios"].append(
                    {
                        "id": name,
                        "map": name,
                        "map_file": str(path),
                        "start": list(start),
                        "goal": list(goal),
                        "theta": float(np.arctan2(goal[1] - start[1], goal[0] - start[0])),
                        "difficulty": difficulty,
                    }
                )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output / "scenarios.json").write_text(
        json.dumps(scenarios, indent=2), encoding="utf-8"
    )
    print(f"generated={len(manifest['maps'])} output={args.output}")


if __name__ == "__main__":
    main()
