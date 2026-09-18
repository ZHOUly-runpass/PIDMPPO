from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from pidmppo.envs import load_map


def connected(grid, start: tuple[float, float], goal: tuple[float, float]) -> bool:
    start_cell = (int(start[1] // grid.cell_size), int(start[0] // grid.cell_size))
    goal_cell = (int(goal[1] // grid.cell_size), int(goal[0] // grid.cell_size))
    frontier = [start_cell]
    visited = {start_cell}
    while frontier:
        row, column = frontier.pop()
        if (row, column) == goal_cell:
            return True
        for candidate in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            next_row, next_column = candidate
            if 0 <= next_row < grid.occupied.shape[0] and 0 <= next_column < grid.occupied.shape[1]:
                if not grid.occupied[candidate] and candidate not in visited:
                    visited.add(candidate)
                    frontier.append(candidate)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and optionally render simulation assets")
    parser.add_argument("--output", type=Path, default=Path("artifacts/simulation_assets.json"))
    parser.add_argument("--render", type=Path, help="Optional directory for PNG map previews")
    return parser.parse_args()


def inspect_urdf() -> dict[str, object]:
    resource = files("pidmppo").joinpath("assets/robots/turtlebot3_burger_sim.urdf")
    root = ET.fromstring(resource.read_text(encoding="utf-8"))
    joints = [joint.attrib["name"] for joint in root.findall("joint")]
    if joints[:2] != ["left_wheel_joint", "right_wheel_joint"]:
        raise ValueError(f"Unexpected wheel joint order: {joints}")
    wheel = root.find("./link[@name='left_wheel_link']/collision/geometry/cylinder")
    body = root.find("./link[@name='base_link']/collision/geometry/cylinder")
    if wheel is None or body is None:
        raise ValueError("URDF is missing wheel or body collision geometry")
    return {
        "resource": str(resource),
        "joint_indices": {name: index for index, name in enumerate(joints)},
        "wheel_radius_m": float(wheel.attrib["radius"]),
        "body_radius_m": float(body.attrib["radius"]),
    }


def inspect_maps(render_directory: Path | None) -> list[dict[str, object]]:
    summaries = []
    if render_directory is not None:
        render_directory.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError("Rendering requires `pip install -e .[analysis]`") from error
    for name in ("map1", "map2", "map3", "map4"):
        grid = load_map(name)
        start, goal = grid.starts[0], grid.goals[0]
        if grid.is_occupied(*start) or grid.is_occupied(*goal):
            raise ValueError(f"{name}: start or goal is occupied")
        if not connected(grid, start, goal):
            raise ValueError(f"{name}: manuscript start and goal are disconnected")
        summaries.append(
            {
                "name": name,
                "source": grid.source,
                "size_m": [grid.width, grid.height],
                "resolution_m": grid.cell_size,
                "wall_thickness_m": grid.wall_thickness,
                "segments": len(grid.segments),
                "start": list(start),
                "goal": list(goal),
                "start_goal_connected": True,
                "free_fraction": float(np.mean(~grid.occupied)),
            }
        )
        if render_directory is not None:
            figure, axis = plt.subplots(figsize=(8, 5))
            for x1, y1, x2, y2 in grid.segments:
                axis.plot((x1, x2), (y1, y2), color="black", linewidth=3)
            axis.scatter(*start, color="tab:blue", marker="o", label="start")
            axis.scatter(*goal, color="tab:red", marker="*", s=120, label="goal")
            axis.set(xlim=(0, grid.width), ylim=(0, grid.height), aspect="equal", title=name)
            axis.legend()
            figure.tight_layout()
            figure.savefig(render_directory / f"{name}.png", dpi=180)
            plt.close(figure)
    return summaries


def main() -> None:
    args = parse_args()
    payload = {
        "status": "valid",
        "maps": inspect_maps(args.render),
        "robot": inspect_urdf(),
        "provenance": "manuscript-figure reconstruction; not original numerical assets",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"maps={len(payload['maps'])} status=valid output={args.output}")


if __name__ == "__main__":
    main()
