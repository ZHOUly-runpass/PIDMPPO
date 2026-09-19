"""Generate immutable, clearance- and time-feasible validation/test assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pidmppo.config import EnvConfig
from pidmppo.envs import MaplessNavigationEnv
from pidmppo.envs.geometry import ClearanceSpace, path_length
from pidmppo.envs.maps import generate_corridor_map, generate_random_map, load_map
from pidmppo.evaluation.reference import validate_reference


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def grid_hash(grid):
    payload = {"occupied": grid.occupied.astype(int).tolist(), "cell_size": grid.cell_size}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_grid(path, grid):
    dump(path, {"occupied": grid.occupied.astype(int).tolist(), "starts": grid.starts, "goals": grid.goals,
                "cell_size": grid.cell_size, "source": grid.source})


def sample_scenes(grid, count, *, rng, map_name, map_file=None, max_steps=540, minimum=2., maximum=8., difficulty="fixed", euclidean_minimum=0.):
    env = MaplessNavigationEnv(EnvConfig(map_name=map_name, map_file=str(map_file) if map_file else None,
                                      randomize_obstacles=False, randomize_start_goal=False))
    seen, scenarios = set(), []
    try:
        for attempt in range(max(500, count * 150)):
            if len(scenarios) == count:
                return scenarios
            choices = rng.choice(len(env._space.centres), size=2, replace=False)
            start, goal = env._space.centres[choices]
            # Uniqueness is the ordered start/goal task, NOT orientation or ID.
            key = tuple(float(value) for value in np.r_[start, goal])
            if key in seen or not env._space.connected(start, goal) or np.linalg.norm(goal - start) < euclidean_minimum:
                continue
            path = env._space.path(start, goal)
            if not minimum <= path_length(path) <= maximum:
                continue
            theta = float(rng.uniform(-np.pi, np.pi))
            try:
                acceptance = validate_reference(env, start, goal, theta, max_steps)
            except ValueError:
                continue
            seen.add(key)
            scenarios.append({"id": f"{map_name}_{len(scenarios):03d}", "map": map_name, "start": start.tolist(), "goal": goal.tolist(),
                              "theta": theta, "difficulty": difficulty, "grid_sha256": grid_hash(grid), **acceptance,
                              **({"map_file": f"maps/{map_file.name}", "map_sha256": hashlib.sha256(map_file.read_bytes()).hexdigest()} if map_file else {})})
        raise RuntimeError(f"Only {len(scenarios)}/{count} feasible scenarios for {map_name}")
    finally:
        env.close()


def write_manifest(path, scenarios, split, seed):
    payload = {"version": 2, "split": split, "generation_seed": seed, "coordinate_unit": "metre", "backend": "grid",
               "clearance": .15, "max_reference_steps": 540, "sampling": "without-replacement ordered start/goal, uniform initial heading",
               "scenarios": scenarios}
    dump(path, payload)
    with path.with_suffix(".sha256").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")


def canonicalize_assets(root):
    """Mechanical LF normalization with dependent hashes refreshed, no resampling."""
    for path in (root / "maps").glob("*.json"):
        dump(path, json.loads(path.read_text(encoding="utf-8")))
    for name in ("fixed", "validation", "random"):
        path = root / f"{name}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") != 2:
            raise ValueError("Canonicalization only accepts v2 assets")
        for scene in raw["scenarios"]:
            if scene.get("map_file"):
                scene["map_sha256"] = hashlib.sha256((root / scene["map_file"]).read_bytes()).hexdigest()
        dump(path, raw)
        with path.with_suffix(".sha256").open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    acceptance = root / "acceptance.json"
    if acceptance.exists():
        raw = json.loads(acceptance.read_text(encoding="utf-8"))
        raw["manifests"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.json") if p.name != "acceptance.json"}
        dump(acceptance, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("configs/scenarios_v2"))
    parser.add_argument("--fixed-count", type=int, default=100)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--suite", choices=["all", "fixed", "validation", "random"], default="all")
    parser.add_argument("--canonicalize", action="store_true", help="Normalize existing v2 JSON to LF and refresh hashes without changing tasks")
    args = parser.parse_args()
    if args.canonicalize:
        canonicalize_assets(args.output)
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing versioned assets: {args.output}")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    assets = []
    if args.suite in {"all", "fixed"}:
        scenes = []
        for name in ("map1", "map2", "map3", "map4"):
            scenes.extend(sample_scenes(load_map(name), args.fixed_count, rng=rng, map_name=name))
            print(f"fixed {name}: {args.fixed_count} accepted", flush=True)
        write_manifest(args.output / "fixed.json", scenes, "final_fixed_test", args.seed)
        assets.extend(scenes)
    if args.suite in {"all", "validation"}:
        scenes = []
        # Each held-out generator seed is independent of train seeds 11/22/33/...;
        # validation uses the complete training geometry/distance distribution.
        for index in range(args.validation_count):
            seed = 9_100_000 + index
            grid = generate_corridor_map(seed=seed)
            path = args.output / "maps" / f"validation_{seed}.json"
            save_grid(path, grid)
            scene = sample_scenes(grid, 1, rng=rng, map_name="validation", map_file=path,
                                  minimum=0., maximum=float("inf"), euclidean_minimum=max(1.5, .35 * grid.diagonal), difficulty="training_distribution")[0]
            scene.update(id=f"validation_{seed}", generator_seed=seed)
            scenes.append(scene)
            if (index + 1) % 10 == 0:
                print(f"validation: {index + 1} accepted", flush=True)
        write_manifest(args.output / "validation.json", scenes, "training_distribution_validation", args.seed)
        assets.extend(scenes)
    if args.suite in {"all", "random"}:
        scenes = []
        for difficulty, density in (("easy", .10), ("medium", .18), ("hard", .28)):
            for geometry in ("none", "u", "c", "concave"):
                for seed in (101, 102, 103, 104, 105):
                    grid = generate_random_map(obstacle_density=density, trap_geometry=geometry, seed=seed)
                    name = f"{difficulty}_{geometry}_{seed}"
                    path = args.output / "maps" / f"{name}.json"
                    save_grid(path, grid)
                    scene = sample_scenes(grid, 1, rng=rng, map_name=name, map_file=path, difficulty=difficulty)[0]
                    scene.update(trap_geometry=geometry, generator_seed=seed)
                    scenes.append(scene)
                print(f"random {difficulty}/{geometry}: 5 accepted", flush=True)
        write_manifest(args.output / "random.json", scenes, "final_zero_shot_test", args.seed)
        assets.extend(scenes)
    dump(args.output / "acceptance.json", {"status": "passed", "version": 2, "scenes": len(assets),
         "all_zero_action_passed": all(s["zero_action_passed"] for s in assets), "all_connected": all(s["connected"] for s in assets),
         "max_reference_steps": max(s["reference_steps"] for s in assets), "minimum_clearance": .15,
         "manifests": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.json") if p.name != "acceptance.json"}})


if __name__ == "__main__":
    main()
