from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pidmppo.config import EnvConfig
from pidmppo.envs import MaplessNavigationEnv
from pidmppo.evaluation import load_scenarios
from pidmppo.evaluation.reference import validate_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("configs/scenarios_v2"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/scenes_v2_audit.json"))
    args = parser.parse_args()
    report = {"status": "passed", "backend": "grid", "scenes": [], "manifests": {}}
    for suite, expected in (("fixed", 400), ("validation", 100), ("random", 60)):
        manifest = args.assets / f"{suite}.json"
        scenes = load_scenarios(manifest)
        assert len(scenes) == expected
        report["manifests"][suite] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        env, previous = None, None
        try:
            for scene in scenes:
                key = (scene.map, scene.map_file)
                if key != previous:
                    if env:
                        env.close()
                    env = MaplessNavigationEnv(EnvConfig(map_name=scene.map, map_file=scene.map_file, randomize_obstacles=False, randomize_start_goal=False))
                    previous = key
                result = validate_reference(env, scene.start, scene.goal, scene.theta)
                if suite == "random":
                    assert 2 <= result["reference_path_length"] <= 8
                report["scenes"].append({"id": scene.id, "suite": suite, **result})
        finally:
            if env:
                env.close()
        print(f"{suite}: {expected} independently revalidated", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
