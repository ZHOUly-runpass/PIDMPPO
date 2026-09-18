from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed U-trap mechanism episodes")
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment_matrix.yaml"))
    parser.add_argument("--scenarios", type=Path, default=Path("configs/mechanism_scenarios.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/mechanism"))
    parser.add_argument("--variants", nargs="+", default=["pidmppo"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    jobs = []
    for variant in args.variants:
        if variant not in raw["core_variants"]:
            raise ValueError(f"Variant {variant!r} is not in the core matrix")
        for seed in raw["seeds"]:
            checkpoint = Path(raw["output_root"]) / variant / f"seed_{seed}" / "checkpoints/final.pt"
            output = args.output / variant / f"seed_{seed}"
            command = [
                sys.executable,
                "scripts/evaluate.py",
                str(checkpoint),
                "--config", str(raw["base_config"]),
                "--variant", variant,
                "--scenarios", str(args.scenarios),
                "--episodes-per-map", str(args.episodes),
                "--trace-episodes", str(args.episodes),
                "--training-seed", str(seed),
                "--output", str(output),
            ]
            jobs.append((checkpoint, command))
    print(f"jobs={len(jobs)} episodes_per_job={args.episodes} execute={args.execute}")
    for index, (checkpoint, command) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {shlex.join(command)}")
        if args.execute:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
