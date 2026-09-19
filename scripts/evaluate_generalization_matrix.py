from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on the fixed randomized-map suite")
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment_matrix.yaml"))
    parser.add_argument("--scenarios", type=Path, default=Path("configs/scenarios_v2/random.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/generalization_v2"))
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    if args.generate:
        subprocess.run([sys.executable, "scripts/generate_scenarios_v2.py", "--suite", "random", "--output", str(args.scenarios.parent)], check=True)
    if not args.scenarios.exists():
        raise FileNotFoundError(
            f"Missing {args.scenarios}; run with --generate before scheduling evaluation"
        )
    variants = args.variants or raw["core_variants"]
    unknown = sorted(set(variants) - set(raw["core_variants"]))
    if unknown:
        raise ValueError(f"Variants are not in the core matrix: {unknown}")
    jobs = []
    for variant in variants:
        for seed in raw["seeds"]:
            checkpoint = Path(raw["output_root"]) / variant / f"seed_{seed}" / "checkpoints/final.pt"
            output = args.output / variant / f"seed_{seed}"
            command = [
                sys.executable,
                "scripts/evaluate.py",
                str(checkpoint),
                "--variant", variant,
                "--scenarios", str(args.scenarios),
                "--episodes-per-map", "1",
                "--training-seed", str(seed),
                "--output", str(output),
            ]
            jobs.append((checkpoint, command))
    print(f"jobs={len(jobs)} randomized_scenarios={args.scenarios} execute={args.execute}")
    for index, (checkpoint, command) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {shlex.join(command)}")
        if args.execute:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
