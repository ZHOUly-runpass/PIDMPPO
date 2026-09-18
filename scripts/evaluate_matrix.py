from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-seed evaluation jobs; execution is opt-in")
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment_matrix.yaml"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/revision"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--trace-episodes", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    base_config = Path(raw["base_config"])
    run_root = Path(raw["output_root"])
    rule = "final.pt"
    jobs = []
    for variant in raw["core_variants"]:
        for seed in raw["seeds"]:
            checkpoint = run_root / variant / f"seed_{seed}" / "checkpoints" / rule
            output = args.output / variant / f"seed_{seed}"
            command = [
                sys.executable,
                "scripts/evaluate.py",
                str(checkpoint),
                "--config",
                str(base_config),
                "--variant",
                variant,
                "--training-seed",
                str(seed),
                "--output",
                str(output),
                "--trace-episodes",
                str(args.trace_episodes),
            ]
            jobs.append((checkpoint, command))
    print(f"jobs={len(jobs)} execute={args.execute} checkpoint_rule=final")
    for index, (checkpoint, command) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {shlex.join(command)}")
        if args.execute:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
