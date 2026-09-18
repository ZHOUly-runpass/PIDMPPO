from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Job:
    name: str
    variant: str
    seed: int
    overrides: tuple[str, ...]
    output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the revision experiment matrix; execution is opt-in"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment_matrix.yaml"))
    parser.add_argument("--stage", choices=("core", "sensitivity", "all"), default="core")
    parser.add_argument("--execute", action="store_true", help="Actually launch jobs sequentially")
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def load_jobs(path: Path, stage: str) -> tuple[Path, list[Job]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_config = Path(raw["base_config"])
    output_root = Path(raw["output_root"])
    seeds = [int(seed) for seed in raw["seeds"]]
    jobs: list[Job] = []
    if stage in {"core", "all"}:
        for variant in raw["core_variants"]:
            for seed in seeds:
                jobs.append(Job(variant, variant, seed, (), output_root / variant / f"seed_{seed}"))
    if stage in {"sensitivity", "all"}:
        for item in raw["sensitivity"]:
            for seed in seeds:
                jobs.append(
                    Job(
                        item["name"],
                        item["variant"],
                        seed,
                        tuple(item.get("overrides", ())),
                        output_root / "sensitivity" / item["name"] / f"seed_{seed}",
                    )
                )
    return base_config, jobs


def command_for(job: Job, base_config: Path, steps: int | None) -> list[str]:
    command = [
        sys.executable,
        "scripts/train.py",
        "--config",
        str(base_config),
        "--variant",
        job.variant,
        "--seed",
        str(job.seed),
        "--output",
        str(job.output),
    ]
    if steps is not None:
        command.extend(("--steps", str(steps)))
    for override in job.overrides:
        command.extend(("--set", override))
    return command


def main() -> None:
    args = parse_args()
    base_config, jobs = load_jobs(args.matrix, args.stage)
    print(f"jobs={len(jobs)} execute={args.execute}")
    for index, job in enumerate(jobs, start=1):
        command = command_for(job, base_config, args.steps)
        print(f"[{index}/{len(jobs)}] {shlex.join(command)}")
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

