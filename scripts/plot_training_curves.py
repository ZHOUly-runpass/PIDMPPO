from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all training seeds with mean and 95 percent CI")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--metric", default="success_rate")
    parser.add_argument("--output", type=Path, default=Path("evaluation/training_curves.png"))
    return parser.parse_args()


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Install plotting dependencies with `pip install -e .[analysis]`") from error
    args = parse_args()
    series = defaultdict(list)
    expanded = [Path(match) for item in args.inputs for match in (glob.glob(str(item)) or [str(item)])]
    for path in expanded:
        variant = path.parent.name if path.parent.name.startswith("seed_") is False else path.parent.parent.name
        with path.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        series[variant].append(
            (
                np.asarray([int(row["global_step"]) for row in rows]),
                np.asarray([float(row[args.metric]) for row in rows]),
            )
        )
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for variant, runs in sorted(series.items()):
        common_steps = runs[0][0]
        aligned = np.vstack([np.interp(common_steps, steps, values) for steps, values in runs])
        mean = aligned.mean(axis=0)
        sem = aligned.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean)
        axis.plot(common_steps, mean, label=variant)
        axis.fill_between(common_steps, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.2)
    axis.set(xlabel="Environment steps", ylabel=args.metric)
    axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
