from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pidmppo.evaluation import bootstrap_interval


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two methods across independent training seeds")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--metric", default="success")
    parser.add_argument("--output", type=Path, default=Path("evaluation/method_comparison.json"))
    args = parser.parse_args()
    paths = [Path(match) for item in args.inputs for match in (glob.glob(str(item)) or [str(item)])]
    rows = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] in {args.method_a, args.method_b}:
            grouped[(row["method"], row["map"], row["training_seed"])].append(row)
    output = []
    maps = sorted({key[1] for key in grouped})
    for map_name in maps:
        a_values = []
        b_values = []
        for (method, current_map, _), group in grouped.items():
            if current_map != map_name:
                continue
            values = [as_bool(row[args.metric]) for row in group] if args.metric == "success" else [float(row[args.metric]) for row in group]
            (a_values if method == args.method_a else b_values).append(float(np.mean(values)))
        if not a_values or not b_values:
            continue
        # Seeds are independent; bootstrap the difference between seed-level means.
        rng = np.random.default_rng(0)
        differences = []
        for _ in range(10_000):
            a_sample = rng.choice(a_values, size=len(a_values), replace=True)
            b_sample = rng.choice(b_values, size=len(b_values), replace=True)
            differences.append(float(np.mean(a_sample) - np.mean(b_sample)))
        low, high = np.quantile(differences, [0.025, 0.975])
        output.append(
            {
                "map": map_name,
                "metric": args.metric,
                "method_a": args.method_a,
                "method_b": args.method_b,
                "mean_difference_a_minus_b": float(np.mean(a_values) - np.mean(b_values)),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "seeds_a": len(a_values),
                "seeds_b": len(b_values),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
