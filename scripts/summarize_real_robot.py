from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pidmppo.evaluation import bootstrap_interval, wilson_interval


REQUIRED_COLUMNS = {
    "method",
    "map",
    "trial_id",
    "success",
    "collision",
    "timeout",
    "path_length",
    "elapsed_seconds",
}


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real-robot trials without modifying them")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation/real_robot_summary.json"))
    args = parser.parse_args()
    with args.input.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing columns: {sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ())) }")
        rows = list(reader)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["map"])].append(row)
    output = []
    for (method, map_name), group in sorted(groups.items()):
        successes = sum(as_bool(row["success"]) for row in group)
        low, high = wilson_interval(successes, len(group))
        item = {
            "method": method,
            "map": map_name,
            "trials": len(group),
            "success_rate": successes / len(group),
            "success_ci_low": low,
            "success_ci_high": high,
            "collision_rate": np.mean([as_bool(row["collision"]) for row in group]),
            "timeout_rate": np.mean([as_bool(row["timeout"]) for row in group]),
        }
        for metric in ("path_length", "elapsed_seconds"):
            values = [float(row[metric]) for row in group]
            interval = bootstrap_interval(values, samples=10_000, seed=0)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_ci_low"], item[f"{metric}_ci_high"] = interval
        output.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"groups={len(output)} output={args.output}")


if __name__ == "__main__":
    main()
