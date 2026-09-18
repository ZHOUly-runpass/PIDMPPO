from __future__ import annotations

import argparse
import csv
import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze conservative value error from trace CSV files")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation/value_analysis.json"))
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        event_group = "trap_or_stagnation" if row.get("event") in {"trap", "stagnation"} else "normal"
        method = row.get("method") or "unspecified"
        groups[(method, event_group)].append(row)
    output = {}
    for (method, event_group), group in groups.items():
        mc = np.asarray([float(row["mc_return"]) for row in group])
        v1 = np.asarray([float(row["value1"]) for row in group])
        v2 = np.asarray([float(row["value2"]) for row in group])
        minimum = np.minimum(v1, v2)
        output[f"{method}:{event_group}"] = {
            "method": method,
            "event_group": event_group,
            "samples": len(group),
            "v1_mae": float(np.mean(np.abs(v1 - mc))),
            "v2_mae": float(np.mean(np.abs(v2 - mc))),
            "minimum_mae": float(np.mean(np.abs(minimum - mc))),
            "v1_bias": float(np.mean(v1 - mc)),
            "v2_bias": float(np.mean(v2 - mc)),
            "minimum_bias": float(np.mean(minimum - mc)),
            "head_pearson": float(np.corrcoef(v1, v2)[0, 1]) if len(group) > 1 else float("nan"),
        }
    return output


def main() -> None:
    args = parse_args()
    rows = []
    expanded = [Path(match) for item in args.inputs for match in (glob.glob(str(item)) or [str(item)])]
    for path in expanded:
        with path.open("r", newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    output = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
