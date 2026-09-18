from __future__ import annotations

import argparse
import csv
import json
import glob
from pathlib import Path

from pidmppo.evaluation import aggregate_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw evaluation rows with uncertainty")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--group-by", nargs="+", default=["method", "map"])
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("evaluation/summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    expanded = [Path(match) for item in args.inputs for match in (glob.glob(str(item)) or [str(item)])]
    for path in expanded:
        with path.open("r", newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    summaries = aggregate_results(
        rows,
        group_by=tuple(args.group_by),
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        raise ValueError("No evaluation rows found")
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    args.output.with_suffix(".json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"groups={len(summaries)} output={args.output}")


if __name__ == "__main__":
    main()
