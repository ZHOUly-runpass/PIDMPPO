from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

from pidmppo.evaluation import audit_evaluation_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Reject incomplete or inconsistent evaluation data")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--required-seeds", type=int, default=5)
    parser.add_argument("--required-episodes", type=int, default=100)
    args = parser.parse_args()
    paths = [Path(match) for item in args.inputs for match in (glob.glob(str(item)) or [str(item)])]
    rows = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    errors = audit_evaluation_rows(
        rows,
        required_seeds=args.required_seeds,
        required_episodes_per_map_seed=args.required_episodes,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"protocol_audit=passed rows={len(rows)} files={len(paths)}")


if __name__ == "__main__":
    main()
