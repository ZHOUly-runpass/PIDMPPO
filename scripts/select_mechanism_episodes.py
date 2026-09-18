from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select successful and failed mechanism traces")
    parser.add_argument("episodes", type=Path)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation/mechanism/selection.json"))
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--map-file", type=Path, default=Path("configs/maps/mechanism_u.txt"))
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    args = parse_args()
    with args.episodes.open("r", newline="", encoding="utf-8") as stream:
        episodes = list(csv.DictReader(stream))
    with args.traces.open("r", newline="", encoding="utf-8") as stream:
        traces = list(csv.DictReader(stream))
    traced_ids = {row["episode_id"] for row in traces}
    candidates = [row for row in episodes if row.get("episode_id") in traced_ids]
    successful = next((row for row in candidates if as_bool(row["success"])), None)
    failed = next((row for row in candidates if not as_bool(row["success"])), None)
    selection = {"success": successful, "failure": failed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.plot_dir is not None:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        for label, row in selection.items():
            if row is None:
                continue
            subprocess.run(
                [
                    sys.executable,
                    "scripts/plot_trace.py",
                    str(args.traces),
                    "--episode-id", row["episode_id"],
                    "--map-file", str(args.map_file),
                    "--output", str(args.plot_dir / f"{label}.png"),
                ],
                check=True,
            )
    print(f"success_found={successful is not None} failure_found={failed is not None} output={args.output}")


if __name__ == "__main__":
    main()
