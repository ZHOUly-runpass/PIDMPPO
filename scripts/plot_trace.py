from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PIDM mechanism traces")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument("--output", type=Path, default=Path("evaluation/mechanism_trace.png"))
    parser.add_argument("--map-file", type=Path)
    parser.add_argument("--cell-size", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Install plotting dependencies with `pip install -e .[analysis]`") from error
    args = parse_args()
    with args.trace.open("r", newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    if not all_rows:
        raise ValueError("Trace file is empty")
    episode_id = args.episode_id or all_rows[0]["episode_id"]
    rows = [row for row in all_rows if row["episode_id"] == episode_id]
    steps = np.asarray([int(row["step"]) for row in rows])
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    if args.map_file is not None:
        lines = [line for line in args.map_file.read_text(encoding="utf-8").splitlines() if line]
        occupied = np.asarray([[symbol == "#" for symbol in line] for line in lines], dtype=float)
        axes[0, 0].imshow(
            occupied,
            origin="lower",
            extent=(0, occupied.shape[1] * args.cell_size, 0, occupied.shape[0] * args.cell_size),
            cmap="Greys",
            vmin=0,
            vmax=1,
            alpha=0.35,
        )
    axes[0, 0].plot([float(row["x"]) for row in rows], [float(row["y"]) for row in rows])
    event_rows = [row for row in rows if row.get("event", "normal") != "normal"]
    if event_rows:
        axes[0, 0].scatter(
            [float(row["x"]) for row in event_rows],
            [float(row["y"]) for row in event_rows],
            c="tab:orange",
            s=12,
            label="trap/stagnation/terminal",
        )
    axes[0, 0].scatter(float(rows[0]["x"]), float(rows[0]["y"]), c="tab:blue", label="start")
    axes[0, 0].scatter(float(rows[-1]["x"]), float(rows[-1]["y"]), c="tab:red", marker="x", label="end")
    axes[0, 0].set(title="Trajectory", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()
    for key in ("distance_to_goal", "memory_norm"):
        if key in rows[0] and rows[0][key] != "":
            axes[0, 1].plot(steps, [float(row[key]) for row in rows], label=key)
    axes[0, 1].legend()
    for key in ("gate_mean", "gate_q25", "gate_q75"):
        if key in rows[0] and rows[0][key] != "":
            axes[1, 0].plot(steps, [float(row[key]) for row in rows], label=key)
    axes[1, 0].set_title("Memory gate")
    axes[1, 0].legend()
    for key in ("attention_p", "attention_i", "attention_d"):
        if key in rows[0] and rows[0][key] != "":
            axes[1, 1].plot(steps, [float(row[key]) for row in rows], label=key)
    axes[1, 1].set_title("P I D attention")
    axes[1, 1].legend()
    for key in ("value1", "value2", "value_min", "mc_return"):
        axes[2, 0].plot(steps, [float(row[key]) for row in rows], label=key)
    axes[2, 0].set_title("Value estimates and realized return")
    axes[2, 0].legend()
    axes[2, 1].plot(steps, [float(row["linear_action"]) for row in rows], label="linear")
    axes[2, 1].plot(steps, [float(row["angular_action"]) for row in rows], label="angular")
    axes[2, 1].set_title("Normalized actions")
    axes[2, 1].legend()
    fig.suptitle(episode_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
