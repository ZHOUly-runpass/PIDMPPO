from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml
import torch


METRICS = ("policy_loss", "value_loss", "entropy", "l_loss", "g_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit all smoke-training artifacts")
    parser.add_argument("--matrix", type=Path, default=Path("configs/smoke_matrix.yaml"))
    parser.add_argument("--minimum-steps", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke_training_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    root = Path(raw["output_root"])
    runs = []
    errors = []
    for variant in raw["core_variants"]:
        for seed in raw["seeds"]:
            directory = root / variant / f"seed_{seed}"
            checkpoint = directory / "checkpoints/final.pt"
            manifest = directory / "run_manifest.json"
            metrics = directory / "training_metrics.csv"
            missing = [str(path) for path in (checkpoint, manifest, metrics) if not path.exists()]
            record = {"variant": variant, "seed": seed, "directory": str(directory), "missing": missing}
            if missing:
                errors.append(f"{variant}/seed_{seed}: missing {missing}")
                runs.append(record)
                continue
            with metrics.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                errors.append(f"{variant}/seed_{seed}: empty metrics")
                runs.append(record)
                continue
            final_step = int(rows[-1]["global_step"])
            nonfinite = [
                f"row={index}:{name}"
                for index, row in enumerate(rows)
                for name in METRICS
                if not math.isfinite(float(row[name]))
            ]
            if final_step < args.minimum_steps:
                errors.append(f"{variant}/seed_{seed}: final step {final_step} < {args.minimum_steps}")
            if nonfinite:
                errors.append(f"{variant}/seed_{seed}: non-finite metrics {nonfinite}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_step = int(payload.get("global_step", -1))
            nonfinite_tensors = [
                name
                for name, tensor in payload["model"].items()
                if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all())
            ]
            if checkpoint_step < args.minimum_steps:
                errors.append(
                    f"{variant}/seed_{seed}: checkpoint step {checkpoint_step} < {args.minimum_steps}"
                )
            if nonfinite_tensors:
                errors.append(
                    f"{variant}/seed_{seed}: non-finite checkpoint tensors {nonfinite_tensors}"
                )
            record.update(
                final_step=final_step,
                checkpoint_step=checkpoint_step,
                metric_rows=len(rows),
                nonfinite=nonfinite,
                nonfinite_checkpoint_tensors=nonfinite_tensors,
            )
            runs.append(record)
    payload = {
        "status": "passed" if not errors else "failed",
        "minimum_steps": args.minimum_steps,
        "expected_runs": len(raw["core_variants"]) * len(raw["seeds"]),
        "errors": errors,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
