"""Fail-closed acceptance before dispatching the bounded optimization round."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from pidmppo.config import config_from_dict
from pidmppo.evaluation.optimization import training_health
from pidmppo.models import ActorCritic
from pidmppo.utils import seed_everything
from pidmppo.utils.manifest import source_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/preflight_v2"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/correctness_v2_preflight.json"))
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Preflight output/report already exists; use a fresh directory")
    args.output.mkdir(parents=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(Path("configs/optimization_v2.yaml").read_text(encoding="utf-8"))
    report = {"status": "running", "version": spec["version"], "source_fingerprint": source_fingerprint(), "smoke": {}, "legacy_compatibility": {}}
    def save():
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    def run(command, log_name):
        with (args.output / log_name).open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    save()
    try:
        run([sys.executable, "-m", "pytest", "-q"], "tests.log")
        run([sys.executable, "scripts/audit_scenarios_v2.py", "--output", str(args.output / "scene_audit.json")], "scene_audit.log")
        report["asset_audit"] = str(args.output / "scene_audit.json")
        report["manifest_hashes"] = json.loads((args.output / "scene_audit.json").read_text())["manifests"]
        for name, overrides in spec["candidates"].items():
            output = args.output / name
            command = [sys.executable, "scripts/train.py", "--config", spec["base_config"], "--seed", "11", "--steps", "4096",
                       "--set", "ppo.gradient_diagnostics_interval=1", "--output", str(output)]
            for override in overrides:
                command.extend(("--set", override))
            run(command, f"smoke_{name}.log")
            report["smoke"][name] = training_health(output / "training_metrics.csv")
            save()
        # Existing state_dicts retain the same parameter keys/shapes. Verify all
        # five old final checkpoints load and fixed-policy latent ratios are 1.
        seed_everything(11)
        for seed in spec["legacy_seeds"]:
            path = Path(spec["legacy_root"]) / f"seed_{seed}" / "checkpoints/final.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            config = config_from_dict(checkpoint["config"])
            model = ActorCritic(config.env.observation_dim, 2, config.model)
            model.load_state_dict(checkpoint["model"])
            with torch.no_grad():
                output = model(torch.zeros(1, 3, config.env.observation_dim))
                action, old, raw = model.sample_action(output.mean, output.log_std, return_raw=True)
                new, entropy = model.evaluate_action(output.mean, output.log_std, action, raw_action=raw)
                error = float((new - old).abs().max())
                assert error <= 1e-4 and torch.isfinite(action).all() and torch.isfinite(entropy).all()
            report["legacy_compatibility"][str(seed)] = {"loaded": True, "max_log_prob_error": error,
                  "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            save()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
