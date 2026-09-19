from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from pidmppo.algorithms import PPOTrainer
from pidmppo.config import apply_overrides, load_config
from pidmppo.envs import MaplessNavigationEnv
from pidmppo.experiment import VARIANTS, apply_variant
from pidmppo.utils import seed_everything, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PIDM-PPO or a controlled ablation")
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="pidmppo")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path, default=Path("runs/default"))
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    parser.add_argument("--deadline-unix", type=float, help="Absolute wall-clock deadline; checkpoint at a safe boundary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.output / "run_manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output}")
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    config = apply_overrides(apply_variant(load_config(args.config), args.variant), args.set)
    if args.seed is not None:
        config.seed = args.seed
    if args.steps is not None:
        config.total_timesteps = args.steps
    seed_everything(config.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_run_manifest(args.output / "run_manifest.json", config, args.variant)
    env = MaplessNavigationEnv(config.env)
    trainer = None
    status = {"status": "failed"}
    try:
        trainer = PPOTrainer(env, config)
        status = trainer.train(
            checkpoint_dir=args.output / "checkpoints",
            metrics_path=args.output / "training_metrics.csv",
            stop_requested=lambda: stop["requested"] or (args.deadline_unix is not None and time.time() >= args.deadline_unix),
        )
    except Exception as error:
        status = {"status": "failed", "error": repr(error), "global_step": trainer.global_step if trainer else 0}
        if trainer is not None:
            trainer.save(args.output / "checkpoints" / "failed.pt")
        raise
    finally:
        env.close()
        temporary = args.output / "train_status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(args.output / "train_status.json")


if __name__ == "__main__":
    main()
