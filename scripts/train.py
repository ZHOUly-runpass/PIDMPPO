from __future__ import annotations

import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(apply_variant(load_config(args.config), args.variant), args.set)
    if args.seed is not None:
        config.seed = args.seed
    if args.steps is not None:
        config.total_timesteps = args.steps
    seed_everything(config.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_run_manifest(args.output / "run_manifest.json", config, args.variant)
    env = MaplessNavigationEnv(config.env)
    try:
        trainer = PPOTrainer(env, config)
        trainer.train(
            checkpoint_dir=args.output / "checkpoints",
            metrics_path=args.output / "training_metrics.csv",
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
