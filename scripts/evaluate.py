from __future__ import annotations

import argparse
from pathlib import Path

from pidmppo.algorithms import PPOTrainer
from pidmppo.config import load_config
from pidmppo.envs import MaplessNavigationEnv
from pidmppo.evaluation import evaluate_policy, load_scenarios, write_episode_results
from pidmppo.experiment import VARIANTS, apply_variant
from pidmppo.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one fixed checkpoint without retraining")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="pidmppo")
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--episodes-per-map", type=int)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--trace-episodes", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_variant(load_config(args.config), args.variant)
    seed_everything(args.evaluation_seed)
    scenario_path = args.scenarios or Path(config.evaluation.scenario_manifest)
    scenarios = load_scenarios(scenario_path)
    training_env = MaplessNavigationEnv(config.env)
    try:
        trainer = PPOTrainer(training_env, config)
        trainer.load(args.checkpoint, load_optimizer=False)
        results, recorder = evaluate_policy(
            trainer.model,
            config,
            scenarios,
            method=args.variant,
            training_seed=args.training_seed,
            episodes_per_map=args.episodes_per_map or config.evaluation.episodes_per_map,
            evaluation_seed=args.evaluation_seed,
            device=trainer.device,
            trace_episodes=args.trace_episodes,
        )
    finally:
        training_env.close()
    args.output.mkdir(parents=True, exist_ok=True)
    write_episode_results(results, args.output / "episodes.csv")
    if recorder.rows:
        recorder.write_csv(args.output / "traces.csv")
        recorder.write_json(args.output / "traces.json")
    successes = sum(result.success for result in results)
    print(f"episodes={len(results)} successes={successes} output={args.output}")


if __name__ == "__main__":
    main()
