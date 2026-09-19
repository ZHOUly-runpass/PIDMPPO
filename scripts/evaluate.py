from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch

from pidmppo.algorithms import PPOTrainer
from pidmppo.config import config_from_dict, load_config
from pidmppo.envs import MaplessNavigationEnv
from pidmppo.evaluation import evaluate_policy, load_scenarios, write_episode_results
from pidmppo.experiment import VARIANTS, apply_variant
from pidmppo.utils import seed_everything
from pidmppo.models import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one fixed checkpoint without retraining")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="Explicit override; by default use checkpoint's own configuration")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="pidmppo")
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--episodes-per-map", type=int)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--trace-episodes", type=int, default=0)
    parser.add_argument("--trace-outcomes", action="store_true", help="Save one representative success/collision/timeout trajectory if present")
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.output / "episodes.csv").exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {args.output}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = apply_variant(load_config(args.config), args.variant) if args.config else config_from_dict(checkpoint["config"])
    seed_everything(args.evaluation_seed)
    scenario_path = args.scenarios or Path(config.evaluation.scenario_manifest)
    scenarios = load_scenarios(scenario_path)
    device = PPOTrainer._resolve_device(config.device)
    model = ActorCritic(config.env.observation_dim, 2, config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    results, recorder = evaluate_policy(
            model,
            config,
            scenarios,
            method=args.variant,
            training_seed=args.training_seed,
            episodes_per_map=args.episodes_per_map,
            evaluation_seed=args.evaluation_seed,
            device=device,
            trace_episodes=args.trace_episodes,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_episode_results(results, args.output / "episodes.csv")
    if recorder.rows:
        recorder.write_csv(args.output / "traces.csv")
        recorder.write_json(args.output / "traces.json")
    if args.trace_outcomes:
        selected = []
        for outcome in ("success", "collision", "timeout"):
            result = next((result for result in results if getattr(result, outcome)), None)
            if result:
                selected.append(next(s for s in scenarios if s.id == result.scenario_id))
        if selected:
            _, outcome_recorder = evaluate_policy(model, config, selected, method=args.variant, training_seed=args.training_seed,
                episodes_per_map=None, evaluation_seed=args.evaluation_seed, device=device, trace_episodes=len(selected))
            outcome_recorder.write_csv(args.output / "outcome_traces.csv")
            outcome_recorder.write_json(args.output / "outcome_traces.json")
    successful = [result for result in results if result.success]
    def group_summary(rows):
        return {"episodes": len(rows), "success_rate": sum(r.success for r in rows) / max(1, len(rows)),
                "collision_rate": sum(r.collision for r in rows) / max(1, len(rows)),
                "timeout_rate": sum(r.timeout for r in rows) / max(1, len(rows))}
    summary = {"episodes": len(results), "unique_tasks": len({(r.map, r.scenario_id) for r in results}),
               "success_rate": sum(r.success for r in results) / len(results),
               "collision_rate": sum(r.collision for r in results) / len(results),
               "timeout_rate": sum(r.timeout for r in results) / len(results),
               "successful_path_efficiency": sum(r.path_efficiency for r in successful) / max(1, len(successful)),
               "spin_fraction": sum(r.spin_fraction for r in results) / len(results),
               "stagnation_fraction": sum(r.stagnation_fraction for r in results) / len(results),
               "backend": config.env.backend, "checkpoint": str(args.checkpoint),
               "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "scenario_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
               "scenario_split": json.loads(scenario_path.read_text(encoding="utf-8")).get("split", "legacy"),
               "training_seed": args.training_seed, "implementation_version": checkpoint.get("implementation_version", "legacy-v1"),
               "by_map": {name: group_summary([r for r in results if r.map == name]) for name in sorted({r.map for r in results})}}
    if any(r.map in {"map2", "map3", "map4"} for r in results):
        summary["unseen_maps_2_to_4"] = group_summary([r for r in results if r.map in {"map2", "map3", "map4"}])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    successes = sum(result.success for result in results)
    print(f"episodes={len(results)} successes={successes} output={args.output}")


if __name__ == "__main__":
    main()
