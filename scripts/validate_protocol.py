from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pidmppo.config import load_config
from pidmppo.envs import MaplessNavigationEnv, load_map
from pidmppo.evaluation import load_scenarios
from pidmppo.experiment import VARIANTS, apply_variant
from pidmppo.models import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate every configured experiment without training")
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    args = parser.parse_args()
    base = load_config(args.config)
    scenarios = load_scenarios(base.evaluation.scenario_manifest)
    for scenario in scenarios:
        grid = load_map(scenario.map)
        if grid.is_occupied(*scenario.start) or grid.is_occupied(*scenario.goal):
            raise ValueError(f"Scenario {scenario.id} starts or ends inside an obstacle")
    observation = torch.zeros(2, 3, base.env.observation_dim)
    for name in VARIANTS:
        config = apply_variant(base, name)
        model = ActorCritic(config.env.observation_dim, 2, config.model)
        output = model(observation)
        if output.mean.shape != (2, 3, 2):
            raise AssertionError(f"Invalid policy output for {name}")
    env = MaplessNavigationEnv(base.env)
    try:
        obs, _ = env.reset(seed=base.seed)
        if obs.shape != (base.env.observation_dim,):
            raise AssertionError("Environment observation shape mismatch")
    finally:
        env.close()
    print(f"variants={len(VARIANTS)} scenarios={len(scenarios)} protocol=valid")


if __name__ == "__main__":
    main()
