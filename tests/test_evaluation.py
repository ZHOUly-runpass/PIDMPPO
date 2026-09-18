import json

import numpy as np
import torch

from pidmppo.evaluation import (
    aggregate_results,
    audit_evaluation_rows,
    bootstrap_interval,
    load_scenarios,
    wilson_interval,
)
from pidmppo.evaluation.protocol import Scenario, evaluate_policy
from pidmppo.evaluation.trace import TrajectoryRecorder
from pidmppo.config import ExperimentConfig
from pidmppo.envs import load_map
from pidmppo.models import ActorCritic


def test_formal_manifest_has_fixed_scenarios_for_every_map():
    scenarios = load_scenarios("configs/scenarios.json")
    counts = {name: sum(item.map == name for item in scenarios) for name in {s.map for s in scenarios}}
    assert counts == {"map1": 25, "map2": 25, "map3": 25, "map4": 25}


def test_mechanism_manifest_defines_fixed_u_trap():
    scenarios = load_scenarios("configs/mechanism_scenarios.json")
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.id == "u_trap_escape"
    assert scenario.map_file == "configs/maps/mechanism_u.txt"
    assert scenario.trap_region is not None


def test_intervals_are_well_ordered():
    low, high = wilson_interval(80, 100)
    assert 0.0 < low < 0.8 < high < 1.0
    low, high = bootstrap_interval([1.0, 2.0, 3.0], samples=1000, seed=1)
    assert low <= 2.0 <= high


def test_aggregation_reports_seed_level_uncertainty():
    rows = []
    for seed, outcomes in ((1, (True, True)), (2, (False, True))):
        for outcome in outcomes:
            rows.append(
                {
                    "method": "pidmppo",
                    "map": "map1",
                    "training_seed": seed,
                    "success": outcome,
                    "steps": 10,
                    "episode_return": 1.0,
                    "path_length": 2.0,
                    "elapsed_seconds": 1.0,
                }
            )
    summary = aggregate_results(rows, bootstrap_samples=100, seed=1)[0]
    assert summary["training_seeds"] == 2
    assert summary["success_rate"] == 0.75
    assert "success_seed_ci_low" in summary
    assert summary["collision_rate"] == 0.0
    assert summary["timeout_rate"] == 0.0


def test_protocol_audit_rejects_missing_seeds():
    rows = [
        {
            "method": "pidmppo",
            "map": "map1",
            "training_seed": 1,
            "evaluation_seed": index,
            "scenario_id": f"scenario_{index}",
        }
        for index in range(2)
    ]
    errors = audit_evaluation_rows(rows, required_seeds=2, required_episodes_per_map_seed=2)
    assert any("requires 2" in error for error in errors)


def test_trajectory_recorder_computes_returns_and_pidm_scalars():
    recorder = TrajectoryRecorder(gamma=0.5)
    diagnostics = {
        "p": torch.ones(1, 1, 3),
        "i": torch.ones(1, 1, 3),
        "d": torch.ones(1, 1, 3),
        "memory_norm": torch.tensor([[2.0]]),
        "gate": torch.tensor([[[0.2, 0.5, 0.8]]]),
        "attention": torch.tensor([[[0.2, 0.3, 0.5]]]),
    }
    for step, reward in enumerate((1.0, 2.0)):
        recorder.append(
            episode_id="episode",
            step=step,
            pose=np.zeros(3),
            goal=np.ones(2),
            action=np.zeros(2),
            reward=reward,
            distance=1.0,
            value1=0.2,
            value2=0.1,
            entropy=0.5,
            diagnostics=diagnostics,
        )
    recorder.finalize_episode("episode")
    assert recorder.rows[0]["mc_return"] == 2.0
    assert recorder.rows[0]["attention_d"] == 0.5


def test_evaluation_protocol_runs_without_training():
    config = ExperimentConfig()
    config.model.hidden_size = 8
    config.model.actor_hidden_size = 8
    config.env.max_episode_steps = 2
    model = ActorCritic(85, 2, config.model)
    grid = load_map("map1")
    scenario = Scenario(
        id="smoke",
        map="map1",
        start=grid.starts[0],
        goal=grid.goals[0],
        theta=0.0,
    )
    results, recorder = evaluate_policy(
        model,
        config,
        [scenario],
        method="untrained_smoke",
        training_seed=0,
        episodes_per_map=1,
        evaluation_seed=1,
        device=torch.device("cpu"),
        trace_episodes=1,
    )
    assert len(results) == 1
    assert results[0].timeout
    assert len(recorder.rows) == 2
