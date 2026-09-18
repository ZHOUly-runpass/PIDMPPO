from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig
from ..envs import MaplessNavigationEnv
from ..models import ActorCritic
from .trace import TrajectoryRecorder


@dataclass(frozen=True)
class Scenario:
    id: str
    map: str
    start: tuple[float, float]
    goal: tuple[float, float]
    theta: float
    difficulty: str = "unclassified"
    map_file: str | None = None
    trap_region: tuple[float, float, float, float] | None = None


@dataclass
class EpisodeResult:
    episode_id: str
    method: str
    training_seed: int
    evaluation_seed: int
    scenario_id: str
    map: str
    difficulty: str
    success: bool
    collision: bool
    timeout: bool
    steps: int
    episode_return: float
    path_length: float
    elapsed_seconds: float


def load_scenarios(path: str | Path) -> list[Scenario]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError("Unsupported scenario manifest version")
    scenarios = []
    for item in raw["scenarios"]:
        scenarios.append(
            Scenario(
                id=item["id"],
                map=item["map"],
                start=tuple(item["start"]),
                goal=tuple(item["goal"]),
                theta=float(item["theta"]),
                difficulty=item.get("difficulty", "unclassified"),
                map_file=item.get("map_file"),
                trap_region=tuple(item["trap_region"]) if item.get("trap_region") else None,
            )
        )
    if len({scenario.id for scenario in scenarios}) != len(scenarios):
        raise ValueError("Scenario ids must be unique")
    return scenarios


def evaluate_policy(
    model: ActorCritic,
    config: ExperimentConfig,
    scenarios: list[Scenario],
    *,
    method: str,
    training_seed: int,
    episodes_per_map: int,
    evaluation_seed: int,
    device: torch.device,
    trace_episodes: int = 0,
) -> tuple[list[EpisodeResult], TrajectoryRecorder]:
    model.eval()
    results: list[EpisodeResult] = []
    recorder = TrajectoryRecorder(config.ppo.gamma)
    maps = sorted({scenario.map for scenario in scenarios})
    traced = 0
    for map_name in maps:
        map_scenarios = [scenario for scenario in scenarios if scenario.map == map_name]
        if not map_scenarios:
            continue
        env_config = deepcopy(config.env)
        env_config.map_name = map_name
        env_config.map_file = map_scenarios[0].map_file
        env_config.randomize_start_goal = False
        env_config.randomize_obstacles = False
        env = MaplessNavigationEnv(env_config)
        try:
            for episode_index in range(episodes_per_map):
                scenario = map_scenarios[episode_index % len(map_scenarios)]
                seed = evaluation_seed + episode_index
                observation, _ = env.reset(
                    seed=seed,
                    options={"start": scenario.start, "goal": scenario.goal, "theta": scenario.theta},
                )
                state = model.initial_state(1, device)
                previous_pose = env.pose
                path_length = 0.0
                episode_return = 0.0
                episode_id = f"{training_seed}:{scenario.id}:{episode_index}"
                should_trace = traced < trace_episodes
                final_info = {"success": False, "collision": False}
                for step in range(env_config.max_episode_steps):
                    observation_tensor = torch.as_tensor(observation, device=device).view(1, 1, -1)
                    with torch.no_grad():
                        output = model(
                            observation_tensor,
                            state,
                            torch.tensor([[step == 0]], device=device),
                        )
                        action = torch.tanh(output.mean)
                        _, entropy = model.evaluate_action(output.mean, output.log_std, action)
                    numpy_action = action[0, 0].cpu().numpy()
                    observation, reward, terminated, truncated, final_info = env.step(numpy_action)
                    current_pose = env.pose
                    path_length += float(np.linalg.norm(current_pose[:2] - previous_pose[:2]))
                    previous_pose = current_pose
                    episode_return += reward
                    if should_trace:
                        recorder.append(
                            episode_id=episode_id,
                            step=step,
                            pose=current_pose,
                            goal=env.goal,
                            action=numpy_action,
                            reward=reward,
                            distance=float(final_info["distance_to_goal"]),
                            value1=float(output.value1.item()),
                            value2=float(output.value2.item()),
                            entropy=float(entropy.item()),
                            diagnostics=output.diagnostics,
                            event=_event_label(final_info, env_config, scenario, current_pose),
                            method=method,
                            training_seed=training_seed,
                            scenario_id=scenario.id,
                            map_name=map_name,
                            difficulty=scenario.difficulty,
                        )
                    state = output.state
                    if terminated or truncated:
                        break
                if should_trace:
                    recorder.finalize_episode(episode_id)
                    traced += 1
                results.append(
                    EpisodeResult(
                        episode_id=episode_id,
                        method=method,
                        training_seed=training_seed,
                        evaluation_seed=seed,
                        scenario_id=scenario.id,
                        map=map_name,
                        difficulty=scenario.difficulty,
                        success=bool(final_info.get("success", False)),
                        collision=bool(final_info.get("collision", False)),
                        timeout=not bool(final_info.get("success", False)) and not bool(final_info.get("collision", False)),
                        steps=step + 1,
                        episode_return=episode_return,
                        path_length=path_length,
                        elapsed_seconds=(step + 1) * env_config.control_period,
                    )
                )
        finally:
            env.close()
    return results, recorder


def _event_label(info: dict, config, scenario: Scenario, pose: np.ndarray) -> str:
    if info.get("success"):
        return "goal"
    if info.get("collision"):
        return "collision"
    if scenario.trap_region is not None:
        x_min, y_min, x_max, y_max = scenario.trap_region
        if x_min <= pose[0] <= x_max and y_min <= pose[1] <= y_max:
            return "trap"
    stagnation = info.get("reward_terms", {}).get("stagnation", 0.0)
    if stagnation < 0.0:
        return "stagnation"
    return "normal"


def write_episode_results(results: list[EpisodeResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EpisodeResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
