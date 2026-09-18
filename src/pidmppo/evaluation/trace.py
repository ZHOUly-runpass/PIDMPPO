from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class TrajectoryRecorder:
    """Collects reviewer-requested state, PIDM and value diagnostics."""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.rows: list[dict[str, Any]] = []

    def append(
        self,
        *,
        episode_id: str,
        step: int,
        pose: np.ndarray,
        goal: np.ndarray,
        action: np.ndarray,
        reward: float,
        distance: float,
        value1: float,
        value2: float,
        entropy: float,
        diagnostics: dict[str, torch.Tensor],
        event: str = "normal",
        method: str = "",
        training_seed: int = -1,
        scenario_id: str = "",
        map_name: str = "",
        difficulty: str = "",
    ) -> None:
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "step": step,
            "x": float(pose[0]),
            "y": float(pose[1]),
            "theta": float(pose[2]),
            "goal_x": float(goal[0]),
            "goal_y": float(goal[1]),
            "distance_to_goal": distance,
            "linear_action": float(action[0]),
            "angular_action": float(action[1]),
            "reward": reward,
            "value1": value1,
            "value2": value2,
            "value_min": min(value1, value2),
            "value_head_gap": value1 - value2,
            "entropy": entropy,
            "event": event,
            "method": method,
            "training_seed": training_seed,
            "scenario_id": scenario_id,
            "map": map_name,
            "difficulty": difficulty,
        }
        if diagnostics:
            row.update(self._diagnostic_scalars(diagnostics))
        self.rows.append(row)

    @staticmethod
    def _diagnostic_scalars(diagnostics: dict[str, torch.Tensor]) -> dict[str, float]:
        result: dict[str, float] = {}
        for branch in ("p", "i", "d"):
            value = diagnostics.get(branch)
            if value is not None:
                result[f"{branch}_norm"] = float(torch.linalg.vector_norm(value[0, 0]).detach().cpu())
        memory_norm = diagnostics.get("memory_norm")
        if memory_norm is not None:
            result["memory_norm"] = float(memory_norm[0, 0].detach().cpu())
        gate = diagnostics.get("gate")
        if gate is not None:
            gate_vector = gate[0, 0].detach().cpu().float()
            result.update(
                gate_mean=float(gate_vector.mean()),
                gate_q25=float(torch.quantile(gate_vector, 0.25)),
                gate_q75=float(torch.quantile(gate_vector, 0.75)),
            )
        attention = diagnostics.get("attention")
        if attention is not None:
            weights = attention[0, 0].detach().cpu()
            result.update(attention_p=float(weights[0]), attention_i=float(weights[1]), attention_d=float(weights[2]))
        return result

    def finalize_episode(self, episode_id: str) -> None:
        episode_rows = [row for row in self.rows if row["episode_id"] == episode_id]
        return_value = 0.0
        for row in reversed(episode_rows):
            return_value = float(row["reward"]) + self.gamma * return_value
            row["mc_return"] = return_value
            row["value_min_error"] = float(row["value_min"]) - return_value
            row["value1_error"] = float(row["value1"]) - return_value
            row["value2_error"] = float(row["value2"]) - return_value

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in self.rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.rows, indent=2, ensure_ascii=False), encoding="utf-8")
