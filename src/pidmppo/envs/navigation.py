from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from ..config import EnvConfig
from .backends import NavigationBackend, create_backend
from .maps import GridMap, generate_corridor_map, load_map
from .geometry import ClearanceSpace


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class MaplessNavigationEnv(gym.Env[np.ndarray, np.ndarray]):
    """Deterministic 2-D backend for the paper's mapless policy interface.

    The simulator owns an occupancy map for collision and range sensing. The policy
    never receives that map: observations contain only lidar, relative goal and
    action-history values.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.map: GridMap = load_map(self.config.map_name, map_file=self.config.map_file)
        self.backend: NavigationBackend = create_backend(self.map, self.config)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low = np.concatenate(
            (
                np.zeros(self.config.lidar_beams, dtype=np.float32),
                np.asarray([0.0, -1.0, -1.0], dtype=np.float32),
                np.tile(
                    np.asarray(
                        [0.0 if not self.config.allow_reverse else -1.0, -1.0],
                        dtype=np.float32,
                    ),
                    self.config.action_history,
                ),
            )
        )
        high = np.ones(self.config.observation_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._history = deque(maxlen=self.config.action_history)
        self._progress = deque(maxlen=self.config.stagnation_window)
        self._step_count = 0
        self._previous_distance = 0.0
        self._space = ClearanceSpace(self.map, self.config.spawn_clearance)
        self.training_step = 0

    def set_training_step(self, step: int) -> None:
        """Curriculum changes take effect only at the next episode reset."""
        self.training_step = int(step)

    def curriculum_parameters(self) -> tuple[int, float, float]:
        if self.config.curriculum and self.training_step < 50_000:
            return 2, 1.0, 2.5
        if self.config.curriculum and self.training_step < 100_000:
            return 4, 2.0, 4.0
        return self.config.randomized_obstacle_segments, max(1.5, .35 * self.map.diagonal), float("inf")

    @property
    def pose(self) -> np.ndarray:
        return self.backend.pose

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        randomized_map = (
            self.config.randomize_obstacles
            and self.config.map_name == "map1"
            and self.config.map_file is None
        )
        for attempt in range(100):
            if randomized_map:
                map_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
                self.map = generate_corridor_map(
                    width=self.config.randomized_map_width,
                    height=self.config.randomized_map_height,
                    obstacle_segments=self.curriculum_parameters()[0],
                    seed=map_seed,
                    cell_size=self.config.randomized_cell_size,
                )
                self.backend.update_map(self.map)
                self._space = ClearanceSpace(self.map, self.config.spawn_clearance)
            try:
                start, goal = self._choose_start_goal(options)
                break
            except ValueError:
                # Explicit requests are never silently repaired. Random geometry
                # can be cell-connected yet lose wide enough long paths after
                # inflation; discard that map and sample another from the RNG.
                if not randomized_map or not self.config.randomize_start_goal or "start" in options or "goal" in options:
                    raise
        else:
            raise RuntimeError("No clearance-valid randomized task after 100 maps")
        theta = float(options.get("theta", self.np_random.uniform(-np.pi, np.pi)))
        self.backend.reset(start, theta)
        self._goal[:] = goal
        self._history.clear()
        self._history.extend(np.zeros(2, dtype=np.float32) for _ in range(self.config.action_history))
        self._progress.clear()
        self._step_count = 0
        self._previous_distance = self._distance_to_goal()
        observation = self._observation()
        return observation, self._info(success=False, collision=False)

    def _choose_start_goal(self, options: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if ("start" in options) != ("goal" in options):
            raise ValueError("start and goal must be supplied together")
        if "start" in options and "goal" in options:
            start = np.asarray(options["start"], dtype=np.float32)
            goal = np.asarray(options["goal"], dtype=np.float32)
            if start.shape != (2,) or goal.shape != (2,):
                raise ValueError("start and goal must each contain two coordinates")
            if not self._space.is_valid(start) or not self._space.is_valid(goal):
                raise ValueError("start/goal violates robot, collision-threshold or spawn clearance")
            if not self._space.connected(start, goal):
                raise ValueError("start and goal are disconnected in robot-clearance free space")
            self._space.path(start, goal)
            return start, goal
        if self.config.randomize_start_goal:
            free = self._space.centres
            _, minimum, maximum = self.curriculum_parameters()
            for _ in range(2000):
                if len(free) < 2:
                    break
                indices = self.np_random.choice(len(free), size=2, replace=False)
                start, goal = free[indices]
                if minimum <= float(np.linalg.norm(goal - start)) <= maximum and self._space.connected(start, goal):
                    return start.copy(), goal.copy()
            raise ValueError("Cannot sample a connected start/goal with the required clearance and distance")
        start, goal = np.asarray(self.map.starts[0], dtype=np.float32), np.asarray(self.map.goals[0], dtype=np.float32)
        if not self._space.connected(start, goal):
            raise ValueError("Default map endpoints violate clearance/connectivity; supply valid scenarios")
        return start, goal

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        normalized_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.config.allow_reverse:
            linear = float(normalized_action[0]) * self.config.max_linear_velocity
        else:
            linear = float((normalized_action[0] + 1.0) / 2.0) * self.config.max_linear_velocity
        angular = float(normalized_action[1]) * self.config.max_angular_velocity
        collision = self.backend.step(linear, angular, self.config.control_period)
        lidar = self._lidar()
        collision = collision or bool(lidar.min() < self.config.collision_distance)

        self._history.append(np.asarray([linear, angular], dtype=np.float32))
        self._step_count += 1
        distance = self._distance_to_goal()
        progress = self._previous_distance - distance
        self._progress.append(progress)
        success = not collision and distance <= self.config.goal_radius
        terminated = collision or success
        truncated = self._step_count >= self.config.max_episode_steps and not terminated

        reward_terms = {
            "progress": self.config.progress_scale * progress,
            "step": self.config.step_penalty,
            "angular": -self.config.angular_penalty * abs(angular),
            "stagnation": 0.0,
            "terminal": 0.0,
        }
        if len(self._progress) == self.config.stagnation_window and float(np.mean(self._progress)) < self.config.stagnation_epsilon:
            reward_terms["stagnation"] = self.config.stagnation_penalty
        if success:
            reward_terms["terminal"] = self.config.terminal_reward
        elif collision:
            reward_terms["terminal"] = -self.config.terminal_reward
        reward = float(sum(reward_terms.values()))
        self._previous_distance = distance
        info = self._info(success=success, collision=collision)
        info["executed_velocity"] = np.asarray([linear, angular], dtype=np.float32)
        info["reward_terms"] = reward_terms
        return self._observation(lidar), reward, terminated, truncated, info

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self._goal - self.pose[:2]))

    def _lidar(self) -> np.ndarray:
        return self.backend.lidar(
            self.config.lidar_beams,
            self.config.lidar_min_range,
            self.config.lidar_max_range,
        )

    def _observation(self, lidar: np.ndarray | None = None) -> np.ndarray:
        lidar = self._lidar() if lidar is None else lidar
        lidar = (lidar - self.config.lidar_min_range) / (
            self.config.lidar_max_range - self.config.lidar_min_range
        )
        pose = self.pose
        goal_delta = self._goal - pose[:2]
        distance = float(np.linalg.norm(goal_delta))
        bearing = _wrap_angle(float(np.arctan2(goal_delta[1], goal_delta[0]) - pose[2]))
        goal_features = np.asarray(
            [min(distance / self.map.diagonal, 1.0), np.cos(bearing), np.sin(bearing)],
            dtype=np.float32,
        )
        history = np.asarray(self._history, dtype=np.float32).copy()
        history[:, 0] /= self.config.max_linear_velocity
        history[:, 1] /= self.config.max_angular_velocity
        history = history.reshape(-1)
        return np.concatenate((lidar.astype(np.float32), goal_features, history)).astype(np.float32)

    def _info(self, *, success: bool, collision: bool) -> dict[str, Any]:
        return {
            "success": success,
            "collision": collision,
            "distance_to_goal": self._distance_to_goal(),
            "step_count": self._step_count,
            "pose": self.pose,
            "goal": self.goal,
        }

    def close(self) -> None:
        self.backend.close()
