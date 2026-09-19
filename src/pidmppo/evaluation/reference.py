"""Privileged asset-acceptance controller, NEVER an input to policy training."""
from __future__ import annotations

import numpy as np

from ..envs.navigation import MaplessNavigationEnv, _wrap_angle
from ..envs.geometry import path_length


def validate_reference(env: MaplessNavigationEnv, start, goal, theta: float, max_steps: int = 540) -> dict:
    path = env._space.path(start, goal)
    options = {"start": start, "goal": goal, "theta": theta}
    env.reset(options=options)
    # [-1, 0] is the zero-speed action in the forward-only convention.
    zero = np.array([0. if env.config.allow_reverse else -1., 0.], dtype=np.float32)
    _, _, terminated, _, info = env.step(zero)
    if terminated or info["collision"]:
        raise ValueError("Scene fails start zero-action test")
    env.reset(options=options)
    index, actual_length = 1, 0.
    previous = env.pose[:2]
    for step in range(max_steps):
        pose = env.pose
        while index < len(path) - 1 and np.linalg.norm(path[index] - pose[:2]) < 1e-4:
            index += 1
        delta = path[index] - pose[:2]
        distance = float(np.linalg.norm(delta))
        angle = _wrap_angle(float(np.arctan2(delta[1], delta[0]) - pose[2]))
        # Rotate in place then follow each straight segment exactly. This avoids
        # cutting tight corners; it respects the SAME v/w/dt limits as the agent.
        angular = float(np.clip(angle / env.config.control_period, -env.config.max_angular_velocity, env.config.max_angular_velocity))
        linear = 0. if abs(angle) > 1e-3 else min(env.config.max_linear_velocity, distance / env.config.control_period)
        v = linear / env.config.max_linear_velocity
        action = np.array([v if env.config.allow_reverse else 2 * v - 1, angular / env.config.max_angular_velocity], np.float32)
        _, _, terminated, truncated, info = env.step(action)
        actual_length += float(np.linalg.norm(env.pose[:2] - previous))
        previous = env.pose[:2]
        if terminated or truncated:
            if not info["success"]:
                raise ValueError("Reference controller failed (collision or timeout)")
            return {"reference_path_length": path_length(path), "reference_steps": step + 1,
                    "reference_executed_length": actual_length, "zero_action_passed": True,
                    "clearance": env.config.spawn_clearance, "connected": True,
                    "controller": "rotate-then-translate-v1", "backend": env.config.backend}
    raise ValueError(f"Reference controller exceeds {max_steps} steps")
