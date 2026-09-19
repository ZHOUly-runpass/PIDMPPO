from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import EnvConfig
from .maps import GridMap


class NavigationBackend(Protocol):
    @property
    def pose(self) -> np.ndarray: ...

    def reset(self, start: np.ndarray, theta: float) -> None: ...

    def update_map(self, grid_map: GridMap) -> None: ...

    def step(self, linear: float, angular: float, period: float) -> bool: ...

    def lidar(self, beams: int, minimum: float, maximum: float) -> np.ndarray: ...

    def close(self) -> None: ...


class GridBackend:
    def __init__(self, grid_map: GridMap, robot_radius: float):
        self.map = grid_map
        self.robot_radius = robot_radius
        self._pose = np.zeros(3, dtype=np.float32)

    @property
    def pose(self) -> np.ndarray:
        return self._pose.copy()

    def reset(self, start: np.ndarray, theta: float) -> None:
        self._pose[:] = (start[0], start[1], theta)

    def update_map(self, grid_map: GridMap) -> None:
        self.map = grid_map

    def step(self, linear: float, angular: float, period: float) -> bool:
        theta = float((self._pose[2] + angular * period + np.pi) % (2.0 * np.pi) - np.pi)
        candidate = self._pose.copy()
        candidate[0] += linear * np.cos(theta) * period
        candidate[1] += linear * np.sin(theta) * period
        candidate[2] = theta
        collision = self._circle_collision(float(candidate[0]), float(candidate[1]))
        if not collision:
            self._pose[:] = candidate
        return collision

    def _circle_collision(self, x: float, y: float) -> bool:
        r = self.robot_radius
        if min(x, y, self.map.width - x, self.map.height - y) <= r:
            return True
        size = self.map.cell_size
        for row in range(int((y - r) // size), int((y + r) // size) + 1):
            for col in range(int((x - r) // size), int((x + r) // size) + 1):
                if self.map.occupied[row, col]:
                    dx = max(col * size - x, 0., x - (col + 1) * size)
                    dy = max(row * size - y, 0., y - (row + 1) * size)
                    if dx * dx + dy * dy <= r * r:
                        return True
        return False

    def lidar(self, beams: int, minimum: float, maximum: float) -> np.ndarray:
        angles = self._pose[2] + np.linspace(-np.pi, np.pi, beams, endpoint=False)
        increments = np.arange(minimum, maximum + 0.01, 0.02, dtype=np.float32)
        x = self._pose[0] + np.cos(angles)[:, None] * increments[None, :]
        y = self._pose[1] + np.sin(angles)[:, None] * increments[None, :]
        columns = np.floor(x / self.map.cell_size).astype(np.int64)
        rows = np.floor(y / self.map.cell_size).astype(np.int64)
        valid = (
            (rows >= 0)
            & (columns >= 0)
            & (rows < self.map.occupied.shape[0])
            & (columns < self.map.occupied.shape[1])
        )
        hits = np.ones(rows.shape, dtype=bool)
        hits[valid] = self.map.occupied[rows[valid], columns[valid]]
        has_hit = np.any(hits, axis=1)
        first_hit = np.argmax(hits, axis=1)
        distances = np.where(has_hit, increments[first_hit], maximum).astype(np.float32)
        return np.clip(distances, minimum, maximum)

    def close(self) -> None:
        return None


class PyBulletBackend:
    """Optional PyBullet backend with either URDF wheel control or kinematics.

    A formal reproduction should set ``robot_urdf`` and wheel joint indices.
    Without those values a collision-cylinder robot is used only for integration
    testing; it must not be reported as TurtleBot3 evidence.
    """

    def __init__(self, grid_map: GridMap, config: EnvConfig):
        try:
            import pybullet as bullet
            import pybullet_data
        except ImportError as error:
            raise ImportError(
                "PyBullet backend requested. Install the project with "
                "`pip install -e .[simulation]`."
            ) from error
        self.bullet = bullet
        self.pybullet_data = pybullet_data
        self.map = grid_map
        self.config = config
        mode = bullet.GUI if config.pybullet_gui else bullet.DIRECT
        self.client = bullet.connect(mode)
        self.robot_id = -1
        self.plane_id = -1
        self.obstacle_ids: list[int] = []
        self._pose = np.zeros(3, dtype=np.float32)

    @property
    def pose(self) -> np.ndarray:
        return self._pose.copy()

    def reset(self, start: np.ndarray, theta: float) -> None:
        b = self.bullet
        b.resetSimulation(physicsClientId=self.client)
        b.setAdditionalSearchPath(self.pybullet_data.getDataPath(), physicsClientId=self.client)
        b.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        b.setTimeStep(
            self.config.control_period / self.config.physics_substeps,
            physicsClientId=self.client,
        )
        b.setPhysicsEngineParameter(
            numSolverIterations=self.config.solver_iterations,
            physicsClientId=self.client,
        )
        self.plane_id = b.loadURDF("plane.urdf", physicsClientId=self.client)
        self.obstacle_ids = self._build_obstacles()
        quaternion = b.getQuaternionFromEuler((0.0, 0.0, theta))
        if self.config.robot_urdf:
            robot_urdf = self._resolve_robot_urdf(self.config.robot_urdf)
            self.robot_id = b.loadURDF(
                robot_urdf,
                (float(start[0]), float(start[1]), self.config.robot_base_height),
                quaternion,
                physicsClientId=self.client,
            )
            self._validate_wheel_joints()
        else:
            collision = b.createCollisionShape(
                b.GEOM_CYLINDER,
                radius=self.config.robot_radius,
                height=0.12,
                physicsClientId=self.client,
            )
            visual = b.createVisualShape(
                b.GEOM_CYLINDER,
                radius=self.config.robot_radius,
                length=0.12,
                rgbaColor=(0.1, 0.4, 0.8, 1.0),
                physicsClientId=self.client,
            )
            self.robot_id = b.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=(float(start[0]), float(start[1]), 0.08),
                baseOrientation=quaternion,
                physicsClientId=self.client,
            )
        for link_index in range(-1, b.getNumJoints(self.robot_id, physicsClientId=self.client)):
            b.changeDynamics(
                self.robot_id,
                link_index,
                lateralFriction=self.config.lateral_friction,
                physicsClientId=self.client,
            )
        self._pose[:] = (start[0], start[1], theta)

    def update_map(self, grid_map: GridMap) -> None:
        self.map = grid_map

    def _build_obstacles(self) -> list[int]:
        b = self.bullet
        if self.map.segments:
            ids = []
            wall_half_width = self.map.wall_thickness / 2.0
            for x1, y1, x2, y2 in self.map.segments:
                length = float(np.hypot(x2 - x1, y2 - y1))
                half_extents = (length / 2.0, wall_half_width, 0.25)
                collision = b.createCollisionShape(
                    b.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client
                )
                visual = b.createVisualShape(
                    b.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=(0.3, 0.3, 0.3, 1.0),
                    physicsClientId=self.client,
                )
                ids.append(
                    b.createMultiBody(
                        baseMass=0.0,
                        baseCollisionShapeIndex=collision,
                        baseVisualShapeIndex=visual,
                        basePosition=((x1 + x2) / 2.0, (y1 + y2) / 2.0, 0.25),
                        baseOrientation=b.getQuaternionFromEuler(
                            (0.0, 0.0, float(np.arctan2(y2 - y1, x2 - x1)))
                        ),
                        physicsClientId=self.client,
                    )
                )
            return ids
        half = self.map.cell_size / 2.0
        shape = b.createCollisionShape(
            b.GEOM_BOX, halfExtents=(half, half, 0.25), physicsClientId=self.client
        )
        visual = b.createVisualShape(
            b.GEOM_BOX,
            halfExtents=(half, half, 0.25),
            rgbaColor=(0.3, 0.3, 0.3, 1.0),
            physicsClientId=self.client,
        )
        ids = []
        for row, column in np.argwhere(self.map.occupied):
            x, y = self.map.cell_center(int(row), int(column))
            ids.append(
                b.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=shape,
                    baseVisualShapeIndex=visual,
                    basePosition=(x, y, 0.25),
                    physicsClientId=self.client,
                )
            )
        return ids

    @staticmethod
    def _resolve_robot_urdf(value: str) -> str:
        prefix = "bundled:"
        if value.startswith(prefix):
            filename = value[len(prefix) :]
            if Path(filename).name != filename:
                raise ValueError("Bundled URDF must be a filename without directories")
            return str(files("pidmppo").joinpath("assets/robots", filename))
        return value

    def _validate_wheel_joints(self) -> None:
        b = self.bullet
        joint_count = b.getNumJoints(self.robot_id, physicsClientId=self.client)
        requested = self.config.left_wheel_joints + self.config.right_wheel_joints
        if any(index < 0 or index >= joint_count for index in requested):
            raise ValueError(
                f"Wheel joint index outside URDF range [0, {joint_count - 1}]: {requested}"
            )
        if self.config.robot_urdf.startswith("bundled:"):
            names = {
                index: b.getJointInfo(self.robot_id, index, physicsClientId=self.client)[1].decode()
                for index in requested
            }
            expected = {0: "left_wheel_joint", 1: "right_wheel_joint"}
            if names != expected:
                raise ValueError(f"Bundled TurtleBot3 wheel joint mismatch: {names}")

    def step(self, linear: float, angular: float, period: float) -> bool:
        b = self.bullet
        left = self.config.left_wheel_joints
        right = self.config.right_wheel_joints
        if left and right:
            left_velocity = (linear - angular * self.config.axle_length / 2.0) / self.config.wheel_radius
            right_velocity = (linear + angular * self.config.axle_length / 2.0) / self.config.wheel_radius
            joints = list(left) + list(right)
            velocities = [left_velocity] * len(left) + [right_velocity] * len(right)
            b.setJointMotorControlArray(
                self.robot_id,
                joints,
                b.VELOCITY_CONTROL,
                targetVelocities=velocities,
                physicsClientId=self.client,
            )
            for _ in range(self.config.physics_substeps):
                b.stepSimulation(physicsClientId=self.client)
        else:
            theta = float((self._pose[2] + angular * period + np.pi) % (2.0 * np.pi) - np.pi)
            x = float(self._pose[0] + linear * np.cos(theta) * period)
            y = float(self._pose[1] + linear * np.sin(theta) * period)
            position, _ = b.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
            b.resetBasePositionAndOrientation(
                self.robot_id,
                (x, y, position[2]),
                b.getQuaternionFromEuler((0.0, 0.0, theta)),
                physicsClientId=self.client,
            )
            b.performCollisionDetection(physicsClientId=self.client)
        position, orientation = b.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        yaw = b.getEulerFromQuaternion(orientation)[2]
        self._pose[:] = (position[0], position[1], yaw)
        contacts = b.getContactPoints(bodyA=self.robot_id, physicsClientId=self.client)
        return any(contact[2] != self.plane_id for contact in contacts)

    def lidar(self, beams: int, minimum: float, maximum: float) -> np.ndarray:
        b = self.bullet
        angles = self._pose[2] + np.linspace(-np.pi, np.pi, beams, endpoint=False)
        z = self.config.lidar_height
        start_range = max(minimum, self.config.robot_radius + 0.01)
        starts = [
            (self._pose[0] + start_range * np.cos(a), self._pose[1] + start_range * np.sin(a), z)
            for a in angles
        ]
        ends = [
            (self._pose[0] + maximum * np.cos(a), self._pose[1] + maximum * np.sin(a), z)
            for a in angles
        ]
        results = b.rayTestBatch(starts, ends, physicsClientId=self.client)
        span = maximum - start_range
        return np.asarray(
            [maximum if result[0] < 0 else start_range + result[2] * span for result in results],
            dtype=np.float32,
        )

    def close(self) -> None:
        if self.bullet.isConnected(self.client):
            self.bullet.disconnect(self.client)


def create_backend(grid_map: GridMap, config: EnvConfig) -> NavigationBackend:
    if config.backend == "grid":
        return GridBackend(grid_map, config.robot_radius)
    if config.backend == "pybullet":
        return PyBulletBackend(grid_map, config)
    raise ValueError(f"Unsupported backend: {config.backend}")
