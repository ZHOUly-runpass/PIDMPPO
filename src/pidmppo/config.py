from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class EnvConfig:
    backend: str = "grid"
    map_name: str = "map1"
    map_file: str | None = None
    lidar_beams: int = 64
    lidar_min_range: float = 0.12
    lidar_max_range: float = 3.5
    action_history: int = 9
    goal_radius: float = 0.20
    collision_distance: float = 0.13
    robot_radius: float = 0.105
    control_period: float = 0.10
    max_linear_velocity: float = 0.26
    max_angular_velocity: float = 1.82
    max_episode_steps: int = 600
    progress_scale: float = 1.0
    terminal_reward: float = 100.0
    step_penalty: float = -0.01
    angular_penalty: float = 0.01
    stagnation_window: int = 20
    stagnation_epsilon: float = 0.001
    stagnation_penalty: float = -0.02
    randomize_start_goal: bool = True
    randomize_obstacles: bool = True
    randomized_map_width: int = 26
    randomized_map_height: int = 16
    randomized_obstacle_segments: int = 8
    randomized_cell_size: float = 0.25
    allow_reverse: bool = False
    pybullet_gui: bool = False
    robot_urdf: str | None = None
    left_wheel_joints: tuple[int, ...] = ()
    right_wheel_joints: tuple[int, ...] = ()
    wheel_radius: float = 0.033
    axle_length: float = 0.160
    physics_substeps: int = 24
    lidar_height: float = 0.18
    robot_base_height: float = 0.02
    lateral_friction: float = 0.8
    solver_iterations: int = 50

    @property
    def observation_dim(self) -> int:
        return self.lidar_beams + 3 + 2 * self.action_history


@dataclass
class ModelConfig:
    encoder: str = "pidm"
    hidden_size: int = 512
    actor_hidden_size: int = 512
    memory_mode: str = "pidm"
    use_delta: bool = True
    fusion_mode: str = "attention"
    memory_bound: float = 1.0
    gate_temperature: float = 1.0
    twin_critic: bool = True
    auxiliary_heads: bool = True
    enable_l_head: bool = True
    enable_g_head: bool = True


@dataclass
class PPOConfig:
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 8
    rollout_steps: int = 4096
    minibatch_size: int = 256
    sequence_length: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    auxiliary_coef: float = 0.1
    l_weight: float = 1.0
    g_weight: float = 0.5
    failure_tail_steps: int = 20
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    normalize_g_target: bool = True


@dataclass
class EvaluationConfig:
    episodes_per_map: int = 100
    maps: tuple[str, ...] = ("map1", "map2", "map3", "map4")
    bootstrap_samples: int = 10_000
    confidence_level: float = 0.95
    checkpoint_rule: str = "final"
    scenario_manifest: str = "configs/scenarios.json"


@dataclass
class LoggingConfig:
    log_interval: int = 4096
    save_interval: int = 50_000
    record_diagnostics: bool = False
    write_csv: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    total_timesteps: int = 1_500_000
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        if self.env.observation_dim != 85:
            raise ValueError(
                "The paper configuration must contain 64 lidar values, three goal "
                "features and nine two-dimensional actions (85 values total)."
            )
        if self.model.encoder not in {"pidm", "gru", "lstm", "mlp"}:
            raise ValueError(f"Unsupported encoder: {self.model.encoder}")
        if self.model.memory_mode not in {"pidm", "gru", "none"}:
            raise ValueError(f"Unsupported memory mode: {self.model.memory_mode}")
        if self.model.fusion_mode not in {"attention", "mean", "sum"}:
            raise ValueError(f"Unsupported fusion mode: {self.model.fusion_mode}")
        if self.model.memory_bound <= 0.0:
            raise ValueError("memory_bound must be positive")
        if self.model.gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive")
        if self.env.backend not in {"grid", "pybullet"}:
            raise ValueError(f"Unsupported environment backend: {self.env.backend}")
        if bool(self.env.left_wheel_joints) != bool(self.env.right_wheel_joints):
            raise ValueError("left and right wheel joints must be specified together")
        if self.env.left_wheel_joints and self.env.robot_urdf is None:
            raise ValueError("robot_urdf is required when wheel joints are configured")
        if self.env.randomized_map_width < 8 or self.env.randomized_map_height < 8:
            raise ValueError("randomized training map must be at least 8 by 8 cells")
        if self.env.randomized_obstacle_segments < 0:
            raise ValueError("randomized_obstacle_segments cannot be negative")
        if self.env.robot_radius <= 0.0 or self.env.robot_radius >= self.env.collision_distance:
            raise ValueError("robot_radius must be positive and smaller than collision_distance")
        if self.ppo.sequence_length > self.ppo.rollout_steps:
            raise ValueError("sequence_length cannot exceed rollout_steps")
        if self.ppo.minibatch_size < self.ppo.sequence_length:
            raise ValueError("minibatch_size must be at least sequence_length")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    top_level = {item.name for item in fields(ExperimentConfig)}
    unknown = set(raw) - top_level
    if unknown:
        raise ValueError(f"Unknown ExperimentConfig keys: {sorted(unknown)}")
    config = ExperimentConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        total_timesteps=raw.get("total_timesteps", 1_500_000),
        env=_construct(EnvConfig, raw.get("env", {})),
        model=_construct(ModelConfig, raw.get("model", {})),
        ppo=_construct(PPOConfig, raw.get("ppo", {})),
        evaluation=_construct(EvaluationConfig, raw.get("evaluation", {})),
        logging=_construct(LoggingConfig, raw.get("logging", {})),
    )
    config.validate()
    return config


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply explicit ``section.field=value`` overrides used by matrix jobs."""

    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"Override must be KEY=VALUE: {expression!r}")
        dotted_key, raw_value = expression.split("=", 1)
        parts = dotted_key.split(".")
        target: Any = config
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"Unknown override path: {dotted_key}")
            target = getattr(target, part)
            if not is_dataclass(target):
                raise ValueError(f"Override parent is not a config section: {dotted_key}")
        field_name = parts[-1]
        if not hasattr(target, field_name):
            raise ValueError(f"Unknown override path: {dotted_key}")
        setattr(target, field_name, yaml.safe_load(raw_value))
    config.validate()
    return config
