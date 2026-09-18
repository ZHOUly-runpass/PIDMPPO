"""PIDM-PPO reference implementation."""

from .config import ExperimentConfig, apply_overrides, load_config

__all__ = ["ExperimentConfig", "apply_overrides", "load_config"]
__version__ = "0.1.0"
