from __future__ import annotations

from copy import deepcopy

from .config import ExperimentConfig


VARIANTS = {
    "pidmppo": {},
    "ppo": {"encoder": "mlp", "twin_critic": False, "auxiliary_heads": False},
    "lstm_ppo": {"encoder": "lstm", "twin_critic": False, "auxiliary_heads": False},
    "gru_ppo": {"encoder": "gru", "twin_critic": False, "auxiliary_heads": False},
    "lstm_auxiliary": {"encoder": "lstm", "twin_critic": False, "auxiliary_heads": True},
    "lstm_dual_value": {"encoder": "lstm", "twin_critic": True, "auxiliary_heads": False},
    "matched_lstm": {"encoder": "lstm", "twin_critic": True, "auxiliary_heads": True},
    "matched_gru": {"encoder": "gru", "twin_critic": True, "auxiliary_heads": True},
    "without_i": {"memory_mode": "gru"},
    "without_d": {"use_delta": False},
    "without_attention": {"fusion_mode": "mean"},
    "single_critic": {"twin_critic": False},
    "without_auxiliary": {"auxiliary_heads": False},
    "l_loss_only": {"enable_l_head": True, "enable_g_head": False},
    "g_loss_only": {"enable_l_head": False, "enable_g_head": True},
}


def apply_variant(config: ExperimentConfig, name: str) -> ExperimentConfig:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant {name!r}; choose one of {sorted(VARIANTS)}")
    result = deepcopy(config)
    for key, value in VARIANTS[name].items():
        setattr(result.model, key, value)
    result.validate()
    return result
