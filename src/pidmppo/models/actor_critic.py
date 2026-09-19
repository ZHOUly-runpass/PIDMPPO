from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from ..config import ModelConfig
from .recurrent import RecurrentEncoder, RecurrentState, build_encoder


@dataclass
class PolicyOutput:
    mean: torch.Tensor
    log_std: torch.Tensor
    value1: torch.Tensor
    value2: torch.Tensor
    l_prediction: torch.Tensor
    g_prediction: torch.Tensor
    state: RecurrentState
    hidden: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, config: ModelConfig):
        super().__init__()
        self.encoder: RecurrentEncoder = build_encoder(observation_dim, config)
        shared_input = observation_dim + config.hidden_size
        self.shared = nn.Sequential(
            nn.Linear(shared_input, config.actor_hidden_size),
            nn.ReLU(),
            nn.Linear(config.actor_hidden_size, config.actor_hidden_size),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(config.actor_hidden_size, action_dim)
        self.actor_log_std = nn.Linear(config.actor_hidden_size, action_dim)
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max
        if config.actor_mean_gain is not None:
            nn.init.orthogonal_(self.actor_mean.weight, gain=config.actor_mean_gain)
            nn.init.zeros_(self.actor_mean.bias)
        if config.log_std_init is not None:
            nn.init.zeros_(self.actor_log_std.weight)
            nn.init.constant_(self.actor_log_std.bias, config.log_std_init)
        self.value1 = nn.Linear(config.actor_hidden_size, 1)
        self.value2 = nn.Linear(config.actor_hidden_size, 1) if config.twin_critic else None
        self.auxiliary_heads = config.auxiliary_heads
        self.enable_l_head = config.enable_l_head
        self.enable_g_head = config.enable_g_head
        self.auxiliary_shared = (
            nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU())
            if config.auxiliary_heads
            else None
        )
        self.l_head = (
            nn.Linear(config.hidden_size, 1)
            if config.auxiliary_heads and config.enable_l_head
            else None
        )
        self.g_head = (
            nn.Linear(config.hidden_size, 1)
            if config.auxiliary_heads and config.enable_g_head
            else None
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> RecurrentState:
        return self.encoder.initial_state(batch_size, device)

    def forward(
        self,
        observations: torch.Tensor,
        state: RecurrentState | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> PolicyOutput:
        hidden, new_state = self.encoder.forward_sequence(observations, state, episode_starts)
        shared = self.shared(torch.cat((observations, hidden), dim=-1))
        mean = self.actor_mean(shared)
        log_std = self.actor_log_std(shared).clamp(self.log_std_min, self.log_std_max)
        auxiliary_features = self.auxiliary_shared(hidden) if self.auxiliary_shared is not None else hidden
        if self.l_head is not None:
            l_prediction = torch.sigmoid(self.l_head(auxiliary_features)).squeeze(-1)
        else:
            l_prediction = hidden.new_zeros(hidden.shape[:-1])
        if self.g_head is not None:
            g_prediction = self.g_head(auxiliary_features).squeeze(-1)
        else:
            g_prediction = hidden.new_zeros(hidden.shape[:-1])
        value1 = self.value1(shared).squeeze(-1)
        value2 = self.value2(shared).squeeze(-1) if self.value2 is not None else value1
        return PolicyOutput(
            mean=mean,
            log_std=log_std,
            value1=value1,
            value2=value2,
            l_prediction=l_prediction,
            g_prediction=g_prediction,
            state=new_state,
            hidden=hidden,
            diagnostics=getattr(self.encoder, "last_diagnostics", {}),
        )

    @staticmethod
    def sample_action(mean: torch.Tensor, log_std: torch.Tensor, *, return_raw: bool = False):
        distribution = Normal(mean, log_std.exp())
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_prob = ActorCritic._squashed_log_prob(distribution, raw, action)
        return (action, log_prob, raw) if return_raw else (action, log_prob)

    @staticmethod
    def evaluate_action(
        mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor,
        *, raw_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Inversion is retained only for legacy inference/trace callers. PPO MUST
        # pass the stored latent sample: tanh cannot be inverted at saturation.
        raw = raw_action if raw_action is not None else torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))
        distribution = Normal(mean, log_std.exp())
        log_prob = ActorCritic._squashed_log_prob(distribution, raw)
        # Reparameterized fresh sample estimates entropy of the CURRENT squashed
        # policy, not cross entropy on old rollout actions or base-Gaussian entropy.
        entropy_raw = distribution.rsample()
        entropy = -ActorCritic._squashed_log_prob(distribution, entropy_raw)
        return log_prob, entropy

    @staticmethod
    def _squashed_log_prob(distribution: Normal, raw: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        # Stable log(1 - tanh(raw)^2); no epsilon/clipped inverse approximation.
        correction = 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
        return (distribution.log_prob(raw) - correction).sum(dim=-1)
