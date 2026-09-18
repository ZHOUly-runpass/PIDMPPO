from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from ..config import ModelConfig
from .pidm import PIDMCell


class RecurrentState(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

    def detach(self) -> "RecurrentState":
        return RecurrentState(self.first.detach(), self.second.detach())


class RecurrentEncoder(nn.Module):
    hidden_size: int

    def initial_state(self, batch_size: int, device: torch.device | str) -> RecurrentState:
        zeros = torch.zeros(batch_size, self.hidden_size, device=device)
        return RecurrentState(zeros, zeros.clone())

    def forward_sequence(
        self,
        observations: torch.Tensor,
        state: RecurrentState | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentState]:
        raise NotImplementedError


class PIDMEncoder(RecurrentEncoder):
    def __init__(self, observation_dim: int, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cell = PIDMCell(
            observation_dim,
            config.hidden_size,
            memory_mode=config.memory_mode,
            use_delta=config.use_delta,
            fusion_mode=config.fusion_mode,
            memory_bound=config.memory_bound,
            gate_temperature=config.gate_temperature,
        )
        self.last_attention: torch.Tensor | None = None
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward_sequence(self, observations, state=None, episode_starts=None):
        batch, steps, _ = observations.shape
        state = state or self.initial_state(batch, observations.device)
        feature, memory = state
        outputs = []
        diagnostics: dict[str, list[torch.Tensor]] = {}
        for index in range(steps):
            if episode_starts is not None:
                keep = (1.0 - episode_starts[:, index].float()).unsqueeze(-1)
                feature, memory = feature * keep, memory * keep
            hidden, feature, memory, step_diagnostics = self.cell(
                observations[:, index], feature, memory
            )
            outputs.append(hidden)
            for name, value in step_diagnostics.items():
                diagnostics.setdefault(name, []).append(value)
        self.last_diagnostics = {
            name: torch.stack(values, dim=1) for name, values in diagnostics.items()
        }
        self.last_attention = self.last_diagnostics["attention"]
        return torch.stack(outputs, dim=1), RecurrentState(feature, memory)


class GRUEncoder(RecurrentEncoder):
    def __init__(self, observation_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(observation_dim, hidden_size)

    def forward_sequence(self, observations, state=None, episode_starts=None):
        batch, steps, _ = observations.shape
        state = state or self.initial_state(batch, observations.device)
        hidden = state.first
        outputs = []
        for index in range(steps):
            if episode_starts is not None:
                hidden = hidden * (1.0 - episode_starts[:, index].float()).unsqueeze(-1)
            hidden = self.cell(observations[:, index], hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), RecurrentState(hidden, torch.zeros_like(hidden))


class LSTMEncoder(RecurrentEncoder):
    def __init__(self, observation_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(observation_dim, hidden_size)

    def forward_sequence(self, observations, state=None, episode_starts=None):
        batch, steps, _ = observations.shape
        state = state or self.initial_state(batch, observations.device)
        hidden, cell = state
        outputs = []
        for index in range(steps):
            if episode_starts is not None:
                keep = (1.0 - episode_starts[:, index].float()).unsqueeze(-1)
                hidden, cell = hidden * keep, cell * keep
            hidden, cell = self.cell(observations[:, index], (hidden, cell))
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), RecurrentState(hidden, cell)


class MLPEncoder(RecurrentEncoder):
    """Stateless observation encoder used by the plain PPO baseline."""

    def __init__(self, observation_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh()
        )

    def forward_sequence(self, observations, state=None, episode_starts=None):
        hidden = self.network(observations)
        batch = observations.shape[0]
        return hidden, self.initial_state(batch, observations.device)


def build_encoder(observation_dim: int, config: ModelConfig) -> RecurrentEncoder:
    if config.encoder == "pidm":
        return PIDMEncoder(observation_dim, config)
    if config.encoder == "gru":
        return GRUEncoder(observation_dim, config.hidden_size)
    if config.encoder == "lstm":
        return LSTMEncoder(observation_dim, config.hidden_size)
    if config.encoder == "mlp":
        return MLPEncoder(observation_dim, config.hidden_size)
    raise ValueError(f"Unsupported encoder: {config.encoder}")
