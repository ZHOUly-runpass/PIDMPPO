from __future__ import annotations

import torch
from torch import nn


class PIDMCell(nn.Module):
    """PID-inspired gated residual memory cell.

    ``memory_mode='pidm'`` implements the bounded gated I-like accumulator.
    ``memory_mode='gru'`` is the paper ablation replacing that branch with a
    generic GRU memory. ``memory_mode='none'`` removes the branch entirely.
    """

    def __init__(
        self,
        observation_dim: int,
        hidden_size: int,
        *,
        memory_mode: str = "pidm",
        use_delta: bool = True,
        fusion_mode: str = "attention",
        memory_bound: float = 1.0,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if memory_mode not in {"pidm", "gru", "none"}:
            raise ValueError(f"Unsupported memory mode: {memory_mode}")
        self.hidden_size = hidden_size
        self.memory_mode = memory_mode
        self.use_delta = use_delta
        self.fusion_mode = fusion_mode
        self.memory_bound = memory_bound
        self.gate_temperature = gate_temperature
        self.p_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.d_encoder = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.Tanh())
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.candidate = nn.Linear(2 * hidden_size, hidden_size)
        self.gru_memory = nn.GRUCell(hidden_size, hidden_size)
        self.i_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.attention = nn.Linear(3 * hidden_size, 3)
        self.output = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )

    def forward(
        self,
        observation: torch.Tensor,
        previous_feature: torch.Tensor,
        previous_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        feature = self.p_encoder(observation)
        if self.use_delta:
            delta = feature - previous_feature
            derivative = self.d_encoder(torch.cat((feature, delta), dim=-1))
        else:
            derivative = torch.zeros_like(feature)

        if self.memory_mode == "pidm":
            context = torch.cat((feature, previous_memory), dim=-1)
            gate = torch.sigmoid(self.gate(context) / self.gate_temperature)
            candidate = self.memory_bound * torch.tanh(self.candidate(context))
            memory = (1.0 - gate) * previous_memory + gate * candidate
            integral = self.i_projection(memory)
        elif self.memory_mode == "gru":
            memory = self.gru_memory(feature, previous_memory)
            integral = self.i_projection(memory)
        else:
            memory = torch.zeros_like(previous_memory)
            integral = torch.zeros_like(feature)

        branches = torch.stack((feature, integral, derivative), dim=-2)
        if self.fusion_mode == "attention":
            weights = torch.softmax(self.attention(torch.cat((feature, integral, derivative), dim=-1)), dim=-1)
        else:
            weights = torch.full(
                (*feature.shape[:-1], 3), 1.0 / 3.0, dtype=feature.dtype, device=feature.device
            )
        if self.fusion_mode == "sum":
            fused = branches.sum(dim=-2)
        else:
            fused = (weights.unsqueeze(-1) * branches).sum(dim=-2)
        hidden = self.output(torch.cat((fused, memory, derivative), dim=-1))
        diagnostics = {
            "p": feature,
            "i": integral,
            "d": derivative,
            "memory": memory,
            "memory_norm": torch.linalg.vector_norm(memory, dim=-1),
            "gate": gate if self.memory_mode == "pidm" else torch.zeros_like(feature),
            "attention": weights,
        }
        return hidden, feature, memory, diagnostics
