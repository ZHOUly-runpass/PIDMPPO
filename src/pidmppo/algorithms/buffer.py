from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.recurrent import RecurrentState
from .auxiliary import build_l_targets


@dataclass
class SequenceBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_value1: torch.Tensor
    old_value2: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    l_targets: torch.Tensor
    l_mask: torch.Tensor
    g_targets: torch.Tensor
    valid_mask: torch.Tensor
    episode_starts: torch.Tensor
    initial_state: RecurrentState


class RolloutBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, hidden_size: int):
        self.capacity = capacity
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.raw_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.timeout_values = np.zeros(capacity, dtype=np.float32)
        self.successes = np.zeros(capacity, dtype=bool)
        self.episode_starts = np.zeros(capacity, dtype=np.float32)
        self.value1 = np.zeros(capacity, dtype=np.float32)
        self.value2 = np.zeros(capacity, dtype=np.float32)
        self.state_first = np.zeros((capacity, hidden_size), dtype=np.float32)
        self.state_second = np.zeros((capacity, hidden_size), dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.l_targets = np.zeros(capacity, dtype=np.float32)
        self.l_mask = np.zeros(capacity, dtype=np.float32)
        self.g_targets = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    @property
    def full(self) -> bool:
        return self.position == self.capacity

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        success: bool,
        episode_start: bool,
        value1: float,
        value2: float,
        state: RecurrentState,
        *,
        raw_action: np.ndarray | None = None,
        terminated: bool | None = None,
        truncated: bool = False,
        timeout_value: float = 0.0,
    ) -> None:
        if self.full:
            raise RuntimeError("RolloutBuffer is full")
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        if raw_action is None:
            raise ValueError("Rollouts must store raw_action; saturated actions cannot be inverted")
        self.raw_actions[index] = raw_action
        self.log_probs[index] = log_prob
        self.rewards[index] = reward
        self.dones[index] = done
        self.terminated[index] = done if terminated is None else terminated
        self.truncated[index] = truncated
        if done != bool(self.terminated[index] or truncated):
            raise ValueError("done must equal terminated or truncated")
        self.timeout_values[index] = timeout_value
        self.successes[index] = success
        self.episode_starts[index] = float(episode_start)
        self.value1[index] = value1
        self.value2[index] = value2
        self.state_first[index] = state.first.squeeze(0).detach().cpu().numpy()
        self.state_second[index] = state.second.squeeze(0).detach().cpu().numpy()
        self.position += 1

    def finalize(
        self,
        last_value1: float,
        last_value2: float,
        *,
        gamma: float,
        gae_lambda: float,
        max_episode_steps: int,
        failure_tail_steps: int,
        normalize_g_target: bool,
    ) -> None:
        if not self.full:
            raise RuntimeError("Cannot finalize an incomplete rollout")
        conservative_values = np.minimum(self.value1, self.value2)
        last_value = min(last_value1, last_value2)
        last_advantage = 0.0
        for index in reversed(range(self.capacity)):
            if index == self.capacity - 1:
                next_value = last_value
            else:
                next_value = conservative_values[index + 1]
            if self.truncated[index] and not self.terminated[index]:
                next_value = self.timeout_values[index]
            bootstrap = 1.0 - float(self.terminated[index])
            continuation = 1.0 - float(self.dones[index])
            delta = self.rewards[index] + gamma * next_value * bootstrap - conservative_values[index]
            last_advantage = delta + gamma * gae_lambda * continuation * last_advantage
            self.advantages[index] = last_advantage
        self.returns = self.advantages + conservative_values
        self.l_targets, self.l_mask = build_l_targets(
            self.dones,
            self.successes,
            max_episode_steps=max_episode_steps,
            failure_tail_steps=failure_tail_steps,
        )
        self.g_targets = self.returns.copy()
        if normalize_g_target:
            self.g_targets = (self.g_targets - self.g_targets.mean()) / (self.g_targets.std() + 1e-8)

    def batches(
        self,
        sequence_length: int,
        minibatch_size: int,
        device: torch.device,
        rng: np.random.Generator,
    ):
        chunks: list[tuple[int, int]] = []
        start = 0
        for boundary in np.flatnonzero(self.dones):
            end = int(boundary) + 1
            chunks.extend(self._split(start, end, sequence_length))
            start = end
        if start < self.capacity:
            chunks.extend(self._split(start, self.capacity, sequence_length))
        order = rng.permutation(len(chunks))
        per_batch = max(1, minibatch_size // sequence_length)
        for offset in range(0, len(order), per_batch):
            selected = [chunks[index] for index in order[offset : offset + per_batch]]
            yield self._make_batch(selected, sequence_length, device)

    @staticmethod
    def _split(start: int, end: int, length: int) -> list[tuple[int, int]]:
        return [(index, min(index + length, end)) for index in range(start, end, length)]

    def _make_batch(
        self, chunks: list[tuple[int, int]], sequence_length: int, device: torch.device
    ) -> SequenceBatch:
        batch_size = len(chunks)

        def padded(source: np.ndarray, fill: float = 0.0) -> np.ndarray:
            shape = (batch_size, sequence_length, *source.shape[1:])
            result = np.full(shape, fill, dtype=source.dtype)
            for row, (start, end) in enumerate(chunks):
                result[row, : end - start] = source[start:end]
            return result

        valid = np.zeros((batch_size, sequence_length), dtype=np.float32)
        for row, (start, end) in enumerate(chunks):
            valid[row, : end - start] = 1.0
        first = np.stack([self.state_first[start] for start, _ in chunks])
        second = np.stack([self.state_second[start] for start, _ in chunks])

        tensor = lambda value: torch.as_tensor(value, device=device)
        return SequenceBatch(
            observations=tensor(padded(self.observations)),
            actions=tensor(padded(self.actions)),
            raw_actions=tensor(padded(self.raw_actions)),
            old_log_probs=tensor(padded(self.log_probs)),
            old_value1=tensor(padded(self.value1)),
            old_value2=tensor(padded(self.value2)),
            returns=tensor(padded(self.returns)),
            advantages=tensor(padded(self.advantages)),
            l_targets=tensor(padded(self.l_targets)),
            l_mask=tensor(padded(self.l_mask)),
            g_targets=tensor(padded(self.g_targets)),
            valid_mask=tensor(valid),
            episode_starts=tensor(padded(self.episode_starts)),
            initial_state=RecurrentState(tensor(first), tensor(second)),
        )
