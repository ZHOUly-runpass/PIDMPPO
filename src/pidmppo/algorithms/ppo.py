from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from ..config import ExperimentConfig
from ..models import ActorCritic, RecurrentState
from .auxiliary import masked_mse
from .buffer import RolloutBuffer


def clipped_twin_value_loss(
    value1: torch.Tensor,
    value2: torch.Tensor,
    old_value1: torch.Tensor,
    old_value2: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """PPO-clipped regression for both critics; the minimum is not used here."""

    losses = []
    for value, old_value in ((value1, old_value1), (value2, old_value2)):
        clipped = old_value + (value - old_value).clamp(-clip_coef, clip_coef)
        loss = torch.maximum((value - returns).pow(2), (clipped - returns).pow(2))
        losses.append((loss * mask).sum() / mask.sum().clamp_min(1.0))
    return losses[0] + losses[1]


def clipped_value_loss(
    value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    clipped = old_value + (value - old_value).clamp(-clip_coef, clip_coef)
    loss = torch.maximum((value - returns).pow(2), (clipped - returns).pow(2))
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


@dataclass
class TrainMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    l_loss: float = 0.0
    g_loss: float = 0.0
    episodes: int = 0
    successes: int = 0


class PPOTrainer:
    def __init__(self, env: gym.Env, config: ExperimentConfig):
        self.env = env
        self.config = config
        self.device = self._resolve_device(config.device)
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        self.model = ActorCritic(observation_dim, action_dim, config.model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.ppo.learning_rate)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        self.observation, _ = env.reset(seed=config.seed)
        self.state = self.model.initial_state(1, self.device)
        self.episode_start = True
        self.global_step = 0

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def collect_rollout(self) -> tuple[RolloutBuffer, TrainMetrics]:
        ppo = self.config.ppo
        buffer = RolloutBuffer(
            ppo.rollout_steps,
            self.config.env.observation_dim,
            int(np.prod(self.env.action_space.shape)),
            self.config.model.hidden_size,
        )
        metrics = TrainMetrics()
        for _ in range(ppo.rollout_steps):
            state_before = self.state
            observation_tensor = torch.as_tensor(self.observation, device=self.device).view(1, 1, -1)
            starts = torch.tensor([[self.episode_start]], device=self.device)
            with torch.no_grad():
                output = self.model(observation_tensor, self.state, starts)
                action, log_prob = self.model.sample_action(output.mean, output.log_std)
            numpy_action = action[0, 0].cpu().numpy()
            next_observation, reward, terminated, truncated, info = self.env.step(numpy_action)
            done = terminated or truncated
            buffer.add(
                self.observation,
                numpy_action,
                float(log_prob.item()),
                reward,
                done,
                bool(info.get("success", False)),
                self.episode_start,
                float(output.value1.item()),
                float(output.value2.item()),
                state_before,
            )
            self.global_step += 1
            self.state = output.state.detach()
            self.observation = next_observation
            self.episode_start = done
            if done:
                metrics.episodes += 1
                metrics.successes += int(info.get("success", False))
                self.observation, _ = self.env.reset()
                self.state = self.model.initial_state(1, self.device)

        with torch.no_grad():
            last = self.model(
                torch.as_tensor(self.observation, device=self.device).view(1, 1, -1),
                self.state,
                torch.tensor([[self.episode_start]], device=self.device),
            )
        buffer.finalize(
            float(last.value1.item()),
            float(last.value2.item()),
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            max_episode_steps=self.config.env.max_episode_steps,
            failure_tail_steps=ppo.failure_tail_steps,
            normalize_g_target=ppo.normalize_g_target,
        )
        return buffer, metrics

    def update(self, buffer: RolloutBuffer) -> TrainMetrics:
        ppo = self.config.ppo
        metrics = TrainMetrics()
        updates = 0
        for _ in range(ppo.update_epochs):
            for batch in buffer.batches(
                ppo.sequence_length, ppo.minibatch_size, self.device, self.rng
            ):
                output = self.model(batch.observations, batch.initial_state, batch.episode_starts)
                log_prob, entropy = self.model.evaluate_action(
                    output.mean, output.log_std, batch.actions
                )
                mask = batch.valid_mask
                advantages = batch.advantages
                if ppo.normalize_advantage:
                    selected = advantages[mask.bool()]
                    advantages = (advantages - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
                ratio = (log_prob - batch.old_log_probs).exp()
                unclipped = ratio * advantages
                clipped = ratio.clamp(1.0 - ppo.clip_coef, 1.0 + ppo.clip_coef) * advantages
                policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / mask.sum()
                if self.config.model.twin_critic:
                    value_loss = clipped_twin_value_loss(
                        output.value1,
                        output.value2,
                        batch.old_value1,
                        batch.old_value2,
                        batch.returns,
                        ppo.clip_coef,
                        mask,
                    )
                else:
                    value_loss = clipped_value_loss(
                        output.value1, batch.old_value1, batch.returns, ppo.clip_coef, mask
                    )
                entropy_mean = (entropy * mask).sum() / mask.sum()
                zero = policy_loss.new_zeros(())
                l_loss = (
                    masked_mse(
                        output.l_prediction,
                        batch.l_targets,
                        batch.l_mask,
                        normalizer_mask=mask,
                        scale=0.5,
                    )
                    if self.config.model.auxiliary_heads and self.config.model.enable_l_head
                    else zero
                )
                g_loss = (
                    masked_mse(output.g_prediction, batch.g_targets, mask, normalizer_mask=mask)
                    if self.config.model.auxiliary_heads and self.config.model.enable_g_head
                    else zero
                )
                auxiliary = ppo.l_weight * l_loss + ppo.g_weight * g_loss
                loss = (
                    policy_loss
                    + ppo.value_coef * value_loss
                    - ppo.entropy_coef * entropy_mean
                    + ppo.auxiliary_coef * auxiliary
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), ppo.max_grad_norm)
                self.optimizer.step()
                metrics.policy_loss += float(policy_loss.detach())
                metrics.value_loss += float(value_loss.detach())
                metrics.entropy += float(entropy_mean.detach())
                metrics.l_loss += float(l_loss.detach())
                metrics.g_loss += float(g_loss.detach())
                updates += 1
        if updates:
            for field in ("policy_loss", "value_loss", "entropy", "l_loss", "g_loss"):
                setattr(metrics, field, getattr(metrics, field) / updates)
        return metrics

    def train(
        self,
        total_timesteps: int | None = None,
        checkpoint_dir: str | Path | None = None,
        metrics_path: str | Path | None = None,
    ):
        target = total_timesteps or self.config.total_timesteps
        next_checkpoint = self.config.logging.save_interval
        metrics_file = Path(metrics_path) if metrics_path is not None else None
        if metrics_file is not None:
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
        while self.global_step < target:
            buffer, rollout_metrics = self.collect_rollout()
            update_metrics = self.update(buffer)
            success_rate = (
                rollout_metrics.successes / rollout_metrics.episodes
                if rollout_metrics.episodes
                else float("nan")
            )
            print(
                f"step={self.global_step} episodes={rollout_metrics.episodes} "
                f"success_rate={success_rate:.3f} policy={update_metrics.policy_loss:.4f} "
                f"value={update_metrics.value_loss:.4f} L={update_metrics.l_loss:.4f} "
                f"G={update_metrics.g_loss:.4f}"
            )
            if metrics_file is not None:
                self._append_metrics(metrics_file, rollout_metrics, update_metrics, success_rate)
            if checkpoint_dir is not None and self.global_step >= next_checkpoint:
                self.save(Path(checkpoint_dir) / "latest.pt")
                self.save(Path(checkpoint_dir) / f"step_{self.global_step}.pt")
                while next_checkpoint <= self.global_step:
                    next_checkpoint += self.config.logging.save_interval
        if checkpoint_dir is not None:
            self.save(Path(checkpoint_dir) / "final.pt")
            self.save(Path(checkpoint_dir) / "latest.pt")

    def _append_metrics(
        self,
        path: Path,
        rollout: TrainMetrics,
        update: TrainMetrics,
        success_rate: float,
    ) -> None:
        row = {
            "global_step": self.global_step,
            "episodes": rollout.episodes,
            "successes": rollout.successes,
            "success_rate": success_rate,
            "policy_loss": update.policy_loss,
            "value_loss": update.value_loss,
            "entropy": update.entropy,
            "l_loss": update.l_loss,
            "g_loss": update.g_loss,
        }
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        return checkpoint
