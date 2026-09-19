from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

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
    diagnostics: dict[str, float] = field(default_factory=dict)


class PPOTrainer:
    def __init__(self, env: gym.Env, config: ExperimentConfig):
        self.env = env
        self.config = config
        self.device = self._resolve_device(config.device)
        observation_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        torch.manual_seed(config.seed)
        self.model = ActorCritic(observation_dim, action_dim, config.model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.ppo.learning_rate)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        self.observation, _ = env.reset(seed=config.seed)
        self.state = self.model.initial_state(1, self.device)
        self.episode_start = True
        self.global_step = 0
        self.update_count = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_terms: dict[str, float] = {}
        self.completed_episodes: list[dict[str, Any]] = []
        self.stop_requested: Callable[[], bool] = lambda: False

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
        policy_samples: list[np.ndarray] = []
        diagnostic_samples: list[np.ndarray] = []
        velocity_samples = []
        reward_terms: dict[str, float] = {}
        collisions = timeouts = stagnations = 0
        episode_returns, episode_lengths = [], []
        for _ in range(ppo.rollout_steps):
            state_before = self.state
            observation_tensor = torch.as_tensor(self.observation, device=self.device).view(1, 1, -1)
            starts = torch.tensor([[self.episode_start]], device=self.device)
            with torch.no_grad():
                output = self.model(observation_tensor, self.state, starts)
                action, log_prob, raw_action = self.model.sample_action(output.mean, output.log_std, return_raw=True)
            numpy_action = action[0, 0].cpu().numpy()
            policy_samples.append(torch.cat((output.mean.flatten(), output.log_std.exp().flatten())).cpu().numpy())
            diagnostics = output.diagnostics
            if "attention" in diagnostics:
                gate = diagnostics["gate"]
                diagnostic_samples.append(torch.cat((
                    diagnostics["attention"].flatten(),
                    ((gate < .05) | (gate > .95)).float().mean().view(1),
                    diagnostics["memory_norm"].flatten(),
                )).cpu().numpy())
            next_observation, reward, terminated, truncated, info = self.env.step(numpy_action)
            done = terminated or truncated
            timeout_value = 0.0
            if truncated and not terminated:
                # Use the final observation with the state produced by o_t, NOT
                # the next episode's reset observation or reset recurrent state.
                with torch.no_grad():
                    final_output = self.model(
                        torch.as_tensor(next_observation, device=self.device).view(1, 1, -1),
                        output.state, torch.zeros((1, 1), device=self.device),
                    )
                timeout_value = min(final_output.value1.item(), final_output.value2.item())
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
                raw_action=raw_action[0, 0].cpu().numpy(),
                terminated=terminated, truncated=truncated, timeout_value=timeout_value,
            )
            self.episode_return += float(reward)
            self.episode_length += 1
            for name, value in info.get("reward_terms", {}).items():
                reward_terms[name] = reward_terms.get(name, 0.0) + float(value)
                self.episode_terms[name] = self.episode_terms.get(name, 0.0) + float(value)
            velocity_samples.append(np.asarray(info.get("executed_velocity", [0.0, 0.0])))
            stagnations += int(info.get("reward_terms", {}).get("stagnation", 0.0) < 0)
            self.global_step += 1
            self.state = output.state.detach()
            self.observation = next_observation
            self.episode_start = done
            if done:
                metrics.episodes += 1
                metrics.successes += int(info.get("success", False))
                collisions += int(info.get("collision", False))
                timeouts += int(truncated and not terminated)
                episode_returns.append(self.episode_return)
                episode_lengths.append(self.episode_length)
                self.completed_episodes.append({
                    "global_step": self.global_step, "return": self.episode_return,
                    "length": self.episode_length, "success": bool(info.get("success", False)),
                    "collision": bool(info.get("collision", False)), "timeout": bool(truncated and not terminated),
                    "reward_terms": self.episode_terms.copy(),
                })
                self.episode_return, self.episode_length, self.episode_terms = 0.0, 0, {}
                if hasattr(self.env, "set_training_step"):
                    self.env.set_training_step(self.global_step)
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
        velocity = np.asarray(velocity_samples)
        samples = np.asarray(policy_samples)
        metrics.diagnostics.update({
            "collisions": collisions, "timeouts": timeouts,
            "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "action_saturation_rate": float(np.mean(np.abs(buffer.actions) > .99)),
            "linear_velocity_mean": float(velocity[:, 0].mean()),
            "angular_velocity_abs_mean": float(np.abs(velocity[:, 1]).mean()),
            "spin_fraction": float(np.mean((np.abs(velocity[:, 0]) < .03) & (np.abs(velocity[:, 1]) > 1.0))),
            "stagnation_fraction": stagnations / ppo.rollout_steps,
            "pidm_diagnostics_available": float(bool(diagnostic_samples)),
        })
        for index, axis in enumerate(("linear", "angular")):
            metrics.diagnostics[f"action_{axis}_mean"] = float(buffer.actions[:, index].mean())
            metrics.diagnostics[f"action_{axis}_std"] = float(buffer.actions[:, index].std())
            metrics.diagnostics[f"latent_{axis}_mean"] = float(samples[:, index].mean())
            metrics.diagnostics[f"latent_{axis}_std_mean"] = float(samples[:, 2 + index].mean())
        diagnostic_array = np.asarray(diagnostic_samples) if diagnostic_samples else np.zeros((1, 5))
        for index, name in enumerate(("attention_p", "attention_i", "attention_d", "gate_saturation", "memory_norm")):
            for label, value in (("mean", diagnostic_array[:, index].mean()), ("std", diagnostic_array[:, index].std()),
                                 ("min", diagnostic_array[:, index].min()), ("max", diagnostic_array[:, index].max())):
                metrics.diagnostics[f"{name}_{label}"] = float(value)
        for name in ("progress", "step", "angular", "stagnation", "terminal"):
            metrics.diagnostics[f"reward_{name}_mean"] = reward_terms.get(name, 0.0) / ppo.rollout_steps
        for name, values in (("value1", buffer.value1), ("value2", buffer.value2), ("return", buffer.returns)):
            for label, value in (("mean", values.mean()), ("std", values.std()), ("min", values.min()), ("max", values.max())):
                metrics.diagnostics[f"{name}_{label}"] = float(value)
        for name, values in (("value1", buffer.value1), ("value2", buffer.value2)):
            variance = float(np.var(buffer.returns))
            metrics.diagnostics[f"{name}_explained_variance"] = (
                1.0 - float(np.var(buffer.returns - values)) / variance if variance > 1e-12 else 0.0
            )
        metrics.diagnostics["explained_variance_defined"] = float(np.var(buffer.returns) > 1e-12)
        return buffer, metrics

    def update(self, buffer: RolloutBuffer) -> TrainMetrics:
        ppo = self.config.ppo
        metrics = TrainMetrics()
        updates = 0
        self.update_count += 1
        gradient_measured = False
        sums = {name: 0.0 for name in ("approx_kl", "clip_fraction", "gaussian_entropy", "grad_norm_pre_clip")}
        diagnostics = {"epochs_started": 0.0, "epochs_completed": 0.0, "kl_early_stop": 0.0,
                       "encoder_actor_grad_norm": 0.0, "encoder_critic_grad_norm": 0.0,
                       "encoder_auxiliary_grad_norm": 0.0, "encoder_grad_measured": 0.0,
                       "max_approx_kl": 0.0}
        kl_batches = 0
        for _ in range(ppo.update_epochs):
            diagnostics["epochs_started"] += 1
            stopped = False
            for batch in buffer.batches(
                ppo.sequence_length, ppo.minibatch_size, self.device, self.rng
            ):
                if self.stop_requested():
                    stopped = True
                    break
                output = self.model(batch.observations, batch.initial_state, batch.episode_starts)
                log_prob, entropy = self.model.evaluate_action(
                    output.mean, output.log_std, batch.actions, raw_action=batch.raw_actions,
                )
                mask = batch.valid_mask
                advantages = batch.advantages
                if ppo.normalize_advantage:
                    selected = advantages[mask.bool()]
                    advantages = (advantages - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
                ratio = (log_prob - batch.old_log_probs).exp()
                with torch.no_grad():
                    log_ratio = log_prob - batch.old_log_probs
                    approx_kl = float((((ratio - 1) - log_ratio) * mask).sum() / mask.sum())
                    clip_fraction = float((((ratio - 1).abs() > ppo.clip_coef).float() * mask).sum() / mask.sum())
                if not math.isfinite(approx_kl):
                    raise FloatingPointError("Non-finite PPO KL; stopping before optimizer update")
                sums["approx_kl"] += approx_kl
                sums["clip_fraction"] += clip_fraction
                kl_batches += 1
                diagnostics["max_approx_kl"] = max(diagnostics["max_approx_kl"], approx_kl)
                if ppo.target_kl is not None and approx_kl > 1.5 * ppo.target_kl:
                    diagnostics["kl_early_stop"] = 1.0
                    stopped = True
                    break
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
                        ppo.value_clip_coef if ppo.value_clip_coef is not None else ppo.clip_coef,
                        mask,
                    )
                else:
                    value_loss = clipped_value_loss(
                        output.value1, batch.old_value1, batch.returns,
                        ppo.value_clip_coef if ppo.value_clip_coef is not None else ppo.clip_coef, mask
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
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss; stopping before optimizer update")
                if self.update_count % ppo.gradient_diagnostics_interval == 0 and not gradient_measured:
                    parameters = tuple(self.model.encoder.parameters())
                    for name, component in (("actor", policy_loss - ppo.entropy_coef * entropy_mean),
                                            ("critic", ppo.value_coef * value_loss),
                                            ("auxiliary", ppo.auxiliary_coef * auxiliary)):
                        gradients = torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True) if component.requires_grad else ()
                        squared = sum(float(gradient.detach().square().sum()) for gradient in gradients if gradient is not None)
                        diagnostics[f"encoder_{name}_grad_norm"] = math.sqrt(squared)
                    gradient_measured = True
                    diagnostics["encoder_grad_measured"] = 1.0
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.model.parameters(), ppo.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                sums["grad_norm_pre_clip"] += float(norm)
                sums["gaussian_entropy"] += float(((output.log_std + .5 * math.log(2 * math.pi * math.e)).sum(-1) * mask).sum().detach() / mask.sum())
                metrics.policy_loss += float(policy_loss.detach())
                metrics.value_loss += float(value_loss.detach())
                metrics.entropy += float(entropy_mean.detach())
                metrics.l_loss += float(l_loss.detach())
                metrics.g_loss += float(g_loss.detach())
                updates += 1
            if stopped:
                break
            diagnostics["epochs_completed"] += 1
        if updates:
            for field in ("policy_loss", "value_loss", "entropy", "l_loss", "g_loss"):
                setattr(metrics, field, getattr(metrics, field) / updates)
        for name, value in sums.items():
            diagnostics[name] = value / max(1, kl_batches if name in {"approx_kl", "clip_fraction"} else updates)
        diagnostics["optimizer_steps"] = float(updates)
        metrics.diagnostics = diagnostics
        return metrics

    def train(
        self,
        total_timesteps: int | None = None,
        checkpoint_dir: str | Path | None = None,
        metrics_path: str | Path | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ):
        self.stop_requested = stop_requested or (lambda: False)
        target = total_timesteps or self.config.total_timesteps
        next_checkpoint = self.config.logging.save_interval
        metrics_file = Path(metrics_path) if metrics_path is not None else None
        if metrics_file is not None:
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
        while self.global_step < target:
            if self.stop_requested():
                break
            started = time.monotonic()
            buffer, rollout_metrics = self.collect_rollout()
            update_metrics = self.update(buffer)
            success_rate = (
                rollout_metrics.successes / rollout_metrics.episodes
                if rollout_metrics.episodes
                else 0.0
            )
            update_metrics.diagnostics["update_seconds"] = time.monotonic() - started
            update_metrics.diagnostics["steps_per_second"] = self.config.ppo.rollout_steps / max(time.monotonic() - started, 1e-9)
            print(
                f"step={self.global_step} episodes={rollout_metrics.episodes} "
                f"success_rate={success_rate:.3f} policy={update_metrics.policy_loss:.4f} "
                f"value={update_metrics.value_loss:.4f} L={update_metrics.l_loss:.4f} "
                f"G={update_metrics.g_loss:.4f}"
            )
            if metrics_file is not None:
                self._append_metrics(metrics_file, rollout_metrics, update_metrics, success_rate)
                episodes_path = metrics_file.with_name("training_episodes.jsonl")
                with episodes_path.open("a", encoding="utf-8") as stream:
                    for episode in self.completed_episodes:
                        stream.write(json.dumps(episode, allow_nan=False) + "\n")
            self.completed_episodes.clear()
            if checkpoint_dir is not None and self.global_step >= next_checkpoint:
                self.save(Path(checkpoint_dir) / "latest.pt")
                self.save(Path(checkpoint_dir) / f"step_{self.global_step}.pt")
                while next_checkpoint <= self.global_step:
                    next_checkpoint += self.config.logging.save_interval
        completed = self.global_step >= target and not self.stop_requested()
        if checkpoint_dir is not None:
            self.save(Path(checkpoint_dir) / ("final.pt" if completed else "interrupted.pt"))
            self.save(Path(checkpoint_dir) / "latest.pt")
        return {"status": "completed" if completed else "interrupted", "global_step": self.global_step,
                "target_timesteps": target, "updates": self.update_count, "implementation_version": "correctness-v2"}

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
            "implementation_version": "correctness-v2",
            "update": self.update_count,
            **rollout.diagnostics,
            **update.diagnostics,
        }
        if any(not math.isfinite(value) for value in row.values() if isinstance(value, (float, int))):
            raise FloatingPointError("Non-finite training metrics")
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "global_step": self.global_step,
                "implementation_version": "correctness-v2",
                "updates": self.update_count,
            },
            temporary,
        )
        temporary.replace(path)

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = int(checkpoint.get("global_step", 0))
        return checkpoint
