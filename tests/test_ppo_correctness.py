from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from pidmppo.algorithms.buffer import RolloutBuffer
from pidmppo.algorithms.ppo import PPOTrainer, clipped_twin_value_loss
from pidmppo.config import ExperimentConfig, ModelConfig
from pidmppo.models import ActorCritic, RecurrentState


@pytest.mark.parametrize("scale", [0., 1., 10., 100.])
def test_squashed_probability_replay_at_extreme_saturation(scale):
    torch.manual_seed(17)
    mean = torch.randn(32, 8, 2) * scale
    log_std = torch.ones_like(mean) * 2
    action, old, raw = ActorCritic.sample_action(mean, log_std, return_raw=True)
    replay, entropy = ActorCritic.evaluate_action(mean, log_std, action, raw_action=raw)
    assert torch.isfinite(entropy).all()
    torch.testing.assert_close(replay, old, atol=1e-4, rtol=0)
    torch.testing.assert_close((replay - old).exp(), torch.ones_like(old), atol=1e-4, rtol=0)


def test_fresh_squashed_entropy_penalizes_saturation_with_finite_gradients():
    torch.manual_seed(8)
    mean = torch.full((4096, 2), 20., requires_grad=True)
    log_std = torch.zeros_like(mean, requires_grad=True)
    _, entropy = ActorCritic.evaluate_action(mean, log_std, torch.tanh(mean), raw_action=mean.detach())
    entropy.mean().backward()
    assert entropy.mean() < 0
    assert torch.isfinite(mean.grad).all() and mean.grad.mean() < 0
    assert torch.isfinite(log_std.grad).all()


@pytest.mark.parametrize("encoder", ["pidm", "gru", "lstm", "mlp"])
def test_sequence_matches_steps_and_episode_reset(encoder):
    torch.manual_seed(13)
    model = ActorCritic(85, 2, ModelConfig(encoder=encoder, hidden_size=8, actor_hidden_size=8))
    observations = torch.randn(2, 7, 85)
    starts = torch.zeros(2, 7)
    starts[:, (0, 3)] = 1
    sequence = model(observations, episode_starts=starts)
    state, means = None, []
    for t in range(7):
        output = model(observations[:, t:t + 1], state, starts[:, t:t + 1])
        state = output.state
        means.append(output.mean)
    torch.testing.assert_close(torch.cat(means, 1), sequence.mean, atol=1e-6, rtol=1e-5)
    independent = model(observations[:, 3:], episode_starts=starts[:, 3:])
    torch.testing.assert_close(independent.mean, sequence.mean[:, 3:])


def add_transition(buffer, reward, *, terminated=False, truncated=False, timeout_value=0.):
    buffer.add(np.zeros(85), np.zeros(2), 0., reward, terminated or truncated, terminated,
               buffer.position == 0, 2., 3., RecurrentState(torch.zeros(1, 8), torch.zeros(1, 8)),
               raw_action=np.zeros(2), terminated=terminated, truncated=truncated, timeout_value=timeout_value)


def test_timeout_bootstrap_and_gae_cut_and_padding():
    buffer = RolloutBuffer(3, 85, 2, 8)
    add_transition(buffer, 1., truncated=True, timeout_value=10.)
    add_transition(buffer, 100., terminated=True)
    add_transition(buffer, 5.)
    buffer.finalize(4., 6., gamma=.9, gae_lambda=1., max_episode_steps=600, failure_tail_steps=20, normalize_g_target=False)
    np.testing.assert_allclose(buffer.returns, [10., 100., 8.6], atol=1e-5)
    batch = buffer._make_batch([(0, 1)], 4, torch.device("cpu"))
    assert batch.valid_mask.tolist() == [[1, 0, 0, 0]]
    value1 = torch.zeros(1, 4, requires_grad=True)
    value2 = torch.zeros(1, 4, requires_grad=True)
    loss = clipped_twin_value_loss(value1, value2, value1.detach(), value2.detach(), batch.returns, .2, batch.valid_mask)
    loss.backward()
    assert torch.count_nonzero(value1.grad[:, 1:]) == 0
    assert torch.count_nonzero(value2.grad[:, 1:]) == 0


class FinalObservationEnv(gym.Env):
    observation_space = gym.spaces.Box(-1000., 1000., (85,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (2,), dtype=np.float32)

    def __init__(self, terminated):
        self.terminated = terminated

    def reset(self, **kwargs):
        return np.full(85, 100., np.float32), {}

    def step(self, action):
        return np.full(85, 7., np.float32), 1., self.terminated, not self.terminated, {"success": self.terminated}


@pytest.mark.parametrize("terminated", [False, True])
def test_trainer_uses_final_observation_and_pre_reset_recurrent_state(terminated):
    config = ExperimentConfig(device="cpu", model=ModelConfig(hidden_size=8, actor_hidden_size=8))
    config.ppo.rollout_steps = 2
    trainer = PPOTrainer(FinalObservationEnv(terminated), config)
    def forward(obs, state, starts):
        state_value = state.first[:, :1] * (1 - starts.float())
        value = obs[:, :, 0] + state_value
        return SimpleNamespace(mean=torch.zeros(1, 1, 2), log_std=torch.zeros(1, 1, 2), value1=value, value2=value,
                               diagnostics={}, state=RecurrentState(state.first + 2, state.second + 2))
    trainer.model.forward = forward
    buffer, _ = trainer.collect_rollout()
    expected = 1. if terminated else 1. + config.ppo.gamma * 9.
    np.testing.assert_allclose(buffer.returns, expected, atol=1e-5)


def test_action_initialization_and_checkpoint_roundtrip(tmp_path):
    config = ExperimentConfig(device="cpu", model=ModelConfig(hidden_size=8, actor_hidden_size=8, actor_mean_gain=.01,
                              log_std_init=-.5, log_std_max=0.))
    trainer = PPOTrainer(FinalObservationEnv(False), config)
    output = trainer.model(torch.randn(2, 3, 85))
    torch.testing.assert_close(output.log_std, torch.full((2, 3, 2), -.5))
    path = tmp_path / "model.pt"
    trainer.save(path)
    other = PPOTrainer(FinalObservationEnv(False), config)
    other.load(path)
    for first, second in zip(trainer.model.parameters(), other.model.parameters()):
        torch.testing.assert_close(first, second)
