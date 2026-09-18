import torch
from torch import nn

from pidmppo.config import ModelConfig
from pidmppo.models.recurrent import PIDMEncoder
from pidmppo.models import ActorCritic


def test_pidm_preserves_temporal_gradient_and_bounded_memory():
    torch.manual_seed(2)
    encoder = PIDMEncoder(5, ModelConfig(hidden_size=16))
    observations = torch.randn(2, 6, 5, requires_grad=True)
    output, state = encoder.forward_sequence(observations)
    output[:, -1].sum().backward()
    assert observations.grad[:, 0].abs().sum() > 0
    assert torch.all(state.second <= 1.0 + 1e-6)
    assert torch.all(state.second >= -1.0 - 1e-6)
    assert encoder.last_attention is not None
    torch.testing.assert_close(
        encoder.last_attention.sum(dim=-1), torch.ones(2, 6), atol=1e-6, rtol=1e-6
    )


def test_without_d_has_zero_derivative_diagnostics():
    encoder = PIDMEncoder(5, ModelConfig(hidden_size=8, use_delta=False))
    encoder.forward_sequence(torch.randn(1, 4, 5))
    assert torch.count_nonzero(encoder.last_diagnostics["d"]) == 0


def test_actor_predicts_state_dependent_log_standard_deviation():
    model = ActorCritic(85, 2, ModelConfig(hidden_size=16, actor_hidden_size=16))
    assert isinstance(model.actor_log_std, nn.Linear)
    output = model(torch.randn(2, 3, 85))
    assert output.log_std.shape == (2, 3, 2)
