import torch

from pidmppo.algorithms.ppo import clipped_twin_value_loss


def test_both_critics_receive_gradients():
    value1 = torch.tensor([[0.2, 0.4]], requires_grad=True)
    value2 = torch.tensor([[0.3, 0.1]], requires_grad=True)
    old1 = value1.detach().clone()
    old2 = value2.detach().clone()
    returns = torch.ones_like(value1)
    mask = torch.ones_like(value1)
    loss = clipped_twin_value_loss(value1, value2, old1, old2, returns, 0.2, mask)
    loss.backward()
    assert value1.grad is not None and value1.grad.abs().sum() > 0
    assert value2.grad is not None and value2.grad.abs().sum() > 0
