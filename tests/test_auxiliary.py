import numpy as np
import torch

from pidmppo.algorithms.auxiliary import build_l_targets, masked_mse


def test_l_targets_mask_failures_and_unfinished_fragments():
    dones = np.array([False, False, True, False, False, False, True, False])
    successes = np.array([False, False, True, False, False, False, False, False])
    target, mask = build_l_targets(
        dones, successes, max_episode_steps=10, failure_tail_steps=2
    )
    np.testing.assert_allclose(target[:3], [0.2, 0.1, 0.0])
    np.testing.assert_allclose(mask[:3], 1.0)
    np.testing.assert_allclose(mask[3:5], 0.0)
    np.testing.assert_allclose(target[5:7], 1.0)
    np.testing.assert_allclose(mask[5:7], 1.0)
    assert mask[7] == 0.0


def test_l_loss_uses_half_batch_expectation_with_masked_terms():
    prediction = torch.tensor([0.0, 0.0, 0.0, 0.0])
    target = torch.ones(4)
    supervision_mask = torch.tensor([1.0, 0.0, 0.0, 0.0])
    valid_mask = torch.ones(4)
    loss = masked_mse(
        prediction,
        target,
        supervision_mask,
        normalizer_mask=valid_mask,
        scale=0.5,
    )
    assert loss.item() == 0.125
