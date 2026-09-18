from __future__ import annotations

import numpy as np


def build_l_targets(
    dones: np.ndarray,
    successes: np.ndarray,
    *,
    max_episode_steps: int,
    failure_tail_steps: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Build learnability targets without leaking unfinished rollout fragments.

    Successful trajectories receive normalized remaining-time labels. For failed
    trajectories only their final K steps are supervised with target one. A
    trajectory that does not finish inside the rollout remains masked.
    """

    dones = np.asarray(dones, dtype=bool)
    successes = np.asarray(successes, dtype=bool)
    if dones.shape != successes.shape:
        raise ValueError("dones and successes must have identical shapes")
    targets = np.zeros(dones.shape, dtype=np.float32)
    mask = np.zeros(dones.shape, dtype=np.float32)
    episode_start = 0
    for episode_end in np.flatnonzero(dones):
        if successes[episode_end]:
            indices = np.arange(episode_start, episode_end + 1)
            targets[indices] = (episode_end - indices) / float(max_episode_steps)
            mask[indices] = 1.0
        else:
            tail_start = max(episode_start, episode_end - failure_tail_steps + 1)
            targets[tail_start : episode_end + 1] = 1.0
            mask[tail_start : episode_end + 1] = 1.0
        episode_start = episode_end + 1
    return targets, mask


def masked_mse(
    prediction,
    target,
    mask,
    normalizer_mask=None,
    scale: float = 1.0,
):
    squared = (prediction - target).pow(2) * mask
    denominator = mask if normalizer_mask is None else normalizer_mask
    return scale * squared.sum() / denominator.sum().clamp_min(1.0)
