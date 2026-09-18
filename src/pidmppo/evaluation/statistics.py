from __future__ import annotations

from collections import defaultdict
from statistics import NormalDist
from typing import Iterable

import numpy as np


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return float(center - margin), float(center + margin)


def bootstrap_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
    statistic=np.mean,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    estimates = np.asarray([statistic(array[index]) for index in indices])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def aggregate_results(
    rows: list[dict],
    *,
    group_by: tuple[str, ...] = ("method", "map"),
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_by)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        successes = np.asarray([_as_bool(row["success"]) for row in group], dtype=np.float64)
        success_low, success_high = wilson_interval(int(successes.sum()), len(successes), confidence)
        collisions = np.asarray(
            [_as_bool(row.get("collision", False)) for row in group], dtype=np.float64
        )
        timeouts = np.asarray(
            [_as_bool(row.get("timeout", False)) for row in group], dtype=np.float64
        )
        collision_low, collision_high = wilson_interval(
            int(collisions.sum()), len(collisions), confidence
        )
        timeout_low, timeout_high = wilson_interval(int(timeouts.sum()), len(timeouts), confidence)
        by_seed: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            by_seed[str(row["training_seed"])].append(row)
        seed_success = np.asarray(
            [np.mean([_as_bool(row["success"]) for row in seed_rows]) for seed_rows in by_seed.values()]
        )
        seed_success_low, seed_success_high = bootstrap_interval(
            seed_success,
            confidence=confidence,
            samples=bootstrap_samples,
            seed=seed,
        )
        summary = {name: value for name, value in zip(group_by, key)}
        summary.update(
            episodes=len(group),
            training_seeds=len(by_seed),
            success_rate=float(successes.mean()),
            success_ci_low=success_low,
            success_ci_high=success_high,
            collision_rate=float(collisions.mean()),
            collision_ci_low=collision_low,
            collision_ci_high=collision_high,
            timeout_rate=float(timeouts.mean()),
            timeout_ci_low=timeout_low,
            timeout_ci_high=timeout_high,
            success_seed_mean=float(seed_success.mean()),
            success_seed_std=float(seed_success.std(ddof=1)) if len(seed_success) > 1 else 0.0,
            success_seed_median=float(np.median(seed_success)),
            success_seed_q25=float(np.quantile(seed_success, 0.25)),
            success_seed_q75=float(np.quantile(seed_success, 0.75)),
            success_seed_ci_low=seed_success_low,
            success_seed_ci_high=seed_success_high,
        )
        for metric in ("steps", "episode_return", "path_length", "elapsed_seconds"):
            values = np.asarray([float(row[metric]) for row in group])
            seed_means = np.asarray(
                [np.mean([float(row[metric]) for row in seed_rows]) for seed_rows in by_seed.values()]
            )
            low, high = bootstrap_interval(
                seed_means,
                confidence=confidence,
                samples=bootstrap_samples,
                seed=seed,
            )
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            summary[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
            summary[f"{metric}_seed_mean"] = float(seed_means.mean())
            summary[f"{metric}_seed_std"] = (
                float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}
