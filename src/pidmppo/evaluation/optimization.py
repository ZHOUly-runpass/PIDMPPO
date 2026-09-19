"""Predeclared selection rules; only training-distribution validation is read."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def select_candidate(results: dict[str, dict], improvement: float = .05) -> str | None:
    choices = [key for key in ("B", "C", "D") if results[key]["success_rate"] + 1e-12 >= results["A"]["success_rate"] + improvement]
    return max(choices, key=lambda key: (results[key]["success_rate"], -results[key]["collision_rate"], results[key]["successful_path_efficiency"])) if choices else None


def confirmation_gate(baseline: list[dict], candidate: list[dict]) -> dict:
    if len(baseline) != 3 or len(candidate) != 3:
        raise ValueError("Confirmation requires exactly three matched training seeds")
    baseline = sorted(baseline, key=lambda x: x["training_seed"])
    candidate = sorted(candidate, key=lambda x: x["training_seed"])
    if [x["training_seed"] for x in baseline] != [x["training_seed"] for x in candidate]:
        raise ValueError("Confirmation seeds must be paired")
    first = np.array([r["success_rate"] for r in baseline])
    second = np.array([r["success_rate"] for r in candidate])
    differences = second - first
    mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
    # 'Clearly reduced' is predeclared as >= 10% relative reduction in the
    # combined spin/stagnation fraction, with neither component increasing.
    base_bad = mean(baseline, "spin_fraction") + mean(baseline, "stagnation_fraction")
    new_bad = mean(candidate, "spin_fraction") + mean(candidate, "stagnation_fraction")
    checks = {
        "mean_gain_at_least_10pp": bool(differences.mean() >= .10 - 1e-12),
        "at_least_two_seeds_improve": bool(np.sum(differences > 1e-12) >= 2),
        "worst_seed_at_least_10pct": bool(second.min() >= .10 - 1e-12),
        "mean_at_least_30pct": bool(second.mean() >= .30 - 1e-12),
        "collision_not_increased": mean(candidate, "collision_rate") <= mean(baseline, "collision_rate") + 1e-12,
        "spin_not_increased": mean(candidate, "spin_fraction") <= mean(baseline, "spin_fraction") + 1e-12,
        "stagnation_not_increased": mean(candidate, "stagnation_fraction") <= mean(baseline, "stagnation_fraction") + 1e-12,
        "bad_behavior_reduced": bool(new_bad <= .9 * base_bad + 1e-12),
    }
    rng = np.random.default_rng(20260919)
    boot = rng.choice(differences, size=(10000, 3), replace=True).mean(1)
    return {"status": "eligible_for_review" if all(checks.values()) else "hold", "checks": checks,
            "mean_success_baseline": float(first.mean()), "mean_success_candidate": float(second.mean()),
            "paired_seed_gains": differences.tolist(), "paired_seed_mean_gain_ci95": np.quantile(boot, [.025, .975]).tolist(),
            "ci_note": "Exploratory paired training-seed bootstrap (n=3); not an episode-level independent-sample CI.",
            "formal_training_dispatched": False}


def training_health(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"approx_kl", "clip_fraction", "epochs_completed", "grad_norm_pre_clip", "value1_explained_variance",
                "value2_explained_variance", "action_saturation_rate", "attention_p_mean", "gate_saturation_mean",
                "memory_norm_mean", "gaussian_entropy", "entropy", "collisions", "timeouts", "reward_progress_mean",
                "encoder_actor_grad_norm", "encoder_critic_grad_norm", "encoder_auxiliary_grad_norm"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("Training health metrics are missing")
    for row in rows:
        for key, value in row.items():
            if key != "implementation_version" and not math.isfinite(float(value)):
                raise ValueError(f"Non-finite health metric {key}")
    measured = [r for r in rows if float(r["encoder_grad_measured"]) == 1.]
    ratios = [float(r["encoder_critic_grad_norm"]) / max(1e-12, float(r["encoder_actor_grad_norm"])) for r in measured]
    tail = rows[-min(10, len(rows)):]
    keys = sorted(required | {"steps_per_second", "spin_fraction", "stagnation_fraction", "success_rate", "encoder_grad_measured"})
    return {"status": "passed", "updates": len(rows), "gradient_measurements": len(measured),
            "critic_actor_gradient_ratios": ratios, "value_gradient_warning": bool(ratios and sum(r > 10 for r in ratios) > len(ratios) / 2),
            "last10_update_means": {key: float(np.mean([float(r[key]) for r in tail])) for key in keys},
            "warning_note": "A value-gradient warning is diagnostic only; value_coef is NOT automatically changed."}
