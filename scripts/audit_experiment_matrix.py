from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from pidmppo.config import load_config
from pidmppo.evaluation import load_scenarios
from pidmppo.experiment import VARIANTS


REQUIRED_CORE = {
    "pidmppo",
    "lstm_ppo",
    "gru_ppo",
    "lstm_auxiliary",
    "lstm_dual_value",
    "matched_lstm",
    "matched_gru",
    "without_d",
    "without_attention",
    "without_auxiliary",
    "l_loss_only",
    "g_loss_only",
    "single_critic",
}
PRIMARY_SENSITIVITY = {
    "memory_bound_low",
    "memory_bound_high",
    "gate_temperature_low",
    "gate_temperature_high",
    "auxiliary_weight_low",
    "auxiliary_weight_high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit revision experiment matrix without training")
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment_matrix.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/training_readiness.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    config = load_config(raw["base_config"])
    scenarios = load_scenarios(config.evaluation.scenario_manifest)
    core = set(raw["core_variants"])
    sensitivity = {item["name"] for item in raw["sensitivity"]}
    seeds = [int(seed) for seed in raw["seeds"]]
    scenario_counts = Counter(item.map for item in scenarios)
    errors: list[str] = []
    warnings: list[str] = []
    missing_core = sorted(REQUIRED_CORE - core)
    if missing_core:
        errors.append(f"missing core variants: {missing_core}")
    unknown_core = sorted(core - set(VARIANTS))
    if unknown_core:
        errors.append(f"unknown core variants: {unknown_core}")
    if len(seeds) != 5 or len(set(seeds)) != 5:
        errors.append("exactly five unique training seeds are required")
    missing_sensitivity = sorted(PRIMARY_SENSITIVITY - sensitivity)
    if missing_sensitivity:
        errors.append(f"missing primary sensitivity cases: {missing_sensitivity}")
    expected_maps = set(config.evaluation.maps)
    if set(scenario_counts) != expected_maps:
        errors.append("scenario maps do not match evaluation.maps")
    if any(scenario_counts[name] != 25 for name in expected_maps):
        errors.append("each paper map must contain 25 fixed start-goal scenarios")
    if config.evaluation.episodes_per_map != 100:
        warnings.append("evaluation episodes per map differs from the planned 100")
    smoke_audit_path = Path("artifacts/smoke_training_audit.json")
    smoke_passed = False
    if smoke_audit_path.exists():
        smoke_passed = json.loads(smoke_audit_path.read_text(encoding="utf-8")).get("status") == "passed"
    mechanism_path = Path("configs/mechanism_scenarios.json")
    mechanism_ready = False
    if mechanism_path.exists():
        mechanism = load_scenarios(mechanism_path)
        mechanism_ready = bool(mechanism) and all(item.trap_region is not None for item in mechanism)
    generalization_path = Path("generated_maps/scenarios.json")
    generalization_ready = False
    randomized_scenarios = []
    if generalization_path.exists():
        randomized_scenarios = load_scenarios(generalization_path)
        generalization_ready = len(randomized_scenarios) == 60 and all(
            item.map_file is not None and Path(item.map_file).exists() for item in randomized_scenarios
        )
    remaining_gates = []
    if not smoke_passed:
        remaining_gates.append("short smoke training must pass before full jobs")
    if not mechanism_ready:
        remaining_gates.append("fixed trap mechanism scenario is missing or incomplete")
    if not generalization_ready:
        remaining_gates.append("fixed randomized-map zero-shot suite is missing or incomplete")
    status = "blocked" if errors else (
        "ready_for_full_training" if not remaining_gates else "ready_for_smoke_training"
    )
    payload = {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "training_seeds": seeds,
        "core_variants": sorted(core),
        "core_training_jobs": len(core) * len(seeds),
        "sensitivity_cases": len(raw["sensitivity"]),
        "sensitivity_training_jobs": len(raw["sensitivity"]) * len(seeds),
        "total_planned_training_jobs": (len(core) + len(raw["sensitivity"])) * len(seeds),
        "evaluation_maps": list(config.evaluation.maps),
        "episodes_per_map_per_checkpoint": config.evaluation.episodes_per_map,
        "fixed_scenarios_per_map": dict(sorted(scenario_counts.items())),
        "checkpoint_rule": config.evaluation.checkpoint_rule,
        "completed_gates": {
            "smoke_training": smoke_passed,
            "fixed_u_trap_mechanism": mechanism_ready,
            "randomized_zero_shot_suite": generalization_ready,
        },
        "randomized_scenarios": len(randomized_scenarios),
        "remaining_gates": remaining_gates,
        "declared_limitations": [
            "successful mechanism traces can only be selected after a trained policy succeeds",
            "map and robot assets are documented reconstructions rather than recovered originals",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
