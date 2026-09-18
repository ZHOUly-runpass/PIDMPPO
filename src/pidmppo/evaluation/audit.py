from __future__ import annotations

from collections import defaultdict


def audit_evaluation_rows(
    rows: list[dict],
    *,
    required_seeds: int = 5,
    required_episodes_per_map_seed: int = 100,
) -> list[str]:
    """Return protocol violations that must be resolved before aggregation."""

    errors: list[str] = []
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    method_map_scenarios: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (str(row["method"]), str(row["map"]), str(row["training_seed"]))
        groups[key].append(row)
        method_map_scenarios[(key[0], key[1])].add(str(row["scenario_id"]))
        unique = (key[0], key[1], key[2], str(row["evaluation_seed"]))
        if unique in seen:
            errors.append(f"Duplicate evaluation row: {unique}")
        seen.add(unique)
    method_maps = {(method, map_name) for method, map_name, _ in groups}
    for method, map_name in sorted(method_maps):
        seeds = {seed for current_method, current_map, seed in groups if current_method == method and current_map == map_name}
        if len(seeds) < required_seeds:
            errors.append(f"{method}/{map_name} has {len(seeds)} seeds; requires {required_seeds}")
        for seed in seeds:
            count = len(groups[(method, map_name, seed)])
            if count < required_episodes_per_map_seed:
                errors.append(
                    f"{method}/{map_name}/seed={seed} has {count} episodes; "
                    f"requires {required_episodes_per_map_seed}"
                )
    for map_name in sorted({map_name for _, map_name in method_maps}):
        scenario_sets = {
            method: scenarios
            for (method, current_map), scenarios in method_map_scenarios.items()
            if current_map == map_name
        }
        if scenario_sets:
            reference_method = sorted(scenario_sets)[0]
            reference = scenario_sets[reference_method]
            for method, scenarios in scenario_sets.items():
                if scenarios != reference:
                    errors.append(
                        f"Scenario mismatch on {map_name}: {method} differs from {reference_method}"
                    )
    return errors
