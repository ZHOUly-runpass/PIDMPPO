from copy import deepcopy
import pytest

from pidmppo.evaluation.optimization import confirmation_gate, select_candidate


def result(success, collision=.1, efficiency=.5, seed=11, spin=.2, stagnation=.4):
    return {"success_rate": success, "collision_rate": collision, "successful_path_efficiency": efficiency,
            "training_seed": seed, "spin_fraction": spin, "stagnation_fraction": stagnation}


def test_screen_gate_and_predeclared_tie_breaks():
    results = {"A": result(.1), "B": result(.14), "C": result(.1), "D": result(.14)}
    assert select_candidate(results) is None
    results.update(B=result(.15, .2), C=result(.15, .1, .6), D=result(.15, .1, .7))
    assert select_candidate(results) == "D"


def test_confirm_gate_is_seed_paired_and_does_not_dispatch():
    first = [result(.2, seed=seed) for seed in (11, 22, 33)]
    second = [result(.35, seed=seed, spin=.1, stagnation=.3) for seed in (33, 11, 22)]
    gate = confirmation_gate(first, second)
    assert gate["status"] == "eligible_for_review"
    assert not gate["formal_training_dispatched"]
    second[0]["collision_rate"] = .4
    assert confirmation_gate(first, second)["status"] == "hold"
    second[0]["training_seed"] = 44
    with pytest.raises(ValueError, match="paired"):
        confirmation_gate(first, second)
