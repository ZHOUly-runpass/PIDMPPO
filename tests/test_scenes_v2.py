import json
from pathlib import Path

import numpy as np
import pytest

from pidmppo.config import EnvConfig
from pidmppo.envs import MaplessNavigationEnv
from pidmppo.envs.geometry import ClearanceSpace
from pidmppo.envs.maps import GridMap
from pidmppo.evaluation import load_scenarios
from pidmppo.evaluation.reference import validate_reference


def test_clearance_uses_physical_cell_faces_and_corners():
    occupied = np.zeros((8, 8), bool)
    occupied[3, 3] = True
    grid = GridMap(occupied, ((.75, .75),), ((3.25, 3.25),), .5)
    space = ClearanceSpace(grid, .15)
    assert not space.is_valid((1.4, 1.75))  # free centre, only .10m from wall
    assert space.is_valid((1.3, 1.75))
    assert not space.is_valid((1.4, 1.4))  # diagonal .1414m clearance
    assert not space.segment_valid((1., 1.75), (2.5, 1.75))
    assert space.segment_valid((1., 1.), (2.5, 1.))


def test_explicit_invalid_endpoints_fail_and_random_resets_are_safe():
    env = MaplessNavigationEnv(EnvConfig())
    for seed in [*range(20), 950]:  # seed 950 originally has no wide-enough long pair
        env.reset(seed=seed)
        assert env._space.connected(env.pose[:2], env.goal)
        assert not env.step(np.array([-1., 0.], np.float32))[4]["collision"]
    with pytest.raises(ValueError, match="clearance"):
        env.reset(options={"start": [0., 0.], "goal": [1., 1.]})
    env.close()


def test_curriculum_boundaries():
    env = MaplessNavigationEnv(EnvConfig(curriculum=True))
    for step, segments, minimum, maximum in ((0, 2, 1., 2.5), (49999, 2, 1., 2.5), (50000, 4, 2., 4.), (99999, 4, 2., 4.)):
        env.set_training_step(step)
        assert env.curriculum_parameters() == (segments, minimum, maximum)
        env.reset(seed=123)
        distance = np.linalg.norm(env.goal - env.pose[:2])
        assert minimum <= distance <= maximum
    env.set_training_step(100000)
    assert env.curriculum_parameters()[0] == 8
    env.close()


def test_committed_v2_assets_have_expected_independent_tasks():
    root = Path(__file__).resolve().parents[1] / "configs/scenarios_v2"
    fixed = load_scenarios(root / "fixed.json")
    validation = load_scenarios(root / "validation.json")
    random = load_scenarios(root / "random.json")
    assert len(fixed) == 400 and len(validation) == 100 and len(random) == 60
    for name in ("map1", "map2", "map3", "map4"):
        tasks = [s for s in fixed if s.map == name]
        assert len({(s.start, s.goal) for s in tasks}) == 100
    assert len({s.map_file for s in validation}) == 100
    assert all(2 <= s.reference_path_length <= 8 for s in random)
    for scene in (fixed[0], fixed[100], fixed[200], fixed[300], validation[0], random[0]):
        env = MaplessNavigationEnv(EnvConfig(map_name=scene.map, map_file=scene.map_file, randomize_obstacles=False))
        result = validate_reference(env, scene.start, scene.goal, scene.theta)
        assert result["reference_steps"] <= 540
        env.close()
