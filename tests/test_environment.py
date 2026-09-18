import numpy as np

from pidmppo.config import EnvConfig
from pidmppo.envs import MaplessNavigationEnv
from pidmppo.envs.backends import GridBackend
from pidmppo.envs.maps import MAPS, generate_corridor_map, generate_random_map, load_map


def test_observation_is_policy_only_and_has_action_history():
    env = MaplessNavigationEnv(EnvConfig(randomize_start_goal=False))
    observation, info = env.reset(seed=7)
    assert observation.shape == (85,)
    assert np.all(observation >= -1.0) and np.all(observation <= 1.0)
    assert set(info) >= {"distance_to_goal", "pose", "goal"}
    assert np.all(observation[:64] >= 0.0) and np.all(observation[:64] <= 1.0)
    action = np.asarray([0.25, -0.5], dtype=np.float32)
    next_observation, _, _, _, _ = env.step(action)
    np.testing.assert_allclose(next_observation[-2:], [0.625, -0.5])


def test_reset_is_seed_reproducible():
    env = MaplessNavigationEnv(EnvConfig())
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=123)
    np.testing.assert_allclose(first, second)


def test_all_declared_maps_are_rectangular_and_loadable():
    for name in MAPS:
        grid = load_map(name)
        assert grid.occupied.ndim == 2
        assert grid.starts and grid.goals


def test_random_map_generation_is_reproducible_and_connected():
    first = generate_random_map(seed=9, obstacle_density=0.15, trap_geometry="u")
    second = generate_random_map(seed=9, obstacle_density=0.15, trap_geometry="u")
    np.testing.assert_array_equal(first.occupied, second.occupied)
    assert first.starts == second.starts and first.goals == second.goals


def test_randomized_training_corridors_are_seed_reproducible():
    first = generate_corridor_map(seed=77)
    second = generate_corridor_map(seed=77)
    np.testing.assert_array_equal(first.occupied, second.occupied)
    assert not first.is_occupied(*first.starts[0])
    assert not first.is_occupied(*first.goals[0])


def test_forward_only_action_mapping_has_zero_speed_at_minus_one():
    env = MaplessNavigationEnv(EnvConfig(randomize_start_goal=False, allow_reverse=False))
    env.reset(seed=3, options={"theta": 0.0})
    before = env.pose
    env.step(np.asarray([-1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(env.pose[:2], before[:2])


def test_vectorized_lidar_matches_scalar_reference():
    grid = load_map("map1")
    backend = GridBackend(grid, robot_radius=0.105)
    backend.reset(np.asarray(grid.starts[0], dtype=np.float32), theta=0.37)
    actual = backend.lidar(beams=64, minimum=0.12, maximum=3.5)
    angles = backend.pose[2] + np.linspace(-np.pi, np.pi, 64, endpoint=False)
    increments = np.arange(0.12, 3.5 + 0.01, 0.02, dtype=np.float32)
    expected = np.full(64, 3.5, dtype=np.float32)
    for index, angle in enumerate(angles):
        for distance in increments:
            x = float(backend.pose[0] + distance * np.cos(angle))
            y = float(backend.pose[1] + distance * np.sin(angle))
            if grid.is_occupied(x, y):
                expected[index] = distance
                break
    np.testing.assert_allclose(actual, np.clip(expected, 0.12, 3.5))
