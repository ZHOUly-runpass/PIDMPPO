from __future__ import annotations

from importlib.resources import files
import importlib.util
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pidmppo.config import load_config
from pidmppo.envs import MaplessNavigationEnv, load_map


@pytest.mark.parametrize("name", ["map1", "map2", "map3", "map4"])
def test_paper_maps_use_metric_segment_geometry(name: str) -> None:
    grid = load_map(name)
    assert grid.source == "manuscript_figure_reconstruction"
    assert grid.width == pytest.approx(6.5)
    assert grid.height == pytest.approx(4.0)
    assert grid.wall_thickness == pytest.approx(0.06)
    assert len(grid.segments) >= 4
    assert not grid.is_occupied(*grid.starts[0])
    assert not grid.is_occupied(*grid.goals[0])


def test_bundled_urdf_has_stable_wheel_joint_indices() -> None:
    resource = files("pidmppo").joinpath("assets/robots/turtlebot3_burger_sim.urdf")
    root = ET.fromstring(resource.read_text(encoding="utf-8"))
    joints = [joint.attrib["name"] for joint in root.findall("joint")]
    assert joints[:2] == ["left_wheel_joint", "right_wheel_joint"]


def test_pybullet_config_selects_bundled_robot() -> None:
    config = load_config("configs/paper_pybullet.yaml")
    assert config.env.backend == "pybullet"
    assert config.env.robot_urdf == "bundled:turtlebot3_burger_sim.urdf"
    assert config.env.left_wheel_joints == [0]
    assert config.env.right_wheel_joints == [1]


@pytest.mark.skipif(importlib.util.find_spec("pybullet") is None, reason="optional PyBullet dependency")
def test_pybullet_bundled_robot_smoke() -> None:
    config = load_config("configs/paper_pybullet.yaml")
    environment = MaplessNavigationEnv(config.env)
    try:
        observation, _ = environment.reset(seed=1)
        assert observation.shape == (85,)
        _, _, terminated, truncated, info = environment.step(
            np.asarray([0.0, 0.0], dtype=np.float32)
        )
        assert not terminated
        assert not truncated
        assert not info["collision"]
    finally:
        environment.close()
