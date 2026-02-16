"""Tests for MyoSkeleton Unitree-style velocity env config."""

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.config.myoskeleton_unitree.env_cfgs import (
  myoskeleton_unitree_flat_env_cfg,
)


def test_myoskeleton_unitree_flat_env_cfg() -> None:
  cfg = myoskeleton_unitree_flat_env_cfg()

  assert "robot" in cfg.scene.entities
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  assert len(robot.articulation.actuators) == 3

  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, JointPositionActionCfg)
  assert isinstance(action_cfg.scale, dict)
  assert len(action_cfg.scale) == 12

  assert cfg.viewer.body_name == "pelvis"
  assert cfg.scene.terrain.terrain_type == "plane"
