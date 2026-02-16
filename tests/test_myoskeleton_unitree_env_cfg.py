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
  assert len(robot.articulation.actuators) == 6

  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, JointPositionActionCfg)
  assert isinstance(action_cfg.scale, dict)
  assert len(action_cfg.scale) == 29

  assert cfg.viewer.body_name == "pelvis"
  assert cfg.scene.terrain.terrain_type == "plane"


def test_myoskeleton_unitree_flat_env_cfg_removes_height_scan() -> None:
  cfg = myoskeleton_unitree_flat_env_cfg()

  sensor_names = tuple(sensor.name for sensor in (cfg.scene.sensors or ()))
  assert "terrain_scan" not in sensor_names
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "height_scan" not in cfg.observations["critic"].terms
  assert "terrain_levels" not in cfg.curriculum


def test_myoskeleton_unitree_flat_env_cfg_play_removes_command_curriculum() -> None:
  cfg = myoskeleton_unitree_flat_env_cfg(play=True)
  assert "command_vel" not in cfg.curriculum


def test_myoskeleton_unitree_flat_env_cfg_sets_foot_sites() -> None:
  cfg = myoskeleton_unitree_flat_env_cfg()
  site_names = cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names
  assert site_names == ("l_foot_touch", "r_foot_touch")

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    reward_sites = cfg.rewards[reward_name].params["asset_cfg"].site_names
    assert reward_sites == ("l_foot_touch", "r_foot_touch")


def test_myoskeleton_unitree_flat_env_cfg_contact_budget() -> None:
  cfg = myoskeleton_unitree_flat_env_cfg()
  assert cfg.sim.njmax == 1000
  assert cfg.sim.nconmax == 100


def test_myoskeleton_unitree_pose_std_dicts_are_populated_and_consistent() -> None:
  cfg = myoskeleton_unitree_flat_env_cfg()
  pose_params = cfg.rewards["pose"].params

  standing = pose_params["std_standing"]
  walking = pose_params["std_walking"]
  running = pose_params["std_running"]

  assert len(standing) > 0
  assert len(walking) > 0
  assert len(running) > 0

  # Walking/running should target the same mapped joint groups.
  assert set(walking.keys()) == set(running.keys())

  # Standing should match the mapped controlled-joint family.
  joint_patterns = pose_params["asset_cfg"].joint_names
  assert r"hip_flexion_[lr]" in joint_patterns
  assert r"subtalar_angle_[lr]" in joint_patterns
