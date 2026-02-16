"""MyoSkeleton Unitree-style velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  MYOSKELETON_UNITREE_ACTION_SCALE,
  get_myoskeleton_unitree_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.config.myoskeleton.rl_cfg import myoskeleton_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def myoskeleton_unitree_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a flat-terrain velocity task with Unitree-style MyoSkeleton control."""
  cfg = make_velocity_env_cfg()

  # MyoSkeleton morphology produces more contacts than quadruped defaults.
  cfg.sim.njmax = 1000
  cfg.sim.nconmax = 100

  cfg.scene.entities = {"robot": get_myoskeleton_unitree_robot_cfg()}

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(calcn_l|calcn_r)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (feet_ground_cfg,)

  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, JointPositionActionCfg)
  action_cfg.scale = MYOSKELETON_UNITREE_ACTION_SCALE

  cmd_cfg = cfg.commands["twist"]
  assert isinstance(cmd_cfg, UniformVelocityCommandCfg)
  cmd_cfg.ranges.lin_vel_x = (-1.0, 1.0)
  cmd_cfg.ranges.lin_vel_y = (-0.5, 0.5)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r"^(bofoot[12]?_[lr]_coll|foot[123]?_[lr]_coll)$"
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)

  site_names = ("l_foot_touch", "r_foot_touch")
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names
  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.viewer.body_name = "pelvis"
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Flat terrain has no heightfield: remove raycast height scan wiring.
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum on flat terrain.
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Keep parity with other flat velocity tasks in play mode.
    if "command_vel" in cfg.curriculum:
      del cfg.curriculum["command_vel"]

  return cfg


__all__ = ["myoskeleton_unitree_flat_env_cfg", "myoskeleton_ppo_runner_cfg"]
