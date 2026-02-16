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




# Unitree-inspired posture tolerance schedule mapped to MyoSkeleton controlled joints.
_MYOSKEL_UNITREE_POSE_STD_STANDING = {
  r"(hip|knee|ankle|subtalar|L5_S1|shoulder|elbow|pro_sup|flexion|deviation).*": 0.05
}

# Values mirror Unitree G1 lower/waist/upper-body magnitudes mapped to Myo names.
_MYOSKEL_UNITREE_POSE_STD_WALKING = {
  # Lower body.
  r"hip_flexion_[lr]": 0.3,
  r"hip_adduction_[lr]": 0.15,
  r"hip_rotation_[lr]": 0.15,
  r"knee_angle_[lr]": 0.35,
  r"ankle_angle_[lr]": 0.25,
  r"subtalar_angle_[lr]": 0.1,
  # Waist.
  r"L5_S1_axial_rotation": 0.2,
  r"L5_S1_Lat_Bending": 0.08,
  r"L5_S1_Flex_Ext": 0.1,
  # Arms.
  r"shoulder_elv_[lr]": 0.15,
  r"shoulder1_r2_[lr]": 0.15,
  r"shoulder_rot_[lr]": 0.1,
  r"elbow_flex_[lr]": 0.15,
  r"pro_sup(_l)?$": 0.3,
  r"flexion(_l|_r)?$": 0.3,
  r"deviation(_l)?$": 0.3,
}

_MYOSKEL_UNITREE_POSE_STD_RUNNING = {
  # Lower body.
  r"hip_flexion_[lr]": 0.5,
  r"hip_adduction_[lr]": 0.2,
  r"hip_rotation_[lr]": 0.2,
  r"knee_angle_[lr]": 0.6,
  r"ankle_angle_[lr]": 0.35,
  r"subtalar_angle_[lr]": 0.15,
  # Waist.
  r"L5_S1_axial_rotation": 0.3,
  r"L5_S1_Lat_Bending": 0.08,
  r"L5_S1_Flex_Ext": 0.2,
  # Arms.
  r"shoulder_elv_[lr]": 0.5,
  r"shoulder1_r2_[lr]": 0.2,
  r"shoulder_rot_[lr]": 0.15,
  r"elbow_flex_[lr]": 0.35,
  r"pro_sup(_l)?$": 0.3,
  r"flexion(_l|_r)?$": 0.3,
  r"deviation(_l)?$": 0.3,
}

_MYOSKEL_UNITREE_CONTROLLED_JOINT_REGEX = (
  r"hip_flexion_[lr]",
  r"hip_adduction_[lr]",
  r"hip_rotation_[lr]",
  r"knee_angle_[lr]",
  r"ankle_angle_[lr]",
  r"subtalar_angle_[lr]",
  r"L5_S1_(Flex_Ext|Lat_Bending|axial_rotation)",
  r"shoulder_elv_[lr]",
  r"shoulder1_r2_[lr]",
  r"shoulder_rot_[lr]",
  r"elbow_flex_[lr]",
  r"pro_sup(_l)?$",
  r"flexion(_l|_r)?$",
  r"deviation(_l)?$",
)


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

  # Restrict posture reward to the 12 controlled joints and use Unitree-like std schedule.
  cfg.rewards["pose"].params["asset_cfg"].joint_names = _MYOSKEL_UNITREE_CONTROLLED_JOINT_REGEX
  cfg.rewards["pose"].params["std_standing"] = _MYOSKEL_UNITREE_POSE_STD_STANDING
  cfg.rewards["pose"].params["std_walking"] = _MYOSKEL_UNITREE_POSE_STD_WALKING
  cfg.rewards["pose"].params["std_running"] = _MYOSKEL_UNITREE_POSE_STD_RUNNING

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
