"""MyoLeg velocity env config: flat terrain, tendon effort, posture from keyframe."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import TendonEffortActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .myoleg_robot_cfg import get_myoleg_robot_cfg


def myoleg_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """MyoLeg flat velocity: keyframe posture, randomized at reset; tendon effort action."""
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_myoleg_robot_cfg()}

  # Terrain scan frame: base config uses name="" (invalid "robot/"); use pelvis.
  (terrain_scan,) = (
    s for s in cfg.scene.sensors if getattr(s, "name", None) == "terrain_scan"
  )
  terrain_scan_pelvis = replace(
    terrain_scan,
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
  )
  # Feet–ground contact sensor (same subtree pattern as MyoSkeleton: calcn includes toes).
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(calcn_l|calcn_r)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (terrain_scan_pelvis, feet_ground_cfg)

  # Replace joint_pos with tendon effort (muscle activation).
  cfg.actions = {
    "tendon_effort": TendonEffortActionCfg(
      entity_name="robot",
      actuator_names=(r".*_tendon",),
      scale=1.0,
    ),
  }

  # Base velocity from entity state (no IMU in MyoLeg XML).
  cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
    func=mdp.base_lin_vel,
    params={"asset_cfg": SceneEntityCfg("robot")},
    noise=cfg.observations["actor"].terms["base_lin_vel"].noise,
  )
  cfg.observations["actor"].terms["base_ang_vel"] = ObservationTermCfg(
    func=mdp.base_ang_vel,
    params={"asset_cfg": SceneEntityCfg("robot")},
    noise=cfg.observations["actor"].terms["base_ang_vel"].noise,
  )
  cfg.observations["critic"].terms["base_lin_vel"] = cfg.observations["actor"].terms[
    "base_lin_vel"
  ]
  cfg.observations["critic"].terms["base_ang_vel"] = cfg.observations["actor"].terms[
    "base_ang_vel"
  ]

  # Muscle state and COM (myoLegWalk-style observations).
  cfg.observations["actor"].terms["tendon_length"] = ObservationTermCfg(
    func=mdp.tendon_length,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.observations["actor"].terms["tendon_velocity"] = ObservationTermCfg(
    func=mdp.tendon_velocity,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.observations["actor"].terms["actuator_force"] = ObservationTermCfg(
    func=mdp.actuator_force,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.observations["actor"].terms["com_pos"] = ObservationTermCfg(
    func=mdp.com_pos_w,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.observations["actor"].terms["com_lin_vel"] = ObservationTermCfg(
    func=mdp.com_lin_vel_w,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  for key in [
    "tendon_length",
    "tendon_velocity",
    "actuator_force",
    "com_pos",
    "com_lin_vel",
  ]:
    cfg.observations["critic"].terms[key] = cfg.observations["actor"].terms[key]

  # No height scan on flat ground.
  if "height_scan" in cfg.observations["actor"].terms:
    del cfg.observations["actor"].terms["height_scan"]
  if "height_scan" in cfg.observations["critic"].terms:
    del cfg.observations["critic"].terms["height_scan"]

  # Foot height: use MyoLeg touch sites.
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = (
    "l_foot_touch",
    "r_foot_touch",
  )

  # Flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  if "terrain_levels" in cfg.curriculum:
    del cfg.curriculum["terrain_levels"]

  # Curriculum expects "step" not "iteration" (base config uses iteration).
  if "command_vel" in cfg.curriculum:
    for s in cfg.curriculum["command_vel"].params.get("velocity_stages", []):
      if "iteration" in s:
        s["step"] = s.pop("iteration")

  # Same rewards as MyoSkeleton flat velocity (where applicable).
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = ".*_col"

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.rewards["track_linear_velocity"].params["std"] = 0.5
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].params["std"] = 0.5
  cfg.rewards["track_angular_velocity"].weight = 2.0

  site_names = ("l_foot_touch", "r_foot_touch")
  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  clean_joints_regex = r"^(?!.*(translation|rotation[2-3]|beta|Abs_t|Abs_r)).*"
  cfg.rewards["pose"].params["asset_cfg"].joint_names = (clean_joints_regex,)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {".*": 0.2}
  cfg.rewards["pose"].params["std_running"] = {".*": 0.4}
  cfg.rewards["pose"].weight = 2.0

  cfg.rewards["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=10.0)
  cfg.rewards["upright"].weight = 5.0
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["action_rate_l2"].weight = -0.01
  # No root_angmom sensor in MyoLeg; keep angular_momentum at 0.
  cfg.rewards["angular_momentum"].weight = 0.0

  # Cyclic hip flexion (myoLegWalk-v0 style): encourage phase-based gait pattern.
  cfg.rewards["cyclic_hip"] = RewardTermCfg(
    func=mdp.cyclic_hip_flexion_penalty,
    weight=-0.5,
    params={
      "hip_period": 100,
      "amplitude": 0.8,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=("hip_flexion_l", "hip_flexion_r"),
        preserve_order=True,
      ),
    },
  )

  # MyoLeg has no self_collision sensor; leave contact-dependent rewards zero.
  for name in [
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    # "soft_landing",
  ]:
    if name in cfg.rewards:
      cfg.rewards[name].weight = 2.0
  cfg.rewards["soft_landing"].weight = 0.002

  # Same terminations as MyoSkeleton flat; min height 0.8 for parity with myoLegWalk-v0.
  cfg.terminations["fell_over"].params["limit_angle"] = 1.0
  cfg.terminations["base_height"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.8}
  )

  # Posture from keyframe; randomize around it (same range as MyoSkeleton).
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

  cfg.episode_length_s = 10.0
  cfg.viewer.body_name = "pelvis"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    if "command_vel" in cfg.curriculum:
      del cfg.curriculum["command_vel"]
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
